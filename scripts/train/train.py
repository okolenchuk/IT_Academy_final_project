import hashlib
import itertools
import math
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer

from scripts.train.datasets import *


def main(args):
    update_vars('trained_model_dir', args.output_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
    )

    pipeline = None
    class_images_dir = Path(args.class_data_dir)
    class_images_dir.mkdir(parents=True, exist_ok=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < args.num_class_images:
        torch_dtype = torch.float16
        if pipeline is None:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                                  subfolder=None,
                                                  revision=None,
                                                  torch_dtype=torch_dtype),
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision='fp16')
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to(accelerator.device)

        num_new_images = args.num_class_images - cur_class_images

        sample_dataset = PromptDataset(args.class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=1)

        sample_dataloader = accelerator.prepare(sample_dataloader)

        with torch.autocast("cuda"), torch.inference_mode():
            for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    update_vars('model_name', args.pretrained_model_name_or_path)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision="fp16",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision="fp16",
    )
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        subfolder="vae",
        revision="fp16",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision="fp16",
        torch_dtype=torch.float32
    )
    unet.enable_gradient_checkpointing()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (itertools.chain(unet.parameters(), text_encoder.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08, )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    update_vars('instance_prompt', args.instance_prompt)
    train_dataset = TrainPhotoDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=512,
        num_class_images=args.num_class_images
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, pin_memory=True
    )

    weight_dtype = torch.float16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if not args.not_cache_latents:
        latents_cache = []
        text_encoder_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                text_encoder_cache.append(batch["input_ids"])
        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=args.max_train_steps,
    )

    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, text_encoder,
                                                                                        optimizer, train_dataloader,
                                                                                        lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    # Train!
    loss_logs = []

    def save_weights(step):
        # Create the pipeline using the trained modules and save it.
        if accelerator.is_main_process:
            text_enc_model = accelerator.unwrap_model(text_encoder)
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                      clip_sample=False, set_alpha_to_one=False)
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_enc_model,
                vae=AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse",
                    subfolder=None,
                    revision=None,
                ),
                safety_checker=None,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision="fp16",
            )
            save_dir = os.path.join(args.output_dir)
            pipeline.save_pretrained(save_dir)

            if args.save_sample_prompt is not None:
                pipeline = pipeline.to(accelerator.device)
                g_cuda = torch.Generator(device=accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                sample_dir = os.path.join(save_dir, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                with torch.autocast("cuda"), torch.inference_mode():
                    for i in tqdm(range(args.n_save_sample), desc="Generating samples"):
                        images = pipeline(
                            args.save_sample_prompt,
                            negative_prompt=args.save_sample_negative_prompt,
                            guidance_scale=args.save_guidance_scale,
                            num_inference_steps=args.save_infer_steps,
                            generator=g_cuda
                        ).images
                        images[0].save(os.path.join(sample_dir, f"{i}.png"))
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"[*] Weights saved at {save_dir}")

            plt.tight_layout()
            plt.savefig('grid.png', dpi=72)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    loss_avg = AverageMeter()
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    if not args.not_cache_latents:
                        latent_dist = batch[0][0]
                    else:
                        latent_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
                    latents = latent_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with text_enc_context:
                    if not args.not_cache_latents:
                        encoder_hidden_states = text_encoder(batch[0][1])[0]
                    else:
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                noise, noise_prior = torch.chunk(noise, 2, dim=0)

                # Compute instance loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                # Compute prior loss
                prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss


                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.detach_(), bsz)

            logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            loss_logs.append(['Average loss on {}'.format(global_step), loss_avg.avg.item()])

            progress_bar.update(1)
            global_step += 1

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    update_vars('logs', loss_logs)
    save_weights(global_step)

    accelerator.end_training()





