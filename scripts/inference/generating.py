import regex as re
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import random
from prompt_selection import *
from pathlib import Path


def generate_prompt_images(prompt: str, model_path, num_samples: int = 2,
                           guidance_scale=7.5, num_inference_steps=100,
                           save_path=r'/result'):

    scheduler = DDIMScheduler(beta_start=0.00085,
                              beta_end=0.012,
                              beta_schedule="scaled_linear",
                              clip_sample=False,
                              set_alpha_to_one=False)
    pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                   scheduler=scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=torch.float16).to("cuda")

    height = 512
    width = 512
    g_cuda = torch.Generator(device='cuda')
    with autocast(g_cuda), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images
    print(prompt)
    name = re.sub(r'[^\w]', ' ', prompt)
    c = 0

    if not Path(save_path).exists():
        Path(save_path).mkdir()

    save_path = str(Path(save_path).joinpath('{}.png'))

    for img in images:
        img.save(save_path.format(name[:50] + '_' + str(c + 1)))
        c += 1


def generate_n_images(class_name, instance_name: str, save_path, num: int = 10):
    list_prompts = class_prompts(class_name)
    for i in range(num):
        prompt = random.choice[list_prompts].replace('*', instance_name)
        generate_prompt_images(prompt, save_path, num_samples=1)


def generate_image_with_random_lexica_prompt(word='', num_samples: int = 2):
    prompt = random_prompt(word=word)
    generate_prompt_images(prompt, num_samples)
