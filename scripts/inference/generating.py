import regex as re
import torch
from torch import autocast
import random
from prompt_selection import *


def generate_prompt_images(prompt: str, pipe, num_samples: int = 2,           #change pipe
                           guidance_scale=7.5, num_inference_steps=100,
                           save_path=r'/result/{}.png'):                     #save_path
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
    for img in images:
        img.save(save_path.format(name[:50] + '_' + str(c + 1)))
        c += 1


def generate_n_images(category, class_name: str, num: int = 10):
    list_prompts = class_prompts(category)
    for i in range(num):
        prompt = random.choice[list_prompts].replace('*', class_name)
        generate_prompt_images(prompt, num_samples=1)


def generate_image_with_random_lexica_prompt(word='', num_samples: int = 2):
    prompt = random_prompt(word=word)
    generate_prompt_images(prompt, num_samples)
