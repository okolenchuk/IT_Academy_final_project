import regex as re
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from random_word import RandomWords
import requests
import json
import random
from pathlib import Path
from tqdm.auto import trange



prompts_cat = {'woman': r'female.txt',
               'man': r'male.txt'}


def random_prompt(word='') -> str:
    if word == '':
        word = RandomWords().get_random_word()
    url = r'https://lexica.art/api/v1/search?q=' + word

    raw_prompt = requests.get(url=url).text
    prompts = json.loads(raw_prompt)['images']

    list_prompt = [text['prompt'] for text in prompts]
    prompt = random.choice(list_prompt)
    return prompt


def class_prompts(category) -> list:
    prompts_file = str(Path('IT_Academy_final_project/prompts').joinpath(prompts_cat[category]))
    with open(prompts_file, 'r') as prompts:
        list_prompts = prompts.readlines()
    return list_prompts


def create_pipe(model_path):
    scheduler = DDIMScheduler(beta_start=0.00085,
                              beta_end=0.012,
                              beta_schedule="scaled_linear",
                              clip_sample=False,
                              set_alpha_to_one=False)
    pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                   scheduler=scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=torch.float16).to("cuda")
    return pipe


with open(str(Path('IT_Academy_final_project').joinpath('variables.json')), 'r') as file:
    d = file.read()
d = json.loads(d)
model_path = d['trained_model_dir']
pipe = create_pipe(model_path)


def generate_prompt_images(prompt: str, pipe=pipe, num_samples: int = 2,
                           guidance_scale=7.5, num_inference_steps=100,
                           save_path=r'/result'):
    height = 512
    width = 512
    with autocast('cuda'), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device='cuda')
        ).images
    name = re.sub(r'[^\w]', ' ', prompt)
    c = 0

    if not Path(save_path).exists():
        Path(save_path).mkdir()

    save_path = str(Path(save_path).joinpath('{}.png'))

    for img in images:
        img.save(save_path.format(name[:50] + '_' + str(c + 1)))
        c += 1


def generate_n_images(class_name, instance_name: str, save_path, num: int = 10, num_inference_steps=100):
    list_prompts = class_prompts(class_name)
    for _ in range(num):
        prompt = random.choice(list_prompts).replace('*', '{} {}'.format(class_name, instance_name))
        generate_prompt_images(prompt, num_samples=1,
                               num_inference_steps=num_inference_steps,
                               save_path=save_path)


def generate_image_with_random_lexica_prompt(save_path, word='', num_samples: int = 2, num_inference_steps=100):
    for _ in range(num_samples):
        prompt = random_prompt(word=word)
        generate_prompt_images(prompt, num_samples=num_samples,
                           num_inference_steps=num_inference_steps,
                           save_path=save_path)
