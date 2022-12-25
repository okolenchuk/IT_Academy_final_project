from random_word import RandomWords
import requests
import json
import random
from pathlib import Path

prompts_cat = {'women': r'female.txt',
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
    with open(prompts_file, 'r') as file:
        list_prompts = file.readlines()
    return list_prompts
