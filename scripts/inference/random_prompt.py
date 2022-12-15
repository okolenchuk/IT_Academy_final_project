from random_word import RandomWords
import requests
import json
import random


def random_prompt(word='') -> str:
    if word == '':
        word = RandomWords().get_random_word()
    url = r'https://lexica.art/api/v1/search?q=' + word

    raw_prompt = requests.get(url=url).text
    prompts = json.loads(raw_prompt)['images']

    list_prompt = [text['prompt'] for text in prompts]
    prompt = random.choice(list_prompt)
    return prompt
