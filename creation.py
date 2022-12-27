import argparse
from pathlib import Path
from scripts.inference.generating import *
from scripts.preprocess.preprocess import *


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Creating new images.")
    parser.add_argument(
        "--instance_name",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance that was used on training",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default=None,
        help="Gender with what you want to generate images.",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=4,
        help="The number of samples to 1 prompt.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=100,
        help="The number of inference steps for generating images. 100 are recommended.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/content/generated",
        help="The output directory where the images will be saved.",
    )
    parser.add_argument(
        "--word",
        type=str,
        default="",
        help="Word to create context for random generating",
    )
    parser.add_argument(
        "--use_saved_prompts", choices=('True','False'), help="Whether or not to use presaved prompts."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def create(args):

    save_path = args.save_path

    update_vars('output_path', save_path)
    if args.use_saved_prompts=='True':
        generate_n_images(args.gender, args.instance_name,
                          save_path, num=args.n_images,
                          num_inference_steps=args.save_infer_steps)
    else:
        generate_image_with_random_lexica_prompt(save_path, word='', num_samples=args.n_images,
                                                 num_inference_steps=args.save_infer_steps)


if __name__ == "__main__":
    args = parse_args()
    create(args)
