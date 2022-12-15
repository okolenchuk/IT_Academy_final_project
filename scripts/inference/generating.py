

prompt = "a drawing of a okolenchuk wearing glasses and a sweater with ponytail, a character portrait by L\xFC Ji, pixiv contest winner, digital art, sketchfab, flat shading, speedpainting" #@param {type:"string"}
word = "" #@param {type:"string"}
if prompt == '':
  prompt = random_prompt(word)+' okolenchuk'
negative_prompt = "" #@param {type:"string"}
num_samples = 4 #@param {type:"number"}
guidance_scale = 7.5 #@param {type:"number"}
num_inference_steps = 100 #@param {type:"number"}
height = 512 #@param {type:"number"}
width = 512 #@param {type:"number"}

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).images

for img in images:
    display(img)