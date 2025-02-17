import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def latent_to_img(latents,vae):
    """use vae decoder from stable-diffusion-v1-5 to generate image"""
    latents = (1/0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0,1)
    image = image.detach().cpu().permute(0,2,3,1).numpy()
    images = (image * 255).round().astype("uint8")
    recon_images = [Image.fromarray(image) for image in images ]
    return recon_images