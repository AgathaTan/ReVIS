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

def check_loss(loss, message="loss"):
    if loss.isnan().any():
        raise ValueError(f'NaN loss in {message}')

def soft_clip_loss(preds, targs, temp=0.05, eps=1e-10):
    clip_clip = (targs @ targs.T) / temp + eps
    check_loss(clip_clip, "clip_clip")
    brain_clip = (preds @ targs.T) / temp + eps
    check_loss(brain_clip, "brain_clip")

    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    check_loss(loss1, "loss1")
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    check_loss(loss2, "loss2")

    loss = (loss1 + loss2) / 2
    return loss