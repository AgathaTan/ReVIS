import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL,DiffusionPipeline
import json
from accelerate import Accelerator
from model.ReVisModels_v5 import ReVisEncoder
from load_eegdatasets_v2 import SoloEEGDataset
from torch.utils.data import DataLoader
from utils.Configs import Configs
import os
from diffusion_prior import *

# Load the configuration from the JSON file
data_config_path = "data_config.json"
with open(data_config_path, "r") as data_config_file:
    data_config = json.load(data_config_file)

# Access the pretrained SD 1.5
pretrained_model_path = data_config["pretrained_model_path"]
data_path = data_config["data_path"]
output_dir = data_config["output_dir"]
ip_cache_dir = data_config["ip_cache_dir"]
default_configs = Configs()

pipe = DiffusionPipeline.from_pretrained(
    pretrained_model_path,
    torch_dtype=torch.float16,
    feature_extractor=None,
    safety_checker=None
)

pipe.load_ip_adapter(
    ip_cache_dir, subfolder="models",
    weight_name="ip-adapter_sd15.bin",
    torch_dtype=torch.float16)
# set ip_adapter scale (defauld is 1)
pipe.set_ip_adapter_scale(1)

sub = 'sub-08'
encoder_type = 'semantic'
num_train_epochs = 110
# default_configs.encode_layers = 12
SemanticEncoder = ReVisEncoder(default_configs, encoder_type='semantic', subjects=[sub], cross_sub=False)
model_dict = SemanticEncoder.state_dict()

semantic_model_path = os.path.join(output_dir, 'semanticEncoder_0303')
ckpt_path = f'{semantic_model_path}/{num_train_epochs}.pth'
# ckpt_path = f'{output_dir}/semanticEncoder_0219/{num_train_epochs}.pth'
if os.path.exists(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cuda:0')
else:
    raise FileNotFoundError(
        "semantic encoder model not found at {}, please train SemanticEncoder firstly".format(ckpt_path))

insubject_dict = {}
for key, value in state_dict.items():
    # Handle the case where the key has a 'module.' prefix
    clean_key = key.replace('module.', '')
    if clean_key in model_dict:
        insubject_dict[clean_key] = value

SemanticEncoder.load_state_dict(insubject_dict)
SemanticEncoder.requires_grad_(False)

accelerator = Accelerator(
        log_with=default_configs.log_with,
        # project_config=accelerator_project_config,
        gradient_accumulation_steps=default_configs.gradient_accumulation_steps
    )

eval_dataset = SoloEEGDataset(data_path, subjects=[sub], train=False)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

num_train_epochs = 100
semantic_diffusion_prior = DiffusionPriorUNet(featurn_type="global", cond_dim=768, dropout=0.1)

semantic_model_path = os.path.join(output_dir, 'semantic_global_diffusion_prior_0303')
semantic_ckpt_path = f'{semantic_model_path}/{num_train_epochs}.pth'
# ckpt_path = f'{output_dir}/semanticEncoder_0219/{num_train_epochs}.pth'
if os.path.exists(semantic_ckpt_path):
    semantic_state_dict = torch.load(semantic_ckpt_path, map_location='cuda:0')
else:
    raise FileNotFoundError(
        "semantic_diffusion_prior model not found at {}, please train Semantic_diffusion_prior firstly".format(
            semantic_ckpt_path))

semantic_diffusion_prior.load_state_dict(semantic_state_dict)
semantic_pipe = Pipe(sub=sub, diffusion_prior=semantic_diffusion_prior, device=accelerator.device,
                     encoder_type="semantic", output_dir=output_dir)
# semantic_pipe.requires_grad_(False)
semantic_diffusion_prior.requires_grad_(False)

num_train_epochs = 100
submodal_diffusion_prior = DiffusionPriorUNet(featurn_type = "global", cond_dim=768, dropout=0.1)

submodal_model_path = os.path.join(output_dir, 'submodal_global_diffusion_prior_0303')
submodal_ckpt_path = f'{submodal_model_path}/{num_train_epochs}.pth'

if os.path.exists(submodal_ckpt_path):
    submodal_state_dict = torch.load(submodal_ckpt_path, map_location='cuda:0')
else:
    raise FileNotFoundError("submodal_diffusion_prior model not found at {}, please train submodal_diffusion_prior firstly".format(submodal_ckpt_path))
submodal_diffusion_prior.load_state_dict(submodal_state_dict)
submodal_pipe = Pipe(sub = sub, diffusion_prior=submodal_diffusion_prior, device=accelerator.device, encoder_type = "submodal", output_dir=output_dir)
submodal_diffusion_prior.requires_grad_(False)

from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
clip_image_encoder.requires_grad_(False)
clip_image_encoder = clip_image_encoder.to(accelerator.device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
from PIL import Image

def Imgae_Enocder(image, device):
    image_inputs = processor(images=Image.open(image[0]).convert("RGB"), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        image_features = clip_image_encoder(image_inputs)
        image_features = image_features.image_embeds
    return image_features

if __name__ == '__main__':
    directory = f"./generated_imgs/{sub}"
    num_inference_steps = 50
    pipe.to(accelerator.device)
    for idx, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            # encoder_hidden_states = batch["clip_text_hidden_states"]
            semantic_hidden_states_pred, submodal_hidden_states_pred, semantic_features_pred, submodal_features_pred = SemanticEncoder(
                batch["eeg_data"], sub)
            # image_embeds = submodal_pipe.generate(c_embeds=submodal_features_pred, num_inference_steps=50, guidance_scale=5.0)
            image_feature = Imgae_Enocder(batch["img"], accelerator.device)
        for j in range(10):
            image_embeds = (image_embeds.to(dtype=torch.float16)).to(accelerator.device)
            neg_images_embeds = torch.zeros_like(image_embeds)
            # image_embeds = torch.cat([neg_images_embeds, image_embeds]).to(dtype=torch.float16, device=accelerator.device)
            text_prompt = ''
            # print(image_embeds)
            # pipe.generate_ip_adapter_embeds = generate_ip_adapter_embeds.__get__(pipe)
            image = pipe(
                prompt=text_prompt,
                # ip_adapter_image_embeds=image_embeds,
                added_cond_kwargs={"image_embeds": image_feature},
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
            ).images[0]
            # image = pipe.generate_ip_adapter_embeds(
            #     prompt=text_prompt,
            #     ip_adapter_embeds=image_embeds,
            #     num_inference_steps=num_inference_steps,
            #     guidance_scale=0.0,
            # ).images[0]
            path = f'{directory}/{batch["category"]}/{j}.png'
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Save the PIL Image
            image.save(path)