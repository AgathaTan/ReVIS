import torch
import torch.nn.functional as F
import json
import numpy as np
import os
import clip
from utils.Configs import Configs
import re
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import itertools
from diffusers import DDPMScheduler, UNet2DConditionModel

from load_eegdatasets import SoloEEGDataset
from torch.utils.data import DataLoader

from model.ReVisModels import ReVisEncoder

# Classifier-Free Guidance Training

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


# Load the configuration from the JSON file
data_config_path = "data_config.json"
with open(data_config_path, "r") as data_config_file:
    data_config = json.load(data_config_file)

# Access the pretrained SD 1.5
pretrained_model_path = data_config["pretrained_model_path"]
# Access the paths from the config
data_path = data_config["data_path"]
output_dir = data_config["output_dir"]
logging_dir = data_config["Semanric_logging_dir"]

default_configs = Configs()


def main():
    logging_directory = os.path.join(output_dir, logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_directory)
    accelerator = Accelerator(
        log_with=default_configs.log_with,
        project_config=accelerator_project_config,
        gradient_accumulation_steps=default_configs.gradient_accumulation_steps
    )

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    frozen_unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    frozen_unet.requires_grad_(False)
    frozen_unet.to(accelerator.device)

    sub = 'sub-08'
    encoder_type = 'semantic'

    SemanticEncoder = ReVisEncoder(default_configs, encoder_type=encoder_type)
    optimizer = torch.optim.AdamW(itertools.chain(SemanticEncoder.parameters()), lr=default_configs.learning_rate)
    train_dataset = SoloEEGDataset(data_path, subjects=[sub], train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=default_configs.train_batch_size, shuffle=True, num_workers=0,
                              drop_last=True)

    SemanticEncoder, optimizer, train_dataloader = accelerator.prepare(SemanticEncoder, optimizer, train_dataloader)
    for epoch in range(default_configs.num_train_epochs):
        loss_sum = 0
        for idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{default_configs.num_train_epochs}", ncols=100)):
            with accelerator.accumulate(SemanticEncoder):
                latents = batch["vae_img_features"]
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                subject_id = extract_id_from_string(sub)
                subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(accelerator.device)

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = SemanticEncoder(batch["eeg_data"].to(accelerator.device), sub_ids=subject_ids)
                noise_pred = frozen_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean().item()
                if accelerator.is_main_process:
                    loss_sum += avg_loss
            del batch

        loss_epoch = loss_sum / len(train_dataloader) # (idx+1)
        accelerator.log({"epoch": epoch, "train_loss": loss_epoch})
        print(f'epoch: {epoch}, loss: {loss_epoch}')

        if (epoch + 1) % 10 == 0:
            # Save the model every 10 epochs
            save_model_path = os.path.join(output_dir, encoder_type+'Encoder', sub)
            os.makedirs(save_model_path, exist_ok=True)
            file_path = f'{save_model_path}/{epoch+1}.pth'
            torch.save(SemanticEncoder.state_dict(), file_path) # torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter.bin")
            print(f"Model saved in {file_path}!")


    accelerator.end_training()

if __name__ == '__main__':
    main()