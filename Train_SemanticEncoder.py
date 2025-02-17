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

os.environ["WANDB_API_KEY"] = "43bc724910d8a22d37703f2f3b11257b72662b4c"
os.environ["WANDB_MODE"] = 'offline'

# Load the configuration from the JSON file
data_config_path = "data_config.json"
with open(data_config_path, "r") as data_config_file:
    data_config = json.load(data_config_file)

# Access the pretrained SD 1.5
pretrained_model_path = data_config["pretrained_model_path"]
# Access the paths from the config
data_path = data_config["data_path"]
output_dir = data_config["output_dir"]
#logging_dir = data_config["Semanric_logging_dir"]

default_configs = Configs()

def evaluate(model, data_loader, subject_ids, noise_scheduler, frozen_unet, accelerator):
    model.eval()

    with torch.no_grad():
        eval_loss_sum = 0
        for idx, batch in enumerate(data_loader):
            latents = batch["vae_img_features"]
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = model(batch["eeg_data"].to(accelerator.device), sub_ids=subject_ids)
            noise_pred = frozen_unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            eval_avg_loss = accelerator.gather(loss.repeat(batch_size)).mean().item()
            eval_loss_sum += eval_avg_loss
        eval_loss = eval_loss_sum / len(data_loader)

    return eval_loss


def main():
    # logging_directory = os.path.join(output_dir, logging_dir)
    # accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_directory)
    accelerator = Accelerator(
        log_with=default_configs.log_with,
        #project_config=accelerator_project_config,
        gradient_accumulation_steps=default_configs.gradient_accumulation_steps
    )
    accelerator.init_trackers(
        project_name="ReVIS_Semantic_wandb_0128",
        config=default_configs,
        init_kwargs={"wandb": {"entity": "agathatan"}}
    )

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    frozen_unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    frozen_unet.requires_grad_(False)
    frozen_unet.to(accelerator.device)

    sub = 'sub-08'
    encoder_type = 'semantic'

    SemanticEncoder = ReVisEncoder(default_configs, encoder_type=encoder_type)
    optimizer = torch.optim.AdamW(itertools.chain(SemanticEncoder.parameters()), lr=default_configs.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    train_dataset = SoloEEGDataset(data_path, subjects=[sub], train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=default_configs.train_batch_size, shuffle=True, num_workers=0,
                              drop_last=True)
    eval_dataset = SoloEEGDataset(data_path, subjects=[sub], train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=default_configs.train_batch_size, shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

    SemanticEncoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        SemanticEncoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    SemanticEncoder.train()
    for epoch in range(default_configs.num_train_epochs):
        train_loss_sum = 0
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

                train_avg_loss = accelerator.gather(loss.repeat(batch_size)).mean().item()
                train_loss_sum += train_avg_loss
            # del batch

        eval_loss = evaluate(SemanticEncoder, eval_dataloader, subject_ids, noise_scheduler, frozen_unet, accelerator)
        lr_scheduler.step(eval_loss)
        train_loss_epoch = train_loss_sum / len(train_dataloader) # (idx+1)
        accelerator.log({"epoch": epoch + 1, "train_loss": train_loss_epoch, "eval_loss":eval_loss})
        print(f'epoch: {epoch + 1}, train_loss: {train_loss_epoch}, eval_loss: {eval_loss}')

        if (epoch + 1) % 10 == 0:
            # Save the model every 10 epochs
            save_model_path = os.path.join(output_dir, encoder_type+'Encoder_0128', sub)
            os.makedirs(save_model_path, exist_ok=True)
            file_path = f'{save_model_path}/{epoch+1}.pth'
            torch.save(SemanticEncoder.state_dict(), file_path) # torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter.bin")
            print(f"Model saved in {file_path}!")


    accelerator.end_training()

if __name__ == '__main__':
    main()

