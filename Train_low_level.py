import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
import json
import numpy as np
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import clip
from utils.Configs_v3 import Configs
import re
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate import DistributedDataParallelKwargs as DDPK
import itertools
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from diffusion_prior import Low_Level_DiffusionPriorUNet, Low_Level_Pipe
from load_eegdatasets_v3 import SoloEEGDataset, JointEEGDataset
from torch.utils.data import DataLoader
from utils import Util

from model.ReVisModels_v7 import ReVisEncoder #, ATMS
# from diffusion_prior import DiffusionPriorUNet, Pipe
# from model.ReVisModels_v2 import teModel

# Classifier-Free Guidance Training

# def extract_id_from_string(s):
#     match = re.search(r'\d+$', s)
#     if match:
#         return int(match.group())
#                             return None

os.environ["WANDB_API_KEY"] = "49ce86531e1424c97212624c9c2bb4eee575a561"
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

def evaluate(Low_Level_diffusion_prior, eval_dataloader, eeg_model, scheduler, accelerator, sub):
    Low_Level_diffusion_prior.eval()
    eeg_model.eval()
    criterion = nn.MSELoss(reduction='none')
    num_train_timesteps = scheduler.config.num_train_timesteps
    loss_sum = 0
    with torch.no_grad():
        for idx, batch in enumerate(eval_dataloader):
            # c_embeds = batch['c_embedding'].to(device) if 'c_embedding' in batch.keys() else None
            # h_embeds = batch['h_embedding'].to(device)
            submodal_hidden_states_pred, submodal_features_pred = eeg_model(batch["eeg_data"].to(accelerator.device), sub)
            c_embeds = submodal_hidden_states_pred
            h_embeds = batch["vae_img_features"].to(accelerator.device).float()

                # c_embeds = submodal_features_pred
                # h_embeds = batch["clip_img_features"]
            # token_dim = h_embeds.shape[1]
            N = h_embeds.shape[0]

            # 1. randomly replecing c_embeds to None
            if torch.rand(1) < 0.1:
                c_embeds = torch.zeros_like(submodal_hidden_states_pred)

            # 2. Generate noisy embeddings as input
            noise = torch.randn_like(h_embeds)

            # 3. sample timestep
            timesteps = torch.randint(0, num_train_timesteps, (N,), device=accelerator.device)

            # 4. add noise to h_embedding
            perturbed_h_embeds = scheduler.add_noise(
                h_embeds,
                noise,
                timesteps
            )  # (batch_size, embed_dim), (batch_size, )

            # 5. predict noise
            noise_pre = Low_Level_diffusion_prior(perturbed_h_embeds, timesteps, cond=c_embeds)

            # 6. loss function weighted by sigma
            loss = criterion(noise_pre.float(), noise.float())  # (batch_size,)
            loss = (loss).mean()
            loss_sum += loss.item()

        loss_epoch = loss_sum / len(eval_dataloader)

    return loss_epoch


def main():
    accelerator = Accelerator(
        log_with=default_configs.log_with,
        # project_config=accelerator_project_config,
        gradient_accumulation_steps=default_configs.gradient_accumulation_steps,
    )
    sub = 'sub-08'

    encoder_type = 'semantic'
    SemanticEncoder = ReVisEncoder(default_configs, encoder_type=encoder_type, subjects=[sub], cross_sub=False).to(accelerator.device)
    semantic_model_path = os.path.join(output_dir, 'semanticEncoder_0312_dynamic_no_spac_3loss')
    semantic_ckpt_path = f'{semantic_model_path}/{5}.bin'
    # ckpt_path = f'{output_dir}/semanticEncoder_0219/{num_train_epochs}.pth'
    if os.path.exists(semantic_ckpt_path):
        semantic_state_dict = torch.load(semantic_ckpt_path, map_location=accelerator.device)
    else:
        raise FileNotFoundError(
            "semantic encoder model not found at {}, please train SemanticEncoder firstly".format(semantic_ckpt_path))
    SemanticEncoder.load_state_dict(semantic_state_dict)
    SemanticEncoder.requires_grad_(False)

    Low_Level_diffusion_prior = Low_Level_DiffusionPriorUNet(cond_dim=1280, dropout=0.1).float()
    # low_level_pipe = Low_Level_Pipe(Low_Level_diffusion_prior, device=accelerator.device)

    no_decay = ['bias', 'Norm', 'temperature']
    opt_grouped_parameters = [
        {'params': [p for n, p in Low_Level_diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': default_configs.weight_decay},
        {'params': [p for n, p in Low_Level_diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    train_dataset = SoloEEGDataset(data_path, subjects=[sub], train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=default_configs.train_low_level_batch_size, shuffle=False,
                                  num_workers=0, drop_last=True)
    eval_dataset = SoloEEGDataset(data_path, subjects=[sub], train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=default_configs.train_low_level_batch_size, shuffle=False,
                                 num_workers=0, drop_last=True)

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=default_configs.learning_rate)
    criterion = nn.MSELoss(reduction='none')
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * default_configs.num_train_low_level_epochs),
    )
    DDPMscheduler = DDPMScheduler()
    Low_Level_diffusion_prior, optimizer, train_dataloader, eval_dataloader, criterion, lr_scheduler, DDPMscheduler = accelerator.prepare(
        Low_Level_diffusion_prior, optimizer, train_dataloader, eval_dataloader, criterion, lr_scheduler, DDPMscheduler)
    num_train_timesteps = (DDPMscheduler.config.num_train_timesteps)

    for epoch in range(default_configs.num_train_low_level_epochs):
        loss_sum = 0
        SemanticEncoder.eval()
        Low_Level_diffusion_prior.train()
        eval_loss = evaluate(Low_Level_diffusion_prior, eval_dataloader, SemanticEncoder, scheduler=DDPMscheduler,
                             accelerator=accelerator, sub=sub)
        print(f'before eval_loss is： {eval_loss}')
        for idx, batch  in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{default_configs.num_train_low_level_epochs}", ncols=100)):

            submodal_hidden_states_pred, submodal_features_pred = SemanticEncoder(batch["eeg_data"].to(accelerator.device), sub)
            c_embeds = submodal_hidden_states_pred
            h_embeds = batch["vae_img_features"].float()
            N = h_embeds.shape[0]

            # 1. randomly replecing c_embeds to None
            if torch.rand(1) < 0.1:
                c_embeds = torch.zeros_like(submodal_hidden_states_pred)

            # 2. Generate noisy embeddings as input
            noise = torch.randn_like(h_embeds)

            # 3. sample timestep
            timesteps = torch.randint(0, num_train_timesteps, (N,), device=accelerator.device)
            # 检查输入数据是否有 NaN 或 Inf
            if torch.any(torch.isnan(batch["eeg_data"])) or torch.any(torch.isinf(batch["eeg_data"])):
                print("NaN or Inf found in eeg_data!")
            if torch.any(torch.isnan(batch["vae_img_features"])) or torch.any(torch.isinf(batch["vae_img_features"])):
                print("NaN or Inf found in vae_img_features!")
            if torch.any(torch.isnan(noise)) or torch.any(torch.isinf(noise)):
                print("NaN or Inf found in noise!")
            if torch.any(torch.isnan(timesteps)) or torch.any(torch.isinf(timesteps)):
                print("NaN or Inf found in timesteps!")

            # print("h_embeds min:", torch.min(h_embeds).item(), "max:", torch.max(h_embeds).item())
            # print("noise min:", torch.min(noise).item(), "max:", torch.max(noise).item())
            # print("timesteps min:", torch.min(timesteps).item(), "max:", torch.max(timesteps).item())

            # 4. add noise to h_embedding
            perturbed_h_embeds = DDPMscheduler.add_noise(
                h_embeds,
                noise,
                timesteps
            )
            if torch.any(torch.isnan(perturbed_h_embeds)) or torch.any(torch.isinf(perturbed_h_embeds)):
                print("NaN or Inf found in perturbed_h_embeds!")

            # 5. predict noise
            # with torch.autograd.detect_anomaly():
            noise_pre = Low_Level_diffusion_prior(perturbed_h_embeds, timesteps, cond=c_embeds)
            if torch.any(torch.isnan(noise_pre)) or torch.any(torch.isinf(noise_pre)):
                print("NaN or Inf found in noise_pre!")

            # 6. loss
            loss = criterion(noise_pre.float(), noise.float())  # (batch_size,)
            loss = (loss).mean().float()

            # 7. update parameters
            optimizer.zero_grad()
            loss.backward()
                # try:
                #     loss.backward()
                # except RuntimeError as e:
                #     print("🔥 Anomaly detected during backward pass!")
                #     print(e)  # 打印异常信息
                #     import traceback
                #     traceback.print_exc()
                    # loss.backward()
            torch.nn.utils.clip_grad_norm_(Low_Level_diffusion_prior.parameters(), 1.0)
            lr_scheduler.step()
            optimizer.step()
            # print(loss.item())

            loss_sum += loss.item()

        loss_epoch = loss_sum / len(train_dataloader)
        print(f'epoch: {epoch}, loss: {loss_epoch}')
        accelerator.log({"epoch": epoch + 1, "train_loss": loss_epoch})
        # lr_scheduler.step(loss)
        # if (epoch + 1) % 5 == 0:
        eval_loss = evaluate(Low_Level_diffusion_prior, eval_dataloader, SemanticEncoder, scheduler = DDPMscheduler, accelerator=accelerator, sub=sub)
        print(f'epoch: {epoch}, eval_loss: {eval_loss}')
        if (epoch + 1) % 10 == 0:
            save_model_path = os.path.join(output_dir, 'Low_Level_diffusion_prior_0313')
            os.makedirs(save_model_path, exist_ok=True)
            file_path = f'{save_model_path}/{epoch + 1}.bin'
            torch.save(Low_Level_diffusion_prior.state_dict(), file_path)
            print(f"Model saved in {file_path}!")

    accelerator.end_training()

if __name__ == '__main__':
    main()