import torch
import torch.nn as nn
import torch.nn.functional as F
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
from diffusers import DDPMScheduler, UNet2DConditionModel

from load_eegdatasets_v3 import SoloEEGDataset, JointEEGDataset
from torch.utils.data import DataLoader
from utils import Util

from model.ReVisModels_v7 import ReVisEncoder, DiffusionPriorUNet #, ATMS
from model.ReVisModels_v2 import teModel

# Classifier-Free Guidance Training

# def extract_id_from_string(s):
#     match = re.search(r'\d+$', s)
#     if match:
#         return int(match.group())
#     return None

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

def evaluate(model, diffusion_prior, scheduler, data_loader, accelerator, sub, prior_mult):
    model.eval()

    with torch.no_grad():
        eval_loss_sum = 0
        eval_clip_semantic_loss_sum = 0
        eval_mse_semantic_loss_sum = 0
        eval_clip_submodal_loss_sum = 0
        eval_mse_submodal_loss_sum = 0
        eval_diffusion_prior_loss_sum = 0
        loss = 0
        alpha = 0.75
        beta = 0.15
        for idx, batch in enumerate(data_loader):
            logit_scale = model.logit_scale

            # semantic_hidden_states = batch["clip_text_hidden_states"]
            submodal_hidden_states = batch["clip_image_hidden_states"]
            # submodal_features = batch["clip_img_features"]
            semantic_features = batch["clip_text_features"]
            submodal_features = batch['clip_img_features']
            batch_size = submodal_features.shape[0]

            # subject_id = extract_id_from_string(sub)
            # if subject_id is None:
            #     raise ValueError(f"Invalid subject_id extracted from: {sub}")
            # subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(accelerator.device)

            submodal_hidden_states_pred, features_pred  = model(batch["eeg_data"].to(accelerator.device), sub)
            submodal_hidden_states_pred = submodal_hidden_states_pred
            features_pred = features_pred

            features_pred_norm = nn.functional.normalize(features_pred.half().flatten(1), dim=-1)
            submodal_features_norm = nn.functional.normalize(submodal_features.flatten(1), dim=-1)
            # print(f"submodal_features_pred_norm: {submodal_features_pred_norm.mean().item()}")
            # print(f"submodal_features_norm: {submodal_features_norm.mean().item()}")
            # print(f"submodal_clip_loss: {submodal_clip_loss.mean().item()}")
            # submodal_clip_loss = Util.soft_clip_loss(submodal_features_pred_norm, submodal_features_norm)
            submodal_clip_loss = Util.soft_clip_loss(features_pred_norm, submodal_features_norm)
            # if default_configs.mse_submodal_mult:
            #     submodal_mse_loss = nn.MSELoss()(submodal_hidden_states_pred_norm, submodal_hidden_states_norm)
            #     submodal_mse_loss = default_configs.mse_submodal_mult * submodal_mse_loss
            submodal_mse_loss = F.mse_loss(submodal_hidden_states_pred.half(), submodal_hidden_states,
                                           reduction="mean")

            eval_avg_clip_submodal_loss = accelerator.gather(submodal_clip_loss.repeat(batch_size)).mean().item()
            eval_clip_submodal_loss_sum += eval_avg_clip_submodal_loss
            eval_avg_mse_submodal_loss = accelerator.gather(submodal_mse_loss.repeat(batch_size)).mean().item()
            eval_mse_submodal_loss_sum += eval_avg_mse_submodal_loss

            # semantic_features_pred_norm = nn.functional.normalize(semantic_features_pred.flatten(1), dim=-1)
            semantic_features_norm = nn.functional.normalize(semantic_features.flatten(1), dim=-1)
            semantic_clip_loss = Util.soft_clip_loss(features_pred_norm, semantic_features_norm)
            # semantic_clip_loss = model.loss_func(semantic_features_pred_norm, semantic_features_norm, logit_scale)
            # loss += clip_loss
            # if default_configs.mse_semantic_mult:
            #     semantic_mse_loss = nn.MSELoss()(semantic_hidden_states_pred_norm, semantic_hidden_states_norm)
            #     semantic_mse_loss = default_configs.mse_semantic_mult * semantic_mse_loss

            # semantic_mse_loss = F.mse_loss(semantic_hidden_states_pred.float(), semantic_hidden_states.float(),
            #                           reduction="mean")
            # mse_loss = F.mse_loss(encoder_hidden_states_pred.float(), semantic_hidden_states.float(), reduction="mean")
                # loss += default_configs.mse_mult * mse_loss

            eval_avg_clip_semantic_loss = accelerator.gather(semantic_clip_loss.repeat(batch_size)).mean().item()
            eval_clip_semantic_loss_sum += eval_avg_clip_semantic_loss
            # eval_avg_mse_semantic_loss = accelerator.gather(semantic_mse_loss.repeat(batch_size)).mean().item()
            # eval_mse_semantic_loss_sum += eval_avg_mse_semantic_loss
            noise = torch.randn_like(submodal_features)

            # sample timestep
            num_train_timesteps = scheduler.config.num_train_timesteps
            timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=accelerator.device)

            # add noise to h_embedding
            perturbed_h_embeds = scheduler.add_noise(
                submodal_features,
                noise,
                timesteps
            )

            noise_pre = diffusion_prior(perturbed_h_embeds.float(), timesteps, c=features_pred)
            prior_loss = nn.MSELoss(reduction='mean')(noise_pre, noise)
            eval_prior_avg_loss = accelerator.gather(prior_loss.repeat(batch_size)).mean().item()
            eval_diffusion_prior_loss_sum  += eval_prior_avg_loss
        # del batch

        eval_clip_semantic_loss = eval_clip_semantic_loss_sum / len(data_loader)  # (idx+1)
        # eval_mse_semantic_loss = eval_mse_semantic_loss_sum / len(data_loader)  # (idx+1)
        # eval_semantic_loss = eval_mse_semantic_loss + eval_clip_semantic_loss

        eval_clip_submodal_loss = eval_clip_submodal_loss_sum / len(data_loader)  # (idx+1)
        eval_mse_submodal_loss = eval_mse_submodal_loss_sum / len(data_loader)  # (idx+1)
        # eval_submodal_loss = eval_mse_submodal_loss + eval_clip_submodal_loss
        eval_diffusion_prior_loss = eval_diffusion_prior_loss_sum / len(data_loader)
        alpha = 0.7
        beta = 0.2
        eval_loss = alpha * eval_clip_submodal_loss + beta * eval_clip_semantic_loss + (1 - alpha - beta) * eval_mse_submodal_loss + prior_loss
    # return eval_clip_submodal_loss

    return {

            "eval_clip_submodal_loss" : eval_clip_submodal_loss,
            "eval_mse_submodal_loss": eval_mse_submodal_loss,
            # "eval_submodal_loss": eval_submodal_loss,
            # "eval_mse_semantic_loss": eval_mse_semantic_loss,
            "eval_clip_semantic_loss" : eval_clip_semantic_loss,
            # "eval_semantic_loss": eval_semantic_loss,
            "eval_diffusion_prior_loss": eval_diffusion_prior_loss,
            "eval_loss": eval_loss
        }

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


def main():
    # logging_directory = os.path.join(output_dir, logging_dir)
    # accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_directory)
    kwargs = DDPK(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with=default_configs.log_with,
        #project_config=accelerator_project_config,
        gradient_accumulation_steps=default_configs.gradient_accumulation_steps,
        kwargs_handlers = [kwargs]
    )
    configs = {"train_batch_size" : 8, "task_name" : 'reconstruction', "seq_len" : 100,
        "freq_seq_len" : 50,
        "output_attention" : False,
        "d_model" : 256,
        "patch_d_model" : 32,
        "embed" : 'timeF',
        "freq" : 'h',
        "joint_train" : False,
        "num_subjects" : 10,
        "dropout" : 0.25,
        "factor" : 1,
        "n_heads" : 4,
        "encode_layers" : 1,
        "d_ff" : 512,
        "activation" :'gelu',
        "enc_in" : 63,
        "padding_patch" : 'end',
        "patch_len" : 1,
        "patch_stride" : 1,
        "num_joint_train_epochs" : 20,
        "num_train_epochs" : 250,
        "learning_rate" : 1e-3,
        "weight_decay" : 1e-4,
        "extra_submodal_dim" : 4,
        "log_with" : "wandb",
        "gradient_accumulation_steps" : 1 }

    accelerator.init_trackers(
        project_name="ReVIS_Semantic_wandb_0307",
        config = configs,
        init_kwargs={"wandb": {"entity": "agathatan"}}
    )
    print(f"Available devices: {accelerator.state.device}")

    # noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    # frozen_unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    # frozen_unet.requires_grad_(False)
    # frozen_unet.to(accelerator.device)

    sub = 'sub-08'
    # subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07']
    encoder_type = 'semantic'

    SemanticEncoder = ReVisEncoder(default_configs, encoder_type=encoder_type, subjects=[sub],cross_sub = False)

    scheduler = DDPMScheduler()
    diffusion_prior = DiffusionPriorUNet(featurn_type="global", cond_dim=1024, dropout=0.1)
    # SemanticEncoder = ATMS()
    # SemanticEncoder =  teModel(default_configs)
    # semantic_model_path = os.path.join(output_dir, 'semantic_cross_Encoder_0307')
    # ckpt_path = f'{semantic_model_path}/1.pth'
    # if os.path.exists(ckpt_path):
    #     state_dict = torch.load(ckpt_path)
    # model_dict = SemanticEncoder.state_dict()
    # insubject_dict = {}
    # for key, value in state_dict.items():
    # #     # Check if the key is in model_dict, and exclude 'value_embedding' keys from the update
    #     if 'value_embedding' not in key:
    #         # Handle the case where the key has a 'module.' prefix
    #         clean_key = key.replace('module.', '')
    #         if clean_key in model_dict:
    #             insubject_dict[clean_key] = value
    #
    # for key, value in insubject_dict.items():
    #     print(f"Updated key: {key}")
    # model_dict.update(insubject_dict)
    # SemanticEncoder.load_state_dict(model_dict)
    no_decay = ['bias', 'Norm', 'temperature']
    opt_grouped_parameters = [
        {'params': [p for n, p in SemanticEncoder.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': default_configs.weight_decay},
        {'params': [p for n, p in SemanticEncoder.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    # optimizer = torch.optim.AdamW(itertools.chain(SemanticEncoder.parameters()), lr=default_configs.learning_rate,
    #                               weight_decay=default_configs.weight_decay)
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=default_configs.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    train_dataset = SoloEEGDataset(data_path, subjects=[sub], train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=default_configs.train_batch_size, shuffle=False, num_workers=0,
                              drop_last=True)
    eval_dataset = SoloEEGDataset(data_path, subjects=[sub], train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=default_configs.train_batch_size, shuffle=False,
                                  num_workers=0, drop_last=True)

    diffusion_prior, SemanticEncoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        diffusion_prior, SemanticEncoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    SemanticEncoder.train()
    alpha = 0.75
    beta = 0.15
    for epoch in range(default_configs.num_train_epochs):

        train_clip_semantic_loss_sum = 0
        train_mse_semantic_loss_sum = 0
        train_clip_submodal_loss_sum = 0
        train_mse_submodal_loss_sum = 0
        prior_loss_sum = 0
        for idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{default_configs.num_train_epochs}", ncols=100)):
            with accelerator.accumulate(SemanticEncoder):
                logit_scale = SemanticEncoder.logit_scale
                pre_loss = 0.
                # semantic_hidden_states = batch["clip_text_hidden_states"]
                submodal_hidden_states = batch["clip_image_hidden_states"]
                semantic_features = batch["clip_text_features"]
                submodal_features = batch['clip_img_features']
                batch_size = submodal_features.shape[0]

                submodal_hidden_states_pred, features_pred = SemanticEncoder(batch["eeg_data"].to(accelerator.device), sub)
                submodal_hidden_states_pred = submodal_hidden_states_pred
                features_pred = features_pred
                # subject_id = extract_id_from_string(sub)
                # if subject_id is None:
                #     raise ValueError(f"Invalid subject_id extracted from: {sub}")
                # subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(accelerator.device)

                # submodal_features_pred = SemanticEncoder(batch["eeg_data"].to(accelerator.device), subject_ids)
                # print(f'submodal_hidden_states_pred shape is {submodal_hidden_states_pred.shape}')
                # print(f'submodal_hidden_states shape is {submodal_hidden_states.shape}')

                features_pred_norm = nn.functional.normalize(features_pred.half().flatten(1), dim=-1)
                submodal_features_norm = nn.functional.normalize(submodal_features.flatten(1), dim=-1)
                # submodal_clip_loss = Util.soft_clip_loss(submodal_features_pred_norm, submodal_features_norm)
                submodal_clip_loss = Util.soft_clip_loss(features_pred_norm.float(), submodal_features_norm.float())
                pre_loss += alpha * submodal_clip_loss
                # loss += submodal_clip_loss

                # if default_configs.mse_submodal_mult:
                #     submodal_mse_loss = nn.MSELoss()(submodal_hidden_states_pred_norm, submodal_hidden_states_norm)
                #     submodal_mse_loss =default_configs.mse_submodal_mult * submodal_mse_loss
                #     loss += submodal_mse_loss
                # mse loss 用hidden_states算
                if default_configs.mse_submodal_mult:
                    submodal_mse_loss = F.mse_loss(submodal_hidden_states_pred.float(), submodal_hidden_states.float(),
                                                   reduction="mean")
                    pre_loss += (1 - alpha - beta) * submodal_mse_loss
                #
                #
                #
                # semantic_features_pred_norm = nn.functional.normalize(semantic_features_pred.flatten(1), dim=-1)
                semantic_features_norm = nn.functional.normalize(semantic_features.flatten(1), dim=-1)
                semantic_clip_loss = Util.soft_clip_loss(features_pred_norm, semantic_features_norm)
                # semantic_clip_loss = SemanticEncoder.loss_func(semantic_features_pred_norm, semantic_features_norm, logit_scale)
                pre_loss += beta * semantic_clip_loss
                # loss = alpha * submodal_clip_loss + (1 - alpha - beta) * submodal_mse_loss + beta * semantic_clip_loss

                # Generate noisy embeddings as input
                noise = torch.randn_like(submodal_features)

                # sample timestep
                num_train_timesteps = scheduler.config.num_train_timesteps
                timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=accelerator.device)

                # add noise to h_embedding
                perturbed_h_embeds = scheduler.add_noise(
                    submodal_features,
                    noise,
                    timesteps
                )

                noise_pre = diffusion_prior(perturbed_h_embeds.float(), timesteps, c=features_pred)
                prior_loss = F.mse_loss(noise_pre.float(), noise.float(), reduction="mean")
                # if default_configs.mse_semantic_mult:
                #     semantic_mse_loss = nn.MSELoss()(semantic_hidden_states_pred_norm, semantic_hidden_states_norm)
                #     semantic_mse_loss = default_configs.mse_semantic_mult * semantic_mse_loss
                #     loss += semantic_mse_loss
                # mse loss 用hidden_states算
                # if default_configs.mse_semantic_mult:
                #     semantic_mse_loss = F.mse_loss(semantic_hidden_states_pred.float(), semantic_hidden_states.float(),
                #                                    reduction="mean")
                #     loss += (1-alpha) * semantic_mse_loss
                loss = pre_loss + default_configs.prior_mult * prior_loss
                # loss = loss.float()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                # lr_scheduler.step(loss)

                # train_avg_loss = accelerator.gather(loss.repeat(batch_size)).mean().item()
                # train_loss_sum += train_avg_loss
                train_avg_submodal_clip_loss = accelerator.gather(submodal_clip_loss.repeat(batch_size)).mean().item()
                train_clip_submodal_loss_sum += train_avg_submodal_clip_loss
                train_avg_submodal_mse_loss = accelerator.gather(submodal_mse_loss.repeat(batch_size)).mean().item()
                train_mse_submodal_loss_sum += train_avg_submodal_mse_loss

                train_avg_semantic_clip_loss = accelerator.gather(semantic_clip_loss.repeat(batch_size)).mean().item()
                train_clip_semantic_loss_sum += train_avg_semantic_clip_loss
                # train_avg_semantic_mse_loss = accelerator.gather(semantic_mse_loss.repeat(batch_size)).mean().item()
                # train_mse_semantic_loss_sum += train_avg_semantic_mse_loss
                prior_avg_loss = accelerator.gather(prior_loss.repeat(batch_size)).mean().item()
                prior_loss_sum += prior_avg_loss
            # del batch

        # eval_loss = evaluate(SemanticEncoder, eval_dataloader, noise_scheduler, frozen_unet, accelerator, sub)
        eval_loss = evaluate(SemanticEncoder, diffusion_prior, scheduler, eval_dataloader, accelerator, sub, default_configs.mse_submodal_mult)

        lr_scheduler.step(eval_loss["eval_loss"])
        train_clip_submodal_loss_epoch = train_clip_submodal_loss_sum / len(train_dataloader)  # (idx+1)
        train_mse_submodal_loss_epoch = train_mse_submodal_loss_sum / len(train_dataloader) # (idx+1)

        train_clip_semantic_loss_epoch = train_clip_semantic_loss_sum / len(train_dataloader)  # (idx+1)
        # train_mse_semantic_loss_epoch = train_mse_semantic_loss_sum / len(train_dataloader) # (idx+1)
        train_prior_loss_epoch = prior_loss_sum / len(train_dataloader)

        eval_mse_submodal_loss = eval_loss["eval_mse_submodal_loss"]
        eval_clip_submodal_loss = eval_loss['eval_clip_submodal_loss']
        # eval_mse_semantic_loss = eval_loss['eval_mse_semantic_loss']
        eval_clip_semantic_loss = eval_loss['eval_clip_semantic_loss']
        eval_diffusion_prior_loss = eval_loss['eval_diffusion_prior_loss']

        accelerator.log({"epoch": epoch + 1, "train_clip_submodal_loss": train_clip_submodal_loss_epoch,
                         "train_clip_semantic_loss": train_clip_semantic_loss_epoch,
                         "train_mse_submodal_loss": train_mse_submodal_loss_epoch,
                         # "train_mse_semantic_loss": train_mse_semantic_loss_epoch,
                         "eval_mse_submodal_loss": eval_mse_submodal_loss,
                         # "eval_mse_semantic_loss": eval_mse_semantic_loss,
                         "eval_clip_submodal_loss": eval_clip_submodal_loss,
                         "eval_clip_semantic_loss": eval_clip_semantic_loss,
                         "train_prior_loss": train_prior_loss_epoch,
                         "eval_diffusion_prior_loss": eval_diffusion_prior_loss,
                         "eval_loss": eval_loss})
        print(f'epoch: {epoch + 1}, train_clip_submodal_loss: {train_clip_submodal_loss_epoch}, '
              f' eval_clip_image_loss: {eval_clip_submodal_loss}, '
              f'train_clip_text_loss: {train_clip_semantic_loss_epoch}, '
              f' eval_clip_text_loss : {eval_clip_semantic_loss},' 
              f'train_mse_image_loss: {train_mse_submodal_loss_epoch}, '
              # f'train_mse_semantic_loss: {train_mse_semantic_loss_epoch}, '
              f'eval_mse_image_loss: {eval_mse_submodal_loss}, '
              # f'eval_mse_semantic_loss: {eval_mse_semantic_loss}, '            
              f'train_prior_loss: {train_prior_loss_epoch}, '
              f'eval_diffusion_prior_loss: {eval_diffusion_prior_loss}, '              
              f'eval_loss: {eval_loss}')

        if (epoch + 1) % 10 == 0:
            # Save the model every 10 epochs
            save_model_path = os.path.join(output_dir, encoder_type+'Encoder_0307')
            os.makedirs(save_model_path, exist_ok=True)
            file_path = f'{save_model_path}/{epoch+1}.bin'
            torch.save({"SemanticEncoder": SemanticEncoder.state_dict(),
                        "diffusion_prior": diffusion_prior.state_dict()}, file_path)
            # torch.save(SemanticEncoder.state_dict(), file_path) # torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter.bin")
            print(f"Model saved in {file_path}!")


    accelerator.end_training()

if __name__ == '__main__':
    main()