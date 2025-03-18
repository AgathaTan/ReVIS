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
# from diffusers import DDPMScheduler, UNet2DConditionModel

from load_eegdatasets_v3 import SoloEEGDataset, JointEEGDataset
from torch.utils.data import DataLoader
from utils import Util

from model.ReVisModels_v7 import ReVisEncoder #, ATMS
from diffusion_prior import DiffusionPriorUNet, Pipe
# from model.ReVisModels_v2 import teModel

# Classifier-Free Guidance Training

# def extract_id_from_string(s):
#     match = re.search(r'\d+$', s)
#     if match:
#         return int(match.group())
#     return None

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

def evaluate(model, data_loader, accelerator, sub):
    model.eval()

    with torch.no_grad():
        eval_loss_sum = 0
        eval_clip_semantic_loss_sum = 0
        eval_mse_semantic_loss_sum = 0
        eval_clip_submodal_loss_sum = 0
        eval_mse_hidden_submodal_loss_sum = 0
        eval_mse_submodal_loss_sum = 0
        # eval_diffusion_prior_loss_sum = 0
        loss = 0
        alpha = 0.8
        beta = 0.1
        for idx, batch in enumerate(data_loader):
            logit_scale = model.logit_scale

            # semantic_hidden_states = batch["clip_text_hidden_states"]
            submodal_hidden_states = batch["clip_image_hidden_states"]
            # semantic_features = batch["clip_text_features"]
            submodal_features = batch['clip_img_features']
            batch_size = submodal_features.shape[0]

            submodal_hidden_states_pred, submodal_features_pred  = model(batch["eeg_data"].to(accelerator.device), sub)
            # submodal_hidden_states_pred = submodal_hidden_states_pred
            # features_pred = features_pred

            submodal_mse_loss = F.mse_loss(submodal_features_pred.float(), submodal_features.float(),
                                                  reduction="mean")
            submodal_features_pred_norm = nn.functional.normalize(submodal_features_pred.flatten(1), dim=-1)
            submodal_features_norm = nn.functional.normalize(submodal_features.flatten(1), dim=-1)
            submodal_clip_loss = Util.soft_clip_loss(submodal_features_pred_norm.float(), submodal_features_norm.float())

            submodal_mse_hidden_loss = F.mse_loss(submodal_hidden_states_pred.float(), submodal_hidden_states.float(),
                                           reduction="mean")

            eval_avg_clip_submodal_loss = accelerator.gather(submodal_clip_loss.repeat(batch_size)).mean().item()
            eval_clip_submodal_loss_sum += eval_avg_clip_submodal_loss
            eval_avg_mse_hidden_submodal_loss = accelerator.gather(submodal_mse_hidden_loss.repeat(batch_size)).mean().item()
            eval_mse_hidden_submodal_loss_sum += eval_avg_mse_hidden_submodal_loss
            eval_avg_submodal_mse_loss = accelerator.gather(submodal_mse_loss.repeat(batch_size)).mean().item()
            eval_mse_submodal_loss_sum += eval_avg_submodal_mse_loss


        eval_clip_submodal_loss = eval_clip_submodal_loss_sum / len(data_loader)  # (idx+1)
        eval_mse_hidden_submodal_loss = eval_mse_hidden_submodal_loss_sum / len(data_loader)  # (idx+1)
        eval_mse_submodal_loss = eval_mse_submodal_loss_sum / len(data_loader)  # (idx+1)
        # eval_submodal_loss = eval_mse_submodal_loss + eval_clip_submodal_loss
        # eval_diffusion_prior_loss = eval_diffusion_prior_loss_sum / len(data_loader)
        eval_loss = alpha * eval_clip_submodal_loss + + beta * eval_mse_submodal_loss + (1 - alpha- beta) * eval_mse_hidden_submodal_loss# + prior_loss
    # return eval_clip_submodal_loss

    return {

            "eval_clip_submodal_loss" : eval_clip_submodal_loss,
            "eval_mse_submodal_loss": eval_mse_submodal_loss,
            "eval_mse_hidden_submodal_loss" : eval_mse_hidden_submodal_loss,
            "eval_loss": eval_loss
        }

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
        "patch_d_model" : 64,
        "embed" : 'timeF',
        "freq" : 'h',
        "joint_train" : False,
        "num_subjects" : 10,
        "dropout" : 0.25,
        "factor" : 1,
        "n_heads" : 4,
        "encode_layers" : 1,
        "d_ff" : 512,
        "patch_d_ff = 128" : 128,
        "activation" :'gelu',
        "enc_in" : 63,
        "padding_patch" : 'end',
        "patch_len" : 4,
        "patch_stride" : 4,
        "num_joint_train_epochs" : 20,
        "num_train_epochs" : 250,
        "learning_rate" : 1e-3,
        "weight_decay" : 1e-4,
        "extra_submodal_dim" : 4,
        "log_with" : "wandb",
        "gradient_accumulation_steps" : 1 }

    accelerator.init_trackers(
        project_name="ReVIS_Semantic_wandb_0313_dynamic_no_spac_3loss",
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

    SemanticEncoder = ReVisEncoder(default_configs, encoder_type=encoder_type, subjects=[sub], cross_sub = False)

    # semantic_model_path = os.path.join(output_dir, 'semanticEncoder_0313_no_dynamic')
    # semantic_ckpt_path = f'{semantic_model_path}/{5}.bin'
    # # ckpt_path = f'{output_dir}/semanticEncoder_0219/{num_train_epochs}.pth'
    # if os.path.exists(semantic_ckpt_path):
    #     semantic_state_dict = torch.load(semantic_ckpt_path, map_location='cuda:0')
    # else:
    #     raise FileNotFoundError(
    #         "semantic encoder model not found at {}, please train SemanticEncoder firstly".format(semantic_ckpt_path))
    # SemanticEncoder.load_state_dict(semantic_state_dict)
    #
    # prior_model_path = os.path.join(output_dir, 'global_diffusion_prior_0313')
    # diffusion_prior = DiffusionPriorUNet(featurn_type="global", cond_dim=1024, dropout=0.1)
    # prior_ckpt_path = f'{prior_model_path}/{140}.bin'
    # # ckpt_path = f'{output_dir}/semanticEncoder_0219/{num_train_epochs}.pth'
    # if os.path.exists(prior_ckpt_path):
    #     prior_state_dict = torch.load(prior_ckpt_path, map_location='cuda:0')
    # else:
    #     raise FileNotFoundError(
    #         "semantic encoder model not found at {}, please train SemanticEncoder firstly".format(prior_ckpt_path))
    # diffusion_prior.load_state_dict(prior_state_dict)
    # diffusion_prior.requires_grad_(False)
    # pipe = Pipe(sub=sub, diffusion_prior=diffusion_prior, device=accelerator.device, output_dir=output_dir,
    #             num_epochs=150)

    # scheduler = DDPMScheduler()
    # diffusion_prior = DiffusionPriorUNet(featurn_type="global", cond_dim=1024, dropout=0.1)
    # SemanticEncoder = ATMS()
    # SemanticEncoder =  teModel(default_configs)
    # semantic_model_path = os.path.join(output_dir, 'Encoder_0313_dynamic_no_spac_3loss')
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

    SemanticEncoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        SemanticEncoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    SemanticEncoder.train()
    alpha = 0.8
    beta = 0.1
    for epoch in range(default_configs.num_train_epochs):

        train_clip_semantic_loss_sum = 0
        train_mse_semantic_loss_sum = 0
        train_clip_submodal_loss_sum = 0
        train_mse_submodal_loss_sum = 0
        train_mse_hidden_submodal_loss_sum = 0
        # prior_loss_sum = 0
        for idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{default_configs.num_train_epochs}", ncols=100)):
            with accelerator.accumulate(SemanticEncoder):
                logit_scale = SemanticEncoder.logit_scale
                loss = 0.
                # semantic_hidden_states = batch["clip_text_hidden_states"]
                submodal_hidden_states = batch["clip_image_hidden_states"]
                # semantic_features = batch["clip_text_features"]
                submodal_features = batch['clip_img_features']
                batch_size = submodal_features.shape[0]

                submodal_hidden_states_pred, submodal_features_pred = SemanticEncoder(batch["eeg_data"].to(accelerator.device), sub)


                submodal_features_pred_norm = nn.functional.normalize(submodal_features_pred.flatten(1), dim=-1)
                submodal_features_norm = nn.functional.normalize(submodal_features.flatten(1), dim=-1)
                # submodal_clip_loss = Util.soft_clip_loss(submodal_features_pred_norm, submodal_features_norm)
                submodal_clip_loss = Util.soft_clip_loss(submodal_features_pred_norm.float(), submodal_features_norm.float())
                loss += alpha * submodal_clip_loss
                # loss += submodal_clip_loss

                submodal_mse_loss = F.mse_loss(submodal_features_pred.float(), submodal_features.float(),
                                                   reduction="mean")
                loss += beta * submodal_mse_loss

                # if default_configs.mse_submodal_mult:
                #     submodal_mse_loss = nn.MSELoss()(submodal_hidden_states_pred_norm, submodal_hidden_states_norm)
                #     submodal_mse_loss =default_configs.mse_submodal_mult * submodal_mse_loss
                #     loss += submodal_mse_loss
                # mse loss 用hidden_states算
                submodal_hidden_mse_loss = F.mse_loss(submodal_hidden_states_pred.float(), submodal_hidden_states.float(),
                                                   reduction="mean")
                loss += (1 - alpha - beta) * submodal_hidden_mse_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                train_avg_submodal_clip_loss = accelerator.gather(submodal_clip_loss.repeat(batch_size)).mean().item()
                train_clip_submodal_loss_sum += train_avg_submodal_clip_loss
                train_avg_submodal_mse_loss = accelerator.gather(submodal_mse_loss.repeat(batch_size)).mean().item()
                train_mse_submodal_loss_sum += train_avg_submodal_mse_loss
                train_avg_submodal_hidden_mse_loss = accelerator.gather(submodal_hidden_mse_loss.repeat(batch_size)).mean().item()
                train_mse_hidden_submodal_loss_sum += train_avg_submodal_hidden_mse_loss

        eval_loss = evaluate(SemanticEncoder, eval_dataloader, accelerator, sub)

        lr_scheduler.step(eval_loss["eval_loss"])
        train_clip_submodal_loss_epoch = train_clip_submodal_loss_sum / len(train_dataloader)  # (idx+1)
        train_mse_hidden_submodal_loss_epoch = train_mse_hidden_submodal_loss_sum / len(train_dataloader) # (idx+1)
        train_mse_submodal_loss_epoch = train_mse_submodal_loss_sum / len(train_dataloader) # (idx+1)


        eval_mse_submodal_loss = eval_loss["eval_mse_submodal_loss"]
        eval_clip_submodal_loss = eval_loss['eval_clip_submodal_loss']
        eval_mse_hidden_submodal_loss = eval_loss['eval_mse_hidden_submodal_loss']
        eval_loss = eval_loss['eval_loss']

        accelerator.log({"epoch": epoch + 1, "train_clip_submodal_loss": train_clip_submodal_loss_epoch,
                         "train_mse_hidden_submodal_loss": train_mse_hidden_submodal_loss_epoch,
                         "train_mse_submodal_loss": train_mse_submodal_loss_epoch,
                         "eval_mse_submodal_loss": eval_mse_submodal_loss,
                         "eval_mse_hidden_submodal_loss": eval_mse_hidden_submodal_loss,
                         "eval_clip_submodal_loss": eval_clip_submodal_loss,
                         "eval_loss": eval_loss})
        print(f'epoch: {epoch + 1}, train_clip_image_loss: {train_clip_submodal_loss_epoch}, '
              f' eval_clip_image_loss: {eval_clip_submodal_loss}, '
              f'train_mse_image_loss: {train_mse_submodal_loss_epoch},'
              f'eval_mse_image_loss: {eval_mse_submodal_loss}, '
              f'train_mse_image_hidden_loss: {train_mse_hidden_submodal_loss_epoch}, '
              f'eval_mse_hidden_submodal_loss: {eval_mse_hidden_submodal_loss}'            
              f'eval_loss: {eval_loss}')

        if (epoch + 1) % 2 == 0:
            # Save the model every 10 epochs
            save_model_path = os.path.join(output_dir, encoder_type+'Encoder_0313_dynamic_no_spac_3loss')
            os.makedirs(save_model_path, exist_ok=True)
            file_path = f'{save_model_path}/{epoch+1}.bin'
            # torch.save({"SemanticEncoder": SemanticEncoder.state_dict(), "diffusion_prior": diffusion_prior.state_dict()}, file_path)
            torch.save(SemanticEncoder.state_dict(), file_path) # torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter.bin")
            print(f"Model saved in {file_path}!")


    accelerator.end_training()

if __name__ == '__main__':
    main()