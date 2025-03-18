import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
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

from model.ReVisModels_v7 import ReVisEncoder

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

def evaluate(model, data_loader, accelerator, subjects):
    model.eval()

    with torch.no_grad():
        eval_clip_semantic_loss_sum = 0
        eval_mse_semantic_loss_sum = 0
        eval_clip_submodal_loss_sum = 0
        eval_mse_submodal_loss_sum = 0

        for idx, batch in enumerate(data_loader):
            # latents = batch["vae_img_features"]
            # noise = torch.randn_like(latents)
            # batch_size = latents.shape[0]
            # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=latents.device)
            # timesteps = timesteps.long()
            # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # semantic_hidden_states = batch["clip_text_hidden_states"]
            submodal_hidden_states = batch["clip_image_hidden_states"]
            semantic_features = batch["clip_text_features"]
            submodal_features = batch['clip_img_features']
            batch_size = submodal_features.shape[0]

            # Initialize a variable to accumulate loss for this batch
            one_clip_submodal_loss_sum = 0
            one_mse_submodal_loss_sum = 0
            one_clip_semantic_loss_sum = 0
            one_mse_semantic_loss_sum = 0
            num_keys = len(batch["eeg_data"])
            for key, value in batch["eeg_data"].items():
                # encoder_hidden_states = model(x = value, subject_id = key)
                # noise_pred = frozen_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # encoder_hidden_states_pred = model(x = value, subject_id = key)
                submodal_hidden_states_pred, submodal_features_pred = model(x=value.to(accelerator.device), subject_id=key)
                submodal_features_pred_norm = nn.functional.normalize(submodal_features_pred.flatten(1), dim=-1)
                submodal_features_norm = nn.functional.normalize(submodal_features.flatten(1), dim=-1)
                submodal_clip_loss = Util.soft_clip_loss(submodal_features_pred_norm.float(), submodal_features_norm.float())
                # if default_configs.mse_submodal_mult:
                #     submodal_mse_loss = nn.MSELoss()(submodal_hidden_states_pred_norm, submodal_hidden_states_norm)
                #     submodal_mse_loss = default_configs.mse_submodal_mult * submodal_mse_loss
                mse_submodal_loss = F.mse_loss(submodal_hidden_states_pred.float(), submodal_hidden_states.float(),
                                               reduction="mean")

                # noise_pred = frozen_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # encoder_hidden_states_pred_norm = nn.functional.normalize(encoder_hidden_states_pred.flatten(1), dim=-1)
                # semantic_hidden_states_norm = nn.functional.normalize(semantic_hidden_states.flatten(1), dim=-1)
                # clip_loss = Util.soft_clip_loss(encoder_hidden_states_pred_norm, semantic_hidden_states_norm)
                # if default_configs.mse_mult:
                #     mse_loss = nn.MSELoss()(encoder_hidden_states_pred_norm, semantic_hidden_states_norm)
                # loss = F.mse_loss(encoder_hidden_states_pred.float(), semantic_hidden_states.float(), reduction="mean")

                # semantic_features_pred_norm = nn.functional.normalize(semantic_features_pred.flatten(1),dim=-1)
                # semantic_features_norm = nn.functional.normalize(semantic_features.flatten(1), dim=-1)
                # semantic_clip_loss = Util.soft_clip_loss(semantic_features_pred_norm.float(), semantic_features_norm.float())

                # if default_configs.mse_semantic_mult:
                #     semantic_mse_loss = nn.MSELoss()(semantic_hidden_states_pred_norm, semantic_hidden_states_norm)
                #     semantic_mse_loss = default_configs.mse_semantic_mult * semantic_mse_loss
                # semantic_mse_loss = F.mse_loss(semantic_hidden_states_pred.float(), semantic_hidden_states.float(),
                #                                reduction="mean")

                # mse_loss = F.mse_loss(encoder_hidden_states_pred.float(), semantic_hidden_states.float(), reduction="mean")
                submodal_clip_loss = accelerator.gather(submodal_clip_loss.repeat(batch_size)).mean().item()
                one_clip_submodal_loss_sum += submodal_clip_loss
                mse_submodal_loss = accelerator.gather(mse_submodal_loss.repeat(batch_size)).mean().item()
                one_mse_submodal_loss_sum += mse_submodal_loss
                # clip_semantic_loss = accelerator.gather(semantic_clip_loss.repeat(batch_size)).mean().item()
                # one_clip_semantic_loss_sum += clip_semantic_loss
                # semantic_mse_loss = accelerator.gather(semantic_mse_loss.repeat(batch_size)).mean().item()
                # one_mse_semantic_loss_sum += semantic_mse_loss
                # batch_mse_loss_sum += mse_loss

            # batch_clip_semantic_loss = one_clip_semantic_loss_sum / num_keys
            # batch_mse_semantic_loss = batch_mse_semantic_loss_sum / num_keys
            batch_clip_submodal_loss = one_clip_submodal_loss_sum  / num_keys
            batch_mse_submodal_loss = one_mse_submodal_loss_sum / num_keys

            # encoder_hidden_states = model(batch["eeg_data"].to(accelerator.device))
            # noise_pred = frozen_unet(noisy_latents, timesteps, encoder_hidden_states).sample
            # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            # eval_avg_clip_loss = accelerator.gather(batch_clip_loss.repeat(batch_size)).mean().item()
            # eval_avg_mse_loss = accelerator.gather(batch_mse_loss.repeat(batch_size)).mean().item()
            # eval_clip_loss_sum += eval_avg_clip_loss
            # eval_mse_loss_sum += eval_avg_mse_loss
            # eval_clip_semantic_loss_sum += batch_clip_semantic_loss
            # eval_mse_semantic_loss_sum += batch_mse_semantic_loss
            eval_clip_submodal_loss_sum += batch_clip_submodal_loss
            eval_mse_submodal_loss_sum += batch_mse_submodal_loss

        # eval_clip_semantic_loss = eval_clip_semantic_loss_sum / len(data_loader)
        # eval_mse_semantic_loss = eval_mse_semantic_loss_sum / len(data_loader)
        # eval_semantic_loss = eval_mse_semantic_loss + eval_clip_semantic_loss

        eval_clip_submodal_loss = eval_clip_submodal_loss_sum / len(data_loader)
        eval_mse_submodal_loss = eval_mse_submodal_loss_sum / len(data_loader)
        # eval_submodal_loss = eval_mse_submodal_loss + eval_clip_submodal_loss

        alpha = 0.9
        beta = 0.15
        eval_loss = alpha * eval_clip_submodal_loss + (1 - alpha) * eval_mse_submodal_loss # + prior_loss
        # eval_loss = eval_submodal_loss + eval_semantic_loss
    return {
            "eval_mse_submodal_loss": eval_mse_submodal_loss,
            "eval_clip_submodal_loss" : eval_clip_submodal_loss,
            # "eval_submodal_loss": eval_submodal_loss,
            # "eval_mse_semantic_loss": eval_mse_semantic_loss,
            # "eval_clip_semantic_loss" : eval_clip_semantic_loss,
            # "eval_semantic_loss": eval_semantic_loss,
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
    accelerator.init_trackers(
        project_name="ReVIS_Semantic_cross_wandb_0310",
        config=default_configs,
        init_kwargs={"wandb": {"entity": "agathatan"}}
    )
    print(f"Available devices: {accelerator.state.device}")

    # noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    # frozen_unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    # frozen_unet.requires_grad_(False)
    #frozen_unet.to(accelerator.device)

    # sub = 'sub-08'
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07','sub-08','sub-09','sub-10']
    encoder_type = 'semantic'

    SemanticEncoder = ReVisEncoder(default_configs, encoder_type=encoder_type, subjects=subjects, cross_sub = True)
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

    # optimizer = torch.optim.AdamW(itertools.chain(SemanticEncoder.parameters()), lr=default_configs.learning_rate,
    #                               weight_decay=default_configs.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    # train_dataset = SoloEEGDataset(data_path, subjects=[sub], train=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=default_configs.train_batch_size, shuffle=True, num_workers=0,
    #                           drop_last=True)
    # eval_dataset = SoloEEGDataset(data_path, subjects=[sub], train=False)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=default_configs.train_batch_size, shuffle=True,
    #                               num_workers=0, drop_last=True)
    train_dataset = JointEEGDataset(data_path, subjects = subjects, train=True, cross_sub = True)
    train_dataloader = DataLoader(train_dataset, batch_size=default_configs.train_batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)
    eval_dataset = JointEEGDataset(data_path, subjects = subjects, train=False, cross_sub = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=default_configs.train_batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)

    SemanticEncoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        SemanticEncoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    SemanticEncoder.train()
    alpha = 0.9
    beta = 0.15
    for epoch in range(default_configs.num_joint_train_epochs):
        train_clip_semantic_loss_sum = 0
        train_mse_semantic_loss_sum = 0
        train_clip_submodal_loss_sum = 0
        train_mse_submodal_loss_sum = 0

        for idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{default_configs.num_joint_train_epochs}", ncols=100)):
            with accelerator.accumulate(SemanticEncoder):

                # semantic_hidden_states = batch["clip_text_hidden_states"]
                submodal_hidden_states = batch["clip_image_hidden_states"]
                # semantic_features = batch["clip_text_features"]
                submodal_features = batch['clip_img_features']
                batch_size = submodal_features.shape[0]

                # subject_id = extract_id_from_string(sub)
                # subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(accelerator.device)

                # Sample a random timestep for each image
                # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=latents.device)
                # timesteps = timesteps.long()
                # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Initialize a variable to accumulate loss for this batch
                one_train_clip_submodal_loss_sum = 0.  # Initialize a variable to accumulate loss for this batch
                one_train_mse_submodal_loss_sum = 0.
                one_train_clip_semantic_loss_sum = 0.
                one_train_mse_semantic_loss_sum = 0.
                num_keys = len(batch["eeg_data"])
                for key, value in batch["eeg_data"].items():
                    loss = 0.
                    # encoder_hidden_states_pred = SemanticEncoder(x = value, subject_id = key)
                    submodal_hidden_states_pred, submodal_features_pred = SemanticEncoder(x = value.to(accelerator.device), subject_id = key)
                    submodal_features_pred_norm = nn.functional.normalize(submodal_features_pred.flatten(1), dim=-1)
                    submodal_features_norm = nn.functional.normalize(submodal_features.flatten(1), dim=-1)
                    submodal_clip_loss = Util.soft_clip_loss(submodal_features_pred_norm.float(), submodal_features_norm.float())
                    loss += alpha * submodal_clip_loss
                    if default_configs.submodal_mse:
                        # submodal_mse_loss = nn.MSELoss()(submodal_hidden_states_pred_norm, submodal_hidden_states_norm)
                        submodal_mse_loss = F.mse_loss(submodal_hidden_states_pred.float(),submodal_hidden_states.float(),reduction="mean")
                        # loss += default_configs.mse_submodal_mult * submodal_mse_loss
                        loss += (1 - alpha) * submodal_mse_loss

                    # semantic_features_pred_norm = nn.functional.normalize(semantic_features_pred.flatten(1), dim=-1)
                    # semantic_features_norm = nn.functional.normalize(semantic_features.flatten(1), dim=-1)
                    # semantic_clip_loss = Util.soft_clip_loss(semantic_features_pred_norm.float(), semantic_features_norm.float())
                    # loss += beta * semantic_clip_loss
                    # if default_configs.mse_semantic_mult:
                        # semantic_mse_loss = nn.MSELoss()(semantic_hidden_states_pred_norm, semantic_hidden_states_norm)
                        # semantic_mse_loss = F.mse_loss(semantic_hidden_states_pred.float(),semantic_hidden_states.float(),reduction="mean")
                        # loss += default_configs.mse_semantic_mult * semantic_mse_loss
                        # loss += (1-alpha) * semantic_mse_loss

                    # encoder_hidden_states_pred_norm = nn.functional.normalize(encoder_hidden_states_pred.flatten(1),
                    #                                                           dim=-1)
                    # semantic_hidden_states_norm = nn.functional.normalize(semantic_hidden_states.flatten(1), dim=-1)
                    # clip_loss = Util.soft_clip_loss(encoder_hidden_states_pred_norm, semantic_hidden_states_norm)

                    # if default_configs.mse_mult:
                    #     mse_loss = nn.MSELoss()(encoder_hidden_states_pred_norm, semantic_hidden_states_norm)
                    #     loss += default_configs.mse_mult * mse_loss
                    # mse_loss = F.mse_loss(encoder_hidden_states_pred.float(), semantic_hidden_states.float(),
                    #                       reduction="mean")
                    # loss = clip_loss + mse_loss

                    # loss = submodal_clip_loss + default_configs.mse_submodal_mult * submodal_mse_loss \
                    #        + semantic_clip_loss + default_configs.mse_semantic_mult * semantic_mse_loss
                    with torch.autograd.set_detect_anomaly(True):
                        accelerator.backward(loss)
                        optimizer.zero_grad()
                        optimizer.step()


                    train_avg_submodal_clip_loss = accelerator.gather(submodal_clip_loss.repeat(batch_size)).mean().item()
                    one_train_clip_submodal_loss_sum += train_avg_submodal_clip_loss
                    train_avg_submodal_mse_loss = accelerator.gather(submodal_mse_loss.repeat(batch_size)).mean().item()
                    one_train_mse_submodal_loss_sum += train_avg_submodal_mse_loss

                    # train_avg_semantic_clip_loss = accelerator.gather(semantic_clip_loss.repeat(batch_size)).mean().item()
                    # one_train_clip_semantic_loss_sum += train_avg_semantic_clip_loss
                    # train_avg_semantic_mse_loss = accelerator.gather(semantic_mse_loss.repeat(batch_size)).mean().item()
                    # one_train_mse_semantic_loss_sum += train_avg_semantic_mse_loss


                    # train_avg_loss = accelerator.gather(loss.repeat(batch_size)).mean().item()
                    # train_loss_sum += train_avg_loss
                    # train_avg_clip_loss = accelerator.gather(clip_loss.repeat(batch_size)).mean().item()
                    # train_clip_loss_sum += train_avg_clip_loss
                    # train_avg_mse_loss = accelerator.gather(mse_loss.repeat(batch_size)).mean().item()
                    # train_mse_loss_sum += train_avg_mse_loss


                    # loss = F.mse_loss(encoder_hidden_states_pred.float(), semantic_hidden_states.float(), reduction="mean")
                    # accelerator.backward(loss)
                    # optimizer.step()
                    # optimizer.zero_grad()
                    # batch_loss_sum += loss

                one_train_clip_submodal_loss_sum /= num_keys
                one_train_mse_submodal_loss_sum /= num_keys
                # one_train_clip_semantic_loss_sum /= num_keys
                # one_train_mse_semantic_loss_sum /= num_keys

                train_clip_submodal_loss_sum += one_train_clip_submodal_loss_sum
                train_mse_submodal_loss_sum += one_train_mse_submodal_loss_sum
                # train_clip_semantic_loss_sum += one_train_clip_semantic_loss_sum
                # train_mse_semantic_loss_sum += one_train_mse_semantic_loss_sum
                # encoder_hidden_states = SemanticEncoder(batch["eeg_data"].to(accelerator.device))
                # noise_pred = frozen_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                # accelerator.backward(loss)
                # optimizer.step()
                # optimizer.zero_grad()

                # train_avg_clip_loss = accelerator.gather(train_clip_loss_sum.repeat(batch_size)).mean().item()
                # train_clip_loss_sum += train_avg_clip_loss
                # train_avg_mse_loss = accelerator.gather(train_mse_loss_sum.repeat(batch_size)).mean().item()
                # train_mse_loss_sum += train_avg_mse_loss
            # del batch

        eval_loss = evaluate(SemanticEncoder, eval_dataloader, accelerator, subjects=subjects)
        lr_scheduler.step(eval_loss["eval_loss"])
        train_clip_submodal_loss_epoch = train_clip_submodal_loss_sum / len(train_dataloader) # (idx+1)
        train_mse_submodal_loss_epoch = train_mse_submodal_loss_sum / len(train_dataloader) # (idx+1)

        # train_clip_semantic_loss_epoch = train_clip_semantic_loss_sum / len(train_dataloader)  # (idx+1)
        # train_mse_semantic_loss_epoch = train_mse_semantic_loss_sum / len(train_dataloader) # (idx+1)

        eval_mse_submodal_loss = eval_loss["eval_mse_submodal_loss"]
        eval_clip_submodal_loss = eval_loss['eval_clip_submodal_loss']
        # eval_mse_semantic_loss = eval_loss['eval_mse_semantic_loss']
        # eval_clip_semantic_loss = eval_loss['eval_clip_semantic_loss']
        # train_mse_loss_epoch = train_mse_loss_sum / len(train_dataloader)  # (idx+1)
        # accelerator.log({"epoch": epoch + 1, "train_loss": train_loss_epoch, "eval_loss":eval_loss})
        # print(f'epoch: {epoch + 1}, train_loss: {train_loss_epoch}, eval_loss: {eval_loss}')
        accelerator.log(
            {"epoch": epoch + 1, "train_mse_submodal_loss": train_mse_submodal_loss_epoch, "train_clip_submodal_loss": train_clip_submodal_loss_epoch,
             # "train_clip_semantic_loss": train_clip_semantic_loss_epoch,
             "eval_mse_submodal_loss": eval_mse_submodal_loss,
            # "eval_clip_semantic_loss": eval_clip_semantic_loss,
             "eval_clip_submodal_loss": eval_clip_submodal_loss,})
        print(
            f'epoch: {epoch + 1}, train_clip_submodal_loss: {train_clip_submodal_loss_epoch}, eval_clip_submodal_loss: {eval_clip_submodal_loss},  '
            # f'train_clip_semantic_loss: {train_clip_semantic_loss_epoch}, eval_clip_semantic_loss: {eval_clip_semantic_loss},'
            f'train_mse_submodal_loss: {train_mse_submodal_loss_epoch}, eval_mse_submodal_loss: {eval_mse_submodal_loss},   '
              f'eval_loss: {eval_loss}')

        if (epoch + 1) % 10 == 0:
            # Save the model every 10 epochs
            save_model_path = os.path.join(output_dir, encoder_type+'_cross_Encoder_0310')
            os.makedirs(save_model_path, exist_ok=True)
            file_path = f'{save_model_path}/{epoch+1}.pth'
            torch.save(SemanticEncoder.state_dict(), file_path) # torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter.bin")
            print(f"Model saved in {file_path}!")


    accelerator.end_training()

if __name__ == '__main__':
    main()