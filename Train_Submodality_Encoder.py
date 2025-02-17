import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import itertools
import os
from utils.Configs import Configs
import re
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel
import clip
from load_eegdatasets import SoloEEGDataset
from torch.utils.data import DataLoader

from layers.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
from model.ReVisModels import ReVisEncoder

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
# logging_dir = data_config["Submodality_logging_dir"]

default_configs = Configs()

def evaluate(model, data_loader, subject_ids, noise_scheduler, unet, accelerator):
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)

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

            texts = ["" for _ in range(batch_size)]
            text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
            with torch.no_grad():
                text_features = text_encoder(text_inputs)
                encoder_hidden_states = text_features[0]

            submodal_tokens = model(batch["eeg_data"].to(accelerator.device), sub_ids=subject_ids)

            encoder_hidden_states = torch.cat([encoder_hidden_states, submodal_tokens], dim=1)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            # noise_pred = frozen_unet(noisy_latents, timesteps, encoder_hidden_states).sample
            # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            eval_avg_loss = accelerator.gather(loss.repeat(batch_size)).mean().item()
            eval_loss_sum += eval_avg_loss
        eval_loss = eval_loss_sum / len(data_loader)

    return eval_loss

def main():
    # logging_directory = os.path.join(output_dir, logging_dir)
    # accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_directory)
    accelerator = Accelerator(
        log_with=default_configs.log_with,
        # project_config=accelerator_project_config,
        gradient_accumulation_steps=default_configs.gradient_accumulation_steps
    )
    accelerator.init_trackers(
        project_name="ReVIS_Submodal_wandb_0213",
        config=default_configs,
        init_kwargs={"wandb": {"entity": "agathatan"}}
    )

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.to(accelerator.device)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)

    sub = 'sub-08'

    # SemanticEncoder = ReVisEncoder(default_configs, encoder_type='semantic')
    # semantic_model_path = os.path.join(output_dir, 'semanticEncoder', sub)
    # ckpt_path = f'{semantic_model_path}/{default_configs.num_train_epochs}.pth'
    #
    # if os.path.exists(ckpt_path):
    #     state_dict = torch.load(ckpt_path)
    #     SemanticEncoder.load_state_dict(state_dict)
    #     SemanticEncoder.requires_grad_(False)
    # else:
    #     raise FileNotFoundError("semantic encoder model not found at {}, please train SemanticEncoder firstly".format(ckpt_path))

    encoder_type = 'submodality'
    SubmodalityEncoder = ReVisEncoder(default_configs, encoder_type=encoder_type)

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=default_configs.extra_submodal_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    unet.to(accelerator.device)

    # optimizer
    params_to_opt = itertools.chain(SubmodalityEncoder.parameters(), adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=default_configs.learning_rate, weight_decay=default_configs.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Data loader
    train_dataset = SoloEEGDataset(data_path, subjects=[sub], train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=default_configs.train_batch_size, shuffle=True, num_workers=0,
                              drop_last=True)
    eval_dataset = SoloEEGDataset(data_path, subjects=[sub], train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=default_configs.train_batch_size, shuffle=True,
                                 num_workers=0, drop_last=True)

    SubmodalityEncoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        SubmodalityEncoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    SubmodalityEncoder.train()

    for epoch in range(default_configs.num_train_epochs):
        train_loss_sum = 0
        for idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{default_configs.num_train_epochs}", ncols=100)):
            with accelerator.accumulate(SubmodalityEncoder):
                latents = batch["vae_img_features"]
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                subject_id = extract_id_from_string(sub)
                subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(accelerator.device)

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=latents.device)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # with torch.no_grad():
                #     encoder_hidden_states = SemanticEncoder(batch["eeg_data"].to(accelerator.device),
                #                                             sub_ids=subject_ids)
                texts = ["" for _ in range(batch_size)]
                text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
                with torch.no_grad():
                    text_features = text_encoder(text_inputs)
                    encoder_hidden_states = text_features[0]

                # clip_text_hidden_states = F.normalize(text_features, dim=-1).detach()

                submodal_tokens = SubmodalityEncoder(batch["eeg_data"].to(accelerator.device),sub_ids=subject_ids)
                encoder_hidden_states = torch.cat([encoder_hidden_states, submodal_tokens], dim=1)


                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean().item()
                if accelerator.is_main_process:
                    train_loss_sum += avg_loss
            del batch

        eval_loss = evaluate(SubmodalityEncoder, eval_dataloader, subject_ids, noise_scheduler, unet, accelerator)
        lr_scheduler.step(eval_loss)
        train_loss_epoch = train_loss_sum / len(train_dataloader) # (idx+1)
        accelerator.log({"epoch": epoch + 1, "train_loss": train_loss_epoch, "eval_loss":eval_loss})
        print(f'epoch: {epoch + 1}, train_loss: {train_loss_epoch}, eval_loss: {eval_loss}')

        if (epoch + 1) % 10 == 0:
            # Save the model every 10 epochs
            save_model_path = os.path.join(output_dir, encoder_type+'SubmodalEncoder_0213', sub, f'epoch-{epoch+1}')
            os.makedirs(save_model_path, exist_ok=True)
            # file_path = f'{save_model_path}/{epoch+1}.pth'
            # torch.save(SemanticEncoder.state_dict(), file_path)
            # torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter.bin")
            accelerator.save_state(save_model_path)
            print(f"Model saved in {save_model_path}!")



    accelerator.end_training()
    # import torch
    # ckpt = "checkpoint-50000/pytorch_model.bin"
    # sd = torch.load(ckpt, map_location="cpu")
    # image_proj_sd = {}
    # ip_sd = {}
    # for k in sd:
    #     if
    # k.startswith("unet"):
    # pass
    # elif k.startswith("image_proj_model"):
    # image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    # elif k.startswith("adapter_modules"):
    # ip_sd[k.replace("adapter_modules.", "")] = sd[k]
    #
    # torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter.bin")

if __name__ == '__main__':
    main()
