import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import clip
from torch.nn import functional as F
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from torchvision import transforms
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from accelerate import Accelerator
from transformers import CLIPTextModel, AutoProcessor, AutoModelForCausalLM
from PIL import Image



import os
import open_clip
# proxy = 'http://10.252.99.111:7897'
# proxy = 'http://127.0.0.1:7897'
# os.environ['http_proxy'] = proxy
# os.environ['https_proxy'] = proxy
# cuda_device_count = torch.cuda.device_count()
# print(cuda_device_count)
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
accelerator = Accelerator()

import json
# Load the configuration from the JSON file
data_config_path = "data_config.json"
with open(data_config_path, "r") as data_config_file:
    data_config = json.load(data_config_file)

# Access the paths from the config
data_path = data_config["data_path"]
img_directory_training = data_config["img_directory_training"]
img_directory_test = data_config["img_directory_test"]
pretrained_model_path = data_config["pretrained_model_path"]
ip_cache_dir = data_config["ip_cache_dir"]

# IP_image_encoder = CLIPVisionModelWithProjection.from_pretrained(ip_cache_dir,subfolder="sdxl_models/image_encoder")
# IP_image_encoder.requires_grad_(False)
# IP_image_encoder = IP_image_encoder.to(accelerator.device)
# # IP_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
#
# # clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
# # clip_image_encoder.requires_grad_(False)
# # clip_image_encoder = clip_image_encoder.to(accelerator.device)
# # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
#
# # 加载与 image_encoder 对应的 text_encoder
# # clip_text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
# # clip_text_encoder.requires_grad_(False)
# # clip_text_encoder = clip_text_encoder.to(accelerator.device)
# #
# # text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
# # text_encoder.requires_grad_(False)
# # text_encoder = text_encoder.to(accelerator.device)
#
# GITprocessor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
# GITmodel = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
# GITmodel.requires_grad_(False)
#
#
# from diffusers import AutoencoderKL
# from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
#
# pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.bfloat16, variant="fp16")
# # vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
# # vae = pipe.vae.to(dtype=torch.float16)
# # vae.requires_grad_(False)
# vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder='vae').to("cuda")
# vae.to(accelerator.device)

clip_model_type = 'ViT-H-14'
SD_model_type = 'SDXL_turbo'
# vlmodel, processor, feature_extractor = open_clip.create_model_and_transforms(
#     clip_model_type, pretrained='laion2b_s32b_b79k', precision='fp16', device = accelerator.device)
# vlmodel.visual.output_tokens = True
# # vlmodel.text.output_tokens = True
# vlmodel.requires_grad_(False)

# model_type = 'ViT-L-14'


class SoloEEGDataset(Dataset):
    """
    load one subject's data
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """

    def __init__(self, data_path, subjects=None, train=True, time_window=[0, 1.0], avg=True, cross_sub = False, norm_embs = False):
        self.cross_sub = cross_sub
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.avg = avg
        self.norm_embs = norm_embs

        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)
        if self.train:
            self.file_name = 'preprocessed_eeg_training.npy'
            self.n_classes = 1654
            self.samples_per_class = 10
            self.repeat_times = 4
        else:
            self.file_name = 'preprocessed_eeg_test.npy'
            self.n_classes = 200
            self.samples_per_class = 1
            self.repeat_times = 80

        self.eeg_data_all, self.labels, self.category ,self.texts, self.images = self.load_data()

        self.eeg_data_all = self.extract_eeg(self.eeg_data_all, time_window)

        # clip_image_features_filename = os.path.join(f'{model_type}_clip_image_features_train.pt') if self.train else os.path.join(
        #     f'{model_type}_clip_image_features_test.pt')
        # if os.path.exists(clip_image_features_filename):
        #     saved_clip_features = torch.load(clip_image_features_filename, map_location = accelerator.device)
        #     self.clip_img_hidden_states = saved_clip_features['clip_img_hidden_states']
        # else:
        #     self.clip_img_hidden_states = self.ClipImageEncoder(self.images)
        #     torch.save({
        #         'clip_img_hidden_states': self.clip_img_hidden_states.cpu(),
        #     }, clip_image_features_filename)
        #     print(f'clip_img_hidden_states saved to {clip_image_features_filename}')
        # if train:
        #     self.test_features = torch.load('./variables/ViT-H-14_features_train.pt')
        # else:
        #     self.test_features = torch.load('./variables/ViT-H-14_features_test.pt')

        clip_img_features_filename = os.path.join(f'{clip_model_type}_clip_img_features_train.pt') if self.train else os.path.join(
            f'{clip_model_type}_clip_img_features_test.pt')
        if os.path.exists(clip_img_features_filename):
            saved_features = torch.load(clip_img_features_filename)
            self.clip_img_features = saved_features['clip_img_features']
            # self.clip_image_hidden_states = saved_features['clip_image_hidden_states']
            # self.vae_img_features = saved_features['vae_img_features']
        else:
            self.clip_img_features, _ = self.ClipImageEncoder(self.images)
            # self.vae_img_features = self.VAEImageEncoder(self.images)
            torch.save({
                # 'clip_image_hidden_states' : self.clip_image_hidden_states.cpu(),
                'clip_img_features': self.clip_img_features.cpu(),
                # 'vae_img_features': self.vae_img_features.cpu(),
            }, clip_img_features_filename)
            print(f'img_features saved to {clip_img_features_filename}')

        clip_img_hidden_states_filename = os.path.join(
            f'{clip_model_type}_clip_img_hidden_states_train.pt') if self.train else os.path.join(
            f'{clip_model_type}_clip_img_hidden_states_test.pt')
        if os.path.exists(clip_img_hidden_states_filename):
            saved_features = torch.load(clip_img_hidden_states_filename)
            # self.clip_img_features = saved_features['clip_img_features']
            self.clip_image_hidden_states = saved_features['clip_image_hidden_states']
            # self.vae_img_features = saved_features['vae_img_features']
        else:
            _, self.clip_image_hidden_states = self.ClipImageEncoder(self.images)
            # self.vae_img_features = self.VAEImageEncoder(self.images)
            torch.save({
                'clip_image_hidden_states': self.clip_image_hidden_states.cpu(),
                # 'clip_img_features': self.clip_img_features.cpu(),
                # 'vae_img_features': self.vae_img_features.cpu(),
            }, clip_img_hidden_states_filename)
            print(f'img_features saved to {clip_img_hidden_states_filename}')

        vae_img_features_filename = os.path.join(
            f'{clip_model_type}_vae_img_features_train.pt') if self.train else os.path.join(
            f'{clip_model_type}_vae_img_features_test.pt')
        if os.path.exists(vae_img_features_filename):
            saved_features = torch.load(vae_img_features_filename)
            # self.clip_img_features = saved_features['clip_img_features']
            # self.clip_image_hidden_states = saved_features['clip_image_hidden_states']
            self.vae_img_features = saved_features['vae_img_features']
        else:
            # self.clip_img_features, self.clip_image_hidden_states = self.ClipImageEncoder(self.images)
            self.vae_img_features = self.VAEImageEncoder(self.images)
            torch.save({
                # 'clip_image_hidden_states': self.clip_image_hidden_states.cpu(),
                # 'clip_img_features': self.clip_img_features.cpu(),
                'vae_img_features': self.vae_img_features.cpu(),
            }, vae_img_features_filename)
            print(f'img_features saved to {vae_img_features_filename}')

        text_features_filename = os.path.join(f'{clip_model_type}_text_features_train.pt') if self.train else os.path.join(
            f'{clip_model_type}_text_features_test.pt')
        if os.path.exists(text_features_filename):
            saved_features = torch.load(text_features_filename)
            self.clip_text_features = saved_features['clip_text_features']
            # self.clip_text_hidden_states = saved_features['clip_text_hidden_states']
        else:
            self.clip_text_features = self.ClipTextencoder(self.texts)
            torch.save({
                'clip_text_features': self.clip_text_features.cpu(),
                # 'clip_text_hidden_states': self.clip_text_hidden_states.cpu(),
            }, text_features_filename)
            print(f'text_features_filename saved to {text_features_filename}')

        IP_image_features_filename = os.path.join(
            f'{SD_model_type}_IP_image_features_train.pt') if self.train else os.path.join(
            f'{SD_model_type}_IP_image_features_test.pt')
        if os.path.exists(IP_image_features_filename):
            saved_IP_img_features = torch.load(IP_image_features_filename)
            self.IP_img_features = saved_IP_img_features['IP_img_features']
        else:
            self.IP_img_features = self.IPImgaeEnocder(self.images)
            torch.save({
                'IP_img_features': self.IP_img_features.cpu(),
            }, IP_image_features_filename)
            print(f'IP_img_features saved to {IP_image_features_filename}')

    def load_eeg_data(self, subject):
        # if self.train:
        #     file_name = 'preprocessed_eeg_training.npy'
        #     n_classes = 1654
        #     samples_per_class = 10
        #     repeat_times = 4
        # else:
        #     file_name = 'preprocessed_eeg_test.npy'
        #     n_classes = 200
        #     samples_per_class = 1
        #     repeat_times = 80

        file_path = os.path.join(self.data_path, subject, self.file_name)
        data = np.load(file_path, allow_pickle=True)

        preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
        times = torch.from_numpy(data['times']).detach()[20:]
        # times = torch.from_numpy(data['times']).detach()[50:]
        ch_names = data['ch_names']
        self.times = times
        self.ch_names = ch_names

        eeg_data_list = []
        label_list = []
        for i in range(self.n_classes):
            start_index = i * self.samples_per_class
            eeg_data = preprocessed_eeg_data[start_index: start_index + self.samples_per_class]
            if self.avg:
                labels = torch.full((self.samples_per_class,), i, dtype=torch.long).detach()
            else:
                labels = torch.full((self.samples_per_class * self.repeat_times,), i, dtype=torch.long).detach()
            if self.avg:
                if self.train:
                    eeg_data = torch.mean(eeg_data, 1)
                else:
                    indices = torch.randperm(self.repeat_times)[:4]
                    random_samples = eeg_data[:, indices, :, :]
                    eeg_data = torch.mean(random_samples, 1)
                # eeg_data = torch.mean(eeg_data, 1)
            eeg_data_list.append(eeg_data)
            label_list.append(labels)
        return eeg_data_list, label_list

    def load_data(self):
        texts = []
        category = []
        images = []

        if self.train:
            directory = img_directory_training
        else:
            directory = img_directory_test

        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()

        # load classes data
        category_folders = dirnames
        for dir in category_folders:
            try:
                idx = dir.index('_')
                category_description = dir[idx + 1:]
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")
                continue

            # new_class_description = f"This picture is {category_description}"
            category.append(category_description)
        # len(category) = 1654 / 200

        # load image data
        image_folders = dirnames
        for folder in image_folders:
            floder_path = os.path.join(directory, folder)
            all_images = [img for img in os.listdir(floder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            images.extend(os.path.join(floder_path, img) for img in all_images)
        # len(images) = 16540 / 200
        texts_filename = os.path.join("./THINGS", "texts_set", "training_texts.pt") if self.train else (
            os.path.join("THINGS", "texts_set", "test_texts.pt"))
        if os.path.exists(texts_filename):
            texts = torch.load(texts_filename)
        else:
            os.makedirs(os.path.dirname(texts_filename), exist_ok=True)
            for img in images:
                image = Image.open(img)
                pixel_values = GITprocessor(images=image, return_tensors="pt").pixel_values

                generated_ids = GITmodel.generate(pixel_values=pixel_values, max_length=50)
                new_text_description = GITprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(new_text_description)
                texts.append(new_text_description)
            torch.save(texts, texts_filename)

        # len(texts) = 16540 / 200

        # load eeg data
        for subject in self.subjects:
            eeg_data_list, label_list = self.load_eeg_data(subject)

        # train_data_list[0].shape = torch.Size([10, 4, 63, 100])
        # test_data_list[0].shape = torch.Size([1, 80, 63, 100])

        eeg_data_tensor = torch.cat(eeg_data_list, dim=0).view(-1, *eeg_data_list[0].shape[-2:])
        print("eeg_data_tensor", eeg_data_tensor.shape)
        # when avg = False, torch.Size([66160, 63, 100]) / torch.Size([16000, 63, 100])
        # when avg = True, torch.Size([16540, 63, 100]) / torch.Size([200, 63, 100])

        label_tensor = torch.cat(label_list, dim=0)
        print("label_tensor", label_tensor.shape)
        # when avg = False, torch.Size([66160]) / torch.Size([16000])
        # when avg = True, torch.Size([16540]) / torch.Size([200])

        return eeg_data_tensor, label_tensor, category , texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Check the window bounding
        indices = (self.times >= start) & (self.times <= end)
        extracted_data = eeg_data[..., indices]

        return extracted_data

    def ClipTextencoder(self, texts):
        batch_size = 8
        clip_text_features_list = []
        # clip_text_hidden_states_list = []

        text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
        for i in range(0, len(texts), batch_size):
            batch_text_inputs = text_inputs[i: i + batch_size]
            with torch.no_grad():
                batch_text_features = vlmodel.encode_text(batch_text_inputs)
                # batch_text_features = batch_text_features.text_embeds

            clip_text_features_list.append(batch_text_features)
            # clip_text_hidden_states_list.append(batch_text_hidden_states)
        # clip_text_features = F.normalize(text_features, dim=-1).detach()
        clip_text_features = torch.cat(clip_text_features_list, dim = 0)
        # clip_text_hidden_states = torch.cat(clip_text_hidden_states_list, dim = 0)
        if self.norm_embs:
            clip_text_features = F.normalize(clip_text_features, dim=-1).detach()

        return clip_text_features

    # def ClipTextencoder(self, texts):
    #     batch_size = 16
    #     clip_text_features_list = []
    #     text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
    #     for i in range(0, len(texts), batch_size):
    #         batch_text_inputs = text_inputs[i: i + batch_size]
    #         with torch.no_grad():
    #             batch_text_features = clip_text_encoder(batch_text_inputs)
    #             # batch_text_features = batch_text_features.text_embeds
    #             embeds = clip_text_encoder.text_projection(batch_text_features.last_hidden_state)
    #             embeds_pooled = batch_text_features.text_embeds
    #             if self.norm_embs:
    #                 embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1, keepdim=True)
    #
    #         clip_text_features_list.append(embeds)
    #     # clip_text_features = F.normalize(text_features, dim=-1).detach()
    #     clip_text_hidden_states = torch.cat(clip_text_features_list, dim = 0)
    #
    #     return clip_text_hidden_states

    # def ClipTextencoder(self, texts):
    #     text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
    #     with torch.no_grad():
    #         text_features = clip_text_encoder(text_inputs)
    #         text_features = text_features.text_embeds
    #
    #     # clip_text_features = F.normalize(text_features, dim=-1).detach()
    #     clip_text_features = text_features
    #     return clip_text_features

    # def SDTextencoder(self, texts):
    #     batch_size = 16
    #     clip_text_hidden_states_list = []
    #     # TODO : replace clip_text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    #     text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
    #     for i in range(0, len(texts), batch_size):
    #         batch_text_inputs = text_inputs[i: i + batch_size]
    #         with torch.no_grad():
    #             batch_text_hidden_states = text_encoder(batch_text_inputs)
    #             batch_text_hidden_states = batch_text_hidden_states[0]
    #
    #         clip_text_hidden_states_list.append(batch_text_hidden_states)
    #
    #
    #     # clip_text_hidden_states = F.normalize(text_features, dim=-1).detach()
    #     clip_text_hidden_states = torch.cat(clip_text_hidden_states_list, dim = 0)
    #     if self.norm_embs:
    #         SD_clip_text_hidden_states = F.normalize(clip_text_hidden_states, dim=-1).detach()
    #     return SD_clip_text_hidden_states

    def ClipImageEncoder(self, images):
        # Prevent memory overflow on the GPU
        batch_size = 4
        image_features_list = []
        image_hidden_states_list = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([processor(Image.open(img).convert("RGB")) for img in batch_images]).to(accelerator.device).to(dtype=torch.float16)
            # image_inputs = torch.stack([processor(images = Image.open(img).convert("RGB"), return_tensors="pt").pixel_values[0] for img in batch_images]).to(accelerator.device)
            # image_inputs = processor(images=[Image.open(img).convert("RGB") for img in batch_images],
            #                          return_tensors="pt").pixel_values.to(accelerator.device)

            with torch.no_grad():
                # vlmodel.visual.output_tokens = True
                batch_image_features, batch_img_hidden_states = vlmodel.encode_image(image_inputs)
                # batch_image_features = vlmodel.encode_image(image_inputs)
                # batch_image_features = batch_image_features.image_embeds
                # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
            image_hidden_states_list.append(batch_img_hidden_states)

        clip_image_features = torch.cat(image_features_list, dim=0)
        clip_image_hidden_states = torch.cat(image_hidden_states_list, dim =0)
        if self.norm_embs:
            clip_image_features = F.normalize(clip_image_features, dim=-1).detach()

        return clip_image_features, clip_image_hidden_states

    # def ClipImageEncoder(self, images):
    #     # Prevent memory overflow on the GPU
    #     batch_size = 8
    #     image_features_list = []
    #
    #     for i in range(0, len(images), batch_size):
    #         batch_images = images[i:i + batch_size]
    #         # image_inputs = torch.stack([processor(images = Image.open(img).convert("RGB"), return_tensors="pt").pixel_values[0] for img in batch_images]).to(device)
    #         image_inputs = processor(images=[Image.open(img).convert("RGB") for img in batch_images],
    #                                  return_tensors="pt").pixel_values.to(accelerator.device)
    #
    #         with torch.no_grad():
    #             batch_image_features = clip_image_encoder(image_inputs)
    #             batch_image_embeds = batch_image_features.last_hidden_state
    #             # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
    #             # batch_image_features = F.normalize(batch_image_features, dim=-1).detach()
    #             batch_image_embeds = clip_image_encoder.vision_model.post_layernorm(batch_image_embeds)
    #             batch_image_hidden_states = clip_image_encoder.visual_projection(batch_image_embeds)
    #             if self.norm_embs:
    #                 # normalize all tokens by cls token's norm
    #                 batch_image_hidden_states = batch_image_hidden_states / torch.norm(batch_image_hidden_states[:, 0], dim=-1).reshape(-1, 1, 1)
    #
    #         image_features_list.append(batch_image_hidden_states)
    #
    #     clip_img_hidden_states = torch.cat(image_features_list, dim=0)
    #
    #     return clip_img_hidden_states


    def VAEImageEncoder(self, images):
        transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        batch_size = 2
        image_emdeddings_list = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([transform(Image.open(img).convert("RGB")) for img in batch_images]).to(accelerator.device)
            if torch.isnan(image_inputs).any() or torch.isinf(image_inputs).any():
                raise ValueError("NaN or Inf found in input images!")

            with torch.no_grad():
                batch_image_emdedding = vae.encode(image_inputs).latent_dist.mode() * vae.config.scaling_factor #0.18215
                # shape [1, 4, 64, 64]

            # 检查是否有 NaN 或 Inf
            if torch.isnan(batch_image_emdedding).any() or torch.isinf(batch_image_emdedding).any():
                # 终止执行，以免 NaN 继续传播
                raise ValueError("Detected NaN or Inf in batch_image_emdedding!")
            image_emdeddings_list.append(batch_image_emdedding)

        vae_image_features = torch.cat(image_emdeddings_list, dim=0)
        return vae_image_features

    def IPImgaeEnocder(self, images):
        batch_size = 8
        image_features_list = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            # image_inputs = torch.stack([processor(images = Image.open(img).convert("RGB"), return_tensors="pt").pixel_values[0] for img in batch_images]).to(device)
            # processor是从ViT-H加载的
            image_inputs = torch.stack([processor(Image.open(img).convert("RGB")) for img in batch_images]).to(accelerator.device).to(dtype=torch.float16)
            # IP_processor是之前从ViT-L加载的
            # image_inputs = IP_processor(images=[Image.open(img).convert("RGB") for img in batch_images], return_tensors="pt").pixel_values.to(accelerator.device)

            with torch.no_grad():
                batch_image_features = IP_image_encoder(image_inputs)
                batch_image_embeds = batch_image_features.image_embeds

            image_features_list.append(batch_image_embeds)

        IP_image_features = torch.cat(image_features_list, dim=0)
        if self.norm_embs:
            IP_image_features = F.normalize(IP_image_features, dim=-1).detach()

        return IP_image_features

    def __getitem__(self, index):
        """Get data and label corresponding to index"""
        eeg_data = self.eeg_data_all[index]
        label = self.labels[index]

        if self.train:
            if self.avg:
                index_n_sub_train = self.n_cls * 10
                img_index = index % index_n_sub_train
                category_index = (index % index_n_sub_train) // (10)
            else:
                index_n_sub_train = self.n_cls * 10 * 4
                img_index = (index % index_n_sub_train) // (4)
                category_index = (index % index_n_sub_train) // (10 * 4)
        else:

            if self.avg:
                index_n_sub_test = self.n_cls * 1
                img_index = index % index_n_sub_test
                category_index = (index % index_n_sub_test) // (1)
            else:
                index_n_sub_test = self.n_cls * 1 * 80
                img_index = (index % index_n_sub_test) // (80)
                category_index = (index % index_n_sub_test) // (1 * 80)

        category = self.category[category_index]
        text = self.texts[img_index]
        img = self.images[img_index]

        # clip_text_features = self.clip_text_features[text_index]
        # clip_text_hidden_states = self.clip_text_hidden_states[text_index]

        # SD_clip_text_hidden_states = self.SD_clip_text_hidden_states[img_index]
        clip_img_features = self.clip_img_features[img_index]
        # clip_img_hidden_states = self.clip_img_hidden_states[img_index]
        vae_img_features = self.vae_img_features[img_index]
        clip_text_features = self.clip_text_features[img_index]
        IP_img_features = self.IP_img_features[img_index]
        clip_image_hidden_states = self.clip_image_hidden_states[img_index]
        # clip_text_hidden_states = self.clip_text_hidden_states[img_index]
        # test_features = self.test_features[img_index]

        return {
            "eeg_data": eeg_data,
            "label" : label,
            # "clip_text_hidden_states": clip_text_hidden_states,
            "clip_text_features":clip_text_features,
            "clip_img_features" : clip_img_features,
            "vae_img_features" : vae_img_features,
            "clip_image_hidden_states" : clip_image_hidden_states,
            # "SD_clip_text_hidden_states": SD_clip_text_hidden_states,
            # "clip_img_hidden_states": clip_img_hidden_states,
            "IP_img_features" : IP_img_features,
            "text" :text,
            "img" :img,
            "category" :category,
            # "test_features": test_features,

        }

    def __len__(self):
        # len = 16540/200 when avg = True, len = 66160/16000 when avg = False
        return self.eeg_data_all.shape[0]

class JointEEGDataset(Dataset):
    """
    load multi-subjects' data
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """

    def __init__(self, data_path, subjects=None, train=True, time_window=[0, 1.0], avg=True, cross_sub = True, norm_embs=False):
        self.norm_embs = norm_embs
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.cross_sub = cross_sub
        self.n_cls = 1654 if train else 200
        self.avg = avg

        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)
        if self.train:
            self.file_name = 'preprocessed_eeg_training.npy'
            self.n_classes = 1654
            self.samples_per_class = 10
            self.repeat_times = 4
        else:
            self.file_name = 'preprocessed_eeg_test.npy'
            self.n_classes = 200
            self.samples_per_class = 1
            self.repeat_times = 80

        self.eeg_data_all, self.labels, self.category , self.texts, self.images = self.load_data()

        self.eeg_data_all = self.extract_eeg(self.eeg_data_all, time_window)

        # clip_image_features_filename = os.path.join(f'{model_type}_clip_image_features_train.pt') if self.train else os.path.join(
        #     f'{model_type}_clip_image_features_test.pt')
        # if os.path.exists(clip_image_features_filename):
        #     saved_clip_features = torch.load(clip_image_features_filename, map_location = accelerator.device)
        #     self.clip_img_hidden_states = saved_clip_features['clip_img_hidden_states']
        # else:
        #     self.clip_img_hidden_states = self.ClipImageEncoder(self.images)
        #     torch.save({
        #         'clip_img_hidden_states': self.clip_img_hidden_states.cpu(),
        #     }, clip_image_features_filename)
        #     print(f'clip_img_hidden_states saved to {clip_image_features_filename}')

        clip_img_features_filename = os.path.join(
            f'{clip_model_type}_clip_img_features_train.pt') if self.train else os.path.join(
            f'{clip_model_type}_clip_img_features_test.pt')
        if os.path.exists(clip_img_features_filename):
            saved_features = torch.load(clip_img_features_filename)
            self.clip_img_features = saved_features['clip_img_features']
            # self.clip_image_hidden_states = saved_features['clip_image_hidden_states']
            # self.vae_img_features = saved_features['vae_img_features']
        else:
            self.clip_img_features, _ = self.ClipImageEncoder(self.images)
            # self.vae_img_features = self.VAEImageEncoder(self.images)
            torch.save({
                # 'clip_image_hidden_states' : self.clip_image_hidden_states.cpu(),
                'clip_img_features': self.clip_img_features.cpu(),
                # 'vae_img_features': self.vae_img_features.cpu(),
            }, clip_img_features_filename)
            print(f'img_features saved to {clip_img_features_filename}')

        clip_img_hidden_states_filename = os.path.join(
            f'{clip_model_type}_clip_img_hidden_states_train.pt') if self.train else os.path.join(
            f'{clip_model_type}_clip_img_hidden_states_test.pt')
        if os.path.exists(clip_img_hidden_states_filename):
            saved_features = torch.load(clip_img_hidden_states_filename)
            # self.clip_img_features = saved_features['clip_img_features']
            self.clip_image_hidden_states = saved_features['clip_image_hidden_states']
            # self.vae_img_features = saved_features['vae_img_features']
        else:
            _, self.clip_image_hidden_states = self.ClipImageEncoder(self.images)
            # self.vae_img_features = self.VAEImageEncoder(self.images)
            torch.save({
                'clip_image_hidden_states': self.clip_image_hidden_states.cpu(),
                # 'clip_img_features': self.clip_img_features.cpu(),
                # 'vae_img_features': self.vae_img_features.cpu(),
            }, clip_img_hidden_states_filename)
            print(f'img_features saved to {clip_img_hidden_states_filename}')

        vae_img_features_filename = os.path.join(
            f'{clip_model_type}_vae_img_features_train.pt') if self.train else os.path.join(
            f'{clip_model_type}_vae_img_features_test.pt')
        if os.path.exists(vae_img_features_filename):
            saved_features = torch.load(vae_img_features_filename)
            # self.clip_img_features = saved_features['clip_img_features']
            # self.clip_image_hidden_states = saved_features['clip_image_hidden_states']
            self.vae_img_features = saved_features['vae_img_features']
        else:
            # self.clip_img_features, self.clip_image_hidden_states = self.ClipImageEncoder(self.images)
            self.vae_img_features = self.VAEImageEncoder(self.images)
            torch.save({
                # 'clip_image_hidden_states': self.clip_image_hidden_states.cpu(),
                # 'clip_img_features': self.clip_img_features.cpu(),
                'vae_img_features': self.vae_img_features.cpu(),
            }, vae_img_features_filename)
            print(f'img_features saved to {vae_img_features_filename}')

        text_features_filename = os.path.join(
            f'{clip_model_type}_text_features_train.pt') if self.train else os.path.join(
            f'{clip_model_type}_text_features_test.pt')
        if os.path.exists(text_features_filename):
            saved_features = torch.load(text_features_filename)
            self.clip_text_features = saved_features['clip_text_features']
            # self.clip_text_hidden_states = saved_features['clip_text_hidden_states']
        else:
            self.clip_text_features = self.ClipTextencoder(self.texts)
            torch.save({
                'clip_text_features': self.clip_text_features.cpu(),
                # 'clip_text_hidden_states': self.clip_text_hidden_states.cpu(),
            }, text_features_filename)
            print(f'text_features_filename saved to {text_features_filename}')

        IP_image_features_filename = os.path.join(
            f'{SD_model_type}_IP_image_features_train.pt') if self.train else os.path.join(
            f'{SD_model_type}_IP_image_features_test.pt')
        if os.path.exists(IP_image_features_filename):
            saved_IP_img_features = torch.load(IP_image_features_filename)
            self.IP_img_features = saved_IP_img_features['IP_img_features']
        else:
            self.IP_img_features = self.IPImgaeEnocder(self.images)
            torch.save({
                'IP_img_features': self.IP_img_features.cpu(),
            }, IP_image_features_filename)
            print(f'IP_img_features saved to {IP_image_features_filename}')


    def load_eeg_data(self, subject):
        # if self.train:
        #     file_name = 'preprocessed_eeg_training.npy'
        #     n_classes = 1654
        #     samples_per_class = 10
        #     repeat_times = 4
        # else:
        #     file_name = 'preprocessed_eeg_test.npy'
        #     n_classes = 200
        #     samples_per_class = 1
        #     repeat_times = 80

        file_path = os.path.join(self.data_path, subject, self.file_name)
        data = np.load(file_path, allow_pickle=True)

        preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
        times = torch.from_numpy(data['times']).detach()[20:]
        ch_names = data['ch_names']
        self.times = times
        self.ch_names = ch_names

        eeg_data_list = []
        label_list = []
        for i in range(self.n_classes):
            start_index = i * self.samples_per_class
            eeg_data = preprocessed_eeg_data[start_index: start_index + self.samples_per_class]
            if self.avg:
                labels = torch.full((self.samples_per_class,), i, dtype=torch.long).detach()
            else:
                labels = torch.full((self.samples_per_class * self.repeat_times,), i, dtype=torch.long).detach()
            if self.avg:
                if self.train:
                    eeg_data = torch.mean(eeg_data, 1)
                else:
                    indices = torch.randperm(self.repeat_times)[:4]
                    random_samples = eeg_data[:, indices, :, :]
                    eeg_data = torch.mean(random_samples, 1)
                # eeg_data = torch.mean(eeg_data, 1)
            eeg_data_list.append(eeg_data)
            label_list.append(labels)
        return eeg_data_list, label_list

    def load_data(self):
        texts = []
        category = []
        images = []

        if self.train:
            directory = img_directory_training
        else:
            directory = img_directory_test

        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()

        # load category data
        category_folders = dirnames
        for dir in category_folders:
            try:
                idx = dir.index('_')
                category_description = dir[idx + 1:]
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")
                continue

            # new_category_description = f"This picture is {category_description}"
            category.append(category_description)
        # len(category) = 1654 / 200

        # load image data
        image_folders = dirnames
        for folder in image_folders:
            floder_path = os.path.join(directory, folder)
            all_images = [img for img in os.listdir(floder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            images.extend(os.path.join(floder_path, img) for img in all_images)
        # len(images) = 16540 / 200
        texts_filename = os.path.join("./THINGS", "texts_set", "training_texts.pt") if self.train else (
            os.path.join("THINGS", "texts_set", "test_texts.pt"))
        if os.path.exists(texts_filename):
            texts = torch.load(texts_filename)
        else:
            os.makedirs(os.path.dirname(texts_filename), exist_ok=True)
            for img in images:
                image = Image.open(img)
                pixel_values = GITprocessor(images=image, return_tensors="pt").pixel_values

                generated_ids = GITmodel.generate(pixel_values=pixel_values, max_length=50)
                new_text_description = GITprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(new_text_description)
                texts.append(new_text_description)
            torch.save(texts, texts_filename)
        # len(texts) = 16540 / 200

        # load eeg data

        # train_data_list[0].shape = torch.Size([10, 4, 63, 100])
        # test_data_list[0].shape = torch.Size([1, 80, 63, 100])

        if self.cross_sub:
            eeg_data = {}
            for subject in self.subjects:
                eeg_data_list, label_list = self.load_eeg_data(subject)
                eeg_data_tensor = torch.cat(eeg_data_list, dim=0).view(-1, *eeg_data_list[0].shape[-2:])
                # print("eeg_data_tensor", eeg_data_tensor.shape)
                eeg_data[subject] = eeg_data_tensor
            print("eeg_data_tensor", eeg_data_tensor[0].shape)
        # when avg = False, torch.Size([66160, 63, 100]) / torch.Size([16000, 63, 100])
        # when avg = True, torch.Size([16540, 63, 100]) / torch.Size([200, 63, 100])

        label_tensor = torch.cat(label_list, dim=0)
        print("label_tensor", label_tensor.shape)
        # when avg = False, torch.Size([66160]) / torch.Size([16000])
        # when avg = True, torch.Size([16540]) / torch.Size([200])

        return eeg_data, label_tensor, category, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Check the window bounding
        indices = (self.times >= start) & (self.times <= end)

        extracted_data = {}
        for key, value in eeg_data.items():
            extracted_data[key] = value[..., indices]

        return extracted_data

    def ClipTextencoder(self, texts):
        batch_size = 16
        clip_text_features_list = []
        # clip_text_hidden_states_list = []

        text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
        for i in range(0, len(texts), batch_size):
            batch_text_inputs = text_inputs[i: i + batch_size]
            with torch.no_grad():
                batch_text_features = vlmodel.encode_text(batch_text_inputs)
                # batch_text_features = batch_text_features.text_embeds

            clip_text_features_list.append(batch_text_features)
            # clip_text_hidden_states_list.append(batch_text_hidden_states)
        # clip_text_features = F.normalize(text_features, dim=-1).detach()
        clip_text_features = torch.cat(clip_text_features_list, dim=0)
        # clip_text_hidden_states = torch.cat(clip_text_hidden_states_list, dim=0)
        if self.norm_embs:
            clip_text_features = F.normalize(clip_text_features, dim=-1).detach()

        return clip_text_features

    # def ClipTextencoder(self, texts):
    #     batch_size = 16
    #     clip_text_features_list = []
    #     text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
    #     for i in range(0, len(texts), batch_size):
    #         batch_text_inputs = text_inputs[i: i + batch_size]
    #         with torch.no_grad():
    #             batch_text_features = clip_text_encoder(batch_text_inputs)
    #             # batch_text_features = batch_text_features.text_embeds
    #             embeds = clip_text_encoder.text_projection(batch_text_features.last_hidden_state)
    #             embeds_pooled = batch_text_features.text_embeds
    #             if self.norm_embs:
    #                 embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1, keepdim=True)
    #
    #         clip_text_features_list.append(embeds)
    #     # clip_text_features = F.normalize(text_features, dim=-1).detach()
    #     clip_text_hidden_states = torch.cat(clip_text_features_list, dim = 0)
    #
    #     return clip_text_hidden_states

    # def ClipTextencoder(self, texts):
    #     text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
    #     with torch.no_grad():
    #         text_features = clip_text_encoder(text_inputs)
    #         text_features = text_features.text_embeds
    #
    #     # clip_text_features = F.normalize(text_features, dim=-1).detach()
    #     clip_text_features = text_features
    #     return clip_text_features

    # def SDTextencoder(self, texts):
    #     batch_size = 16
    #     clip_text_hidden_states_list = []
    #     # TODO : replace clip_text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    #     text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
    #     for i in range(0, len(texts), batch_size):
    #         batch_text_inputs = text_inputs[i: i + batch_size]
    #         with torch.no_grad():
    #             batch_text_hidden_states = text_encoder(batch_text_inputs)
    #             batch_text_hidden_states = batch_text_hidden_states[0]
    #
    #         clip_text_hidden_states_list.append(batch_text_hidden_states)
    #
    #
    #     # clip_text_hidden_states = F.normalize(text_features, dim=-1).detach()
    #     clip_text_hidden_states = torch.cat(clip_text_hidden_states_list, dim = 0)
    #     if self.norm_embs:
    #         SD_clip_text_hidden_states = F.normalize(clip_text_hidden_states, dim=-1).detach()
    #     return SD_clip_text_hidden_states

    def ClipImageEncoder(self, images):
        # Prevent memory overflow on the GPU
        batch_size = 8
        image_features_list = []
        image_hidden_states_list = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([processor(Image.open(img).convert("RGB")) for img in batch_images]).to(accelerator.device).to(dtype=torch.float16)
            # image_inputs = torch.stack([processor(images = Image.open(img).convert("RGB"), return_tensors="pt").pixel_values[0] for img in batch_images]).to(accelerator.device)
            # image_inputs = processor(images=[Image.open(img).convert("RGB") for img in batch_images],
            #                          return_tensors="pt").pixel_values.to(accelerator.device)

            with torch.no_grad():
                # vlmodel.visual.output_tokens = True
                batch_image_features, batch_img_hidden_states = vlmodel.encode_image(image_inputs)
                # batch_image_features = vlmodel.encode_image(image_inputs)
                # batch_image_features = batch_image_features.image_embeds
                # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
            image_hidden_states_list.append(batch_img_hidden_states)

        clip_image_features = torch.cat(image_features_list, dim=0)
        clip_image_hidden_states = torch.cat(image_hidden_states_list, dim =0)
        if self.norm_embs:
            clip_image_features = F.normalize(clip_image_features, dim=-1).detach()

        return clip_image_features, clip_image_hidden_states

    # def ClipImageEncoder(self, images):
    #     # Prevent memory overflow on the GPU
    #     batch_size = 8
    #     image_features_list = []
    #
    #     for i in range(0, len(images), batch_size):
    #         batch_images = images[i:i + batch_size]
    #         # image_inputs = torch.stack([processor(images = Image.open(img).convert("RGB"), return_tensors="pt").pixel_values[0] for img in batch_images]).to(device)
    #         image_inputs = processor(images=[Image.open(img).convert("RGB") for img in batch_images],
    #                                  return_tensors="pt").pixel_values.to(accelerator.device)
    #
    #         with torch.no_grad():
    #             batch_image_features = clip_image_encoder(image_inputs)
    #             batch_image_embeds = batch_image_features.last_hidden_state
    #             # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
    #             # batch_image_features = F.normalize(batch_image_features, dim=-1).detach()
    #             batch_image_embeds = clip_image_encoder.vision_model.post_layernorm(batch_image_embeds)
    #             batch_image_hidden_states = clip_image_encoder.visual_projection(batch_image_embeds)
    #             if self.norm_embs:
    #                 # normalize all tokens by cls token's norm
    #                 batch_image_hidden_states = batch_image_hidden_states / torch.norm(batch_image_hidden_states[:, 0], dim=-1).reshape(-1, 1, 1)
    #
    #         image_features_list.append(batch_image_hidden_states)
    #
    #     clip_img_hidden_states = torch.cat(image_features_list, dim=0)
    #
    #     return clip_img_hidden_states

    def VAEImageEncoder(self, images):
        transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        batch_size = 16
        image_emdeddings_list = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([transform(Image.open(img).convert("RGB")) for img in batch_images]).to(
                accelerator.device).to(dtype=torch.float16)

            with torch.no_grad():
                batch_image_emdedding = vae.encode(
                    image_inputs).latent_dist.mode() * vae.config.scaling_factor  # 0.18215
                # shape [1, 4, 64, 64]

            image_emdeddings_list.append(batch_image_emdedding)

        vae_image_features = torch.cat(image_emdeddings_list, dim=0)
        return vae_image_features

    def IPImgaeEnocder(self, images):
        batch_size = 8
        image_features_list = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            # processor是从ViT-H加载的
            image_inputs = torch.stack([processor(Image.open(img).convert("RGB")) for img in batch_images]).to(accelerator.device).to(dtype=torch.float16)
            # IP_processor是之前从ViT-L加载的
            # image_inputs = IP_processor(images=[Image.open(img).convert("RGB") for img in batch_images], return_tensors="pt").pixel_values.to(accelerator.device)

            with torch.no_grad():
                batch_image_features = IP_image_encoder(image_inputs)
                batch_image_embeds = batch_image_features.image_embeds

            image_features_list.append(batch_image_embeds)

        IP_image_features = torch.cat(image_features_list, dim=0)
        if self.norm_embs:
            IP_image_features = F.normalize(IP_image_features, dim=-1).detach()

        return IP_image_features

    def __getitem__(self, index):
        """Get data and label corresponding to index"""
        eeg_data ={}
        for key, value in self.eeg_data_all.items():
            eeg_data[key] = value[index]
        # eeg_data = self.eeg_data_all[index]
        label = self.labels[index]

        if self.train:
            if self.avg:
                index_n_sub_train = self.n_cls * 10
                img_index = index % index_n_sub_train
                category_index = (index % index_n_sub_train) // (10)
            else:
                index_n_sub_train = self.n_cls * 10 * 4
                img_index = (index % index_n_sub_train) // (4)
                category_index = (index % index_n_sub_train) // (10 * 4)
        else:

            if self.avg:
                index_n_sub_test = self.n_cls * 1
                img_index = index % index_n_sub_test
                category_index = (index % index_n_sub_test) // (1)
            else:
                index_n_sub_test = self.n_cls * 1 * 80
                img_index = (index % index_n_sub_test) // (80)
                category_index = (index % index_n_sub_test) // (1 * 80)

        category = self.category[category_index]
        text = self.texts[img_index]
        img = self.images[img_index]

        # clip_text_features = self.clip_text_features[text_index]
        # clip_text_hidden_states = self.clip_text_hidden_states[text_index]

        # clip_text_hidden_states = self.clip_text_hidden_states[img_index]
        # SD_clip_text_hidden_states = self.SD_clip_text_hidden_states[img_index]
        # clip_img_hidden_states = self.clip_img_hidden_states[img_index]
        clip_text_features = self.clip_text_features[img_index]
        clip_img_features = self.clip_img_features[img_index]
        vae_img_features = self.vae_img_features[img_index]
        IP_img_features = self.IP_img_features[img_index]
        clip_image_hidden_states = self.clip_image_hidden_states[img_index]


        return {
            "eeg_data": eeg_data,
            "label" : label,
            # "clip_text_hidden_states": clip_text_hidden_states,
            # "SD_clip_text_hidden_states": SD_clip_text_hidden_states,
            # "clip_img_hidden_states": clip_img_hidden_states,
            "clip_img_features" : clip_img_features,
            "vae_img_features" : vae_img_features,
            "clip_text_features" : clip_text_features,
            "IP_img_features" : IP_img_features,
            "clip_image_hidden_states" : clip_image_hidden_states,
            "text" :text,
            "img" :img,
            "category" :category
        }

    def __len__(self):
        # len = 16540/200 when avg = True, len = 66160/16000 when avg = False
        for key, value in self.eeg_data_all.items():
            len = value.shape[0]
            break
        return len

if __name__ == "__main__":
    data_path = data_path
    train_dataset = SoloEEGDataset(data_path, subjects=['sub-08'], train=True)
    test_dataset = SoloEEGDataset(data_path, subjects=['sub-08'], train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    i = 80
    batch = test_dataset[i]
    print(batch["text"])
    print(f'clip_text_features shape is: {batch["clip_text_features"].shape}')