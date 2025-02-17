import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import clip
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from accelerate import Accelerator


import os
import open_clip
proxy = 'http://127.0.0.1:7897'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
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

# model_type = 'ViT-H-14'
# vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
#     model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device = device)
# vlmodel.requires_grad_(False)

# model_type = 'ViT-L/14'
model_type = 'ViT-L-14'
clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
clip_image_encoder.requires_grad_(False)
clip_image_encoder = clip_image_encoder.to(accelerator.device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 加载与 image_encoder 对应的 text_encoder
clip_text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
clip_text_encoder.requires_grad_(False)
clip_text_encoder = clip_text_encoder.to(accelerator.device)

from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
vae.requires_grad_(False)
vae.to(accelerator.device)


class SoloEEGDataset(Dataset):
    """
    load one subject's data
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """

    def __init__(self, data_path, subjects=None, train=True, time_window=[0, 1.0], avg=True):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.avg = avg

        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)

        self.eeg_data_all, self.labels, self.texts, self.images = self.load_data()

        self.eeg_data_all = self.extract_eeg(self.eeg_data_all, time_window)

        features_filename = os.path.join(f'{model_type}_features_train.pt') if self.train else os.path.join(
            f'{model_type}_features_test.pt')
        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename)
            self.clip_text_features = saved_features['clip_text_features']
            self.clip_img_features = saved_features['clip_img_features']
            self.vae_img_features = saved_features['vae_img_features']
        else:
            self.clip_text_features = self.ClipTextencoder(self.texts)
            self.clip_img_features = self.ClipImageEncoder(self.images)
            self.vae_img_features = self.VAEImageEncoder(self.images)
            torch.save({
                'clip_text_features': self.clip_text_features.cpu(),
                'clip_img_features': self.clip_img_features.cpu(),
                'vae_img_features': self.vae_img_features.cpu(),
            }, features_filename)

    def load_eeg_data(self, subject):
        if self.train:
            file_name = 'preprocessed_eeg_training.npy'
            n_classes = 1654
            samples_per_class = 10
            repeat_times = 4
        else:
            file_name = 'preprocessed_eeg_test.npy'
            n_classes = 200
            samples_per_class = 1
            repeat_times = 80

        file_path = os.path.join(self.data_path, subject, file_name)
        data = np.load(file_path, allow_pickle=True)

        preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
        times = torch.from_numpy(data['times']).detach()[20:]
        ch_names = data['ch_names']
        self.times = times
        self.ch_names = ch_names

        eeg_data_list = []
        label_list = []
        for i in range(n_classes):
            start_index = i * samples_per_class
            eeg_data = preprocessed_eeg_data[start_index: start_index + samples_per_class]
            if self.avg:
                labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
            else:
                labels = torch.full((samples_per_class * repeat_times,), i, dtype=torch.long).detach()
            if self.avg:
                eeg_data = torch.mean(eeg_data, 1)
            eeg_data_list.append(eeg_data)
            label_list.append(labels)
        return eeg_data_list, label_list

    def load_data(self):
        texts = []
        images = []

        if self.train:
            directory = img_directory_training
        else:
            directory = img_directory_test

        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()

        # load text data
        text_folders = dirnames
        for dir in text_folders:
            try:
                idx = dir.index('_')
                text_description = dir[idx + 1:]
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")
                continue

            new_text_description = f"This picture is {text_description}"
            texts.append(new_text_description)
        # len(texts) = 1654 / 200

        # load image data
        image_folders = dirnames
        for folder in image_folders:
            floder_path = os.path.join(directory, folder)
            all_images = [img for img in os.listdir(floder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            images.extend(os.path.join(floder_path, img) for img in all_images)
        # len(images) = 16540 / 200

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

        return eeg_data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Check the window bounding
        indices = (self.times >= start) & (self.times <= end)
        extracted_data = eeg_data[..., indices]

        return extracted_data

    def ClipTextencoder(self, texts):
        # TODO : replace clip_text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(accelerator.device)
        with torch.no_grad():
            text_features = clip_text_encoder(text_inputs)
            text_features = text_features.text_embeds

        clip_text_features = F.normalize(text_features, dim=-1).detach()
        return clip_text_features

    def ClipImageEncoder(self, images):
        # Prevent memory overflow on the GPU
        batch_size = 2
        image_features_list = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            # image_inputs = torch.stack([processor(images = Image.open(img).convert("RGB"), return_tensors="pt").pixel_values[0] for img in batch_images]).to(device)
            image_inputs = processor(images=[Image.open(img).convert("RGB") for img in batch_images],
                                     return_tensors="pt").pixel_values.to(accelerator.device)

            with torch.no_grad():
                batch_image_features = clip_image_encoder(image_inputs)
                batch_image_features = batch_image_features.image_embeds
                # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
                batch_image_features = F.normalize(batch_image_features, dim=-1).detach()

            image_features_list.append(batch_image_features)

        clip_image_features = torch.cat(image_features_list, dim=0)

        return clip_image_features

    def VAEImageEncoder(self, images):
        transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        batch_size = 20
        image_emdeddings_list = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([transform(Image.open(img).convert("RGB")) for img in batch_images]).to(accelerator.device)

            with torch.no_grad():
                batch_image_emdedding = vae.encode(image_inputs).latent_dist.mode() * 0.18215

            image_emdeddings_list.append(batch_image_emdedding)

        vae_image_features = torch.cat(image_emdeddings_list, dim=0)
        return vae_image_features

    def __getitem__(self, index):
        """Get data and label corresponding to index"""
        eeg_data = self.eeg_data_all[index]
        label = self.labels[index]

        if self.train:
            if self.avg:
                index_n_sub_train = self.n_cls * 10
                img_index = index % index_n_sub_train
                text_index = (index % index_n_sub_train) // (10)
            else:
                index_n_sub_train = self.n_cls * 10 * 4
                img_index = (index % index_n_sub_train) // (4)
                text_index = (index % index_n_sub_train) // (10 * 4)
        else:

            if self.avg:
                index_n_sub_test = self.n_cls * 1
                img_index = index % index_n_sub_test
                text_index = (index % index_n_sub_test) // (1)
            else:
                index_n_sub_test = self.n_cls * 1 * 80
                img_index = (index % index_n_sub_test) // (80)
                text_index = (index % index_n_sub_test) // (1 * 80)

        text = self.texts[text_index]
        img = self.images[img_index]

        clip_text_features = self.clip_text_features[text_index]
        clip_img_features = self.clip_img_features[img_index]
        vae_img_features = self.vae_img_features[img_index]

        return {
            "eeg_data": eeg_data, 
            "label" : label, 
            "clip_text_features": clip_text_features, 
            "clip_img_features" : clip_img_features, 
            "vae_img_features" : vae_img_features, 
            "text" :text, 
            "img" :img
        }

    def __len__(self):
        # len = 16540/200 when avg = True, len = 66160/16000 when avg = False
        return self.eeg_data_all.shape[0]

if __name__ == "__main__":
    data_path = data_path
    train_dataset = SoloEEGDataset(data_path, subjects=['sub-01'], train=True)
    test_dataset = SoloEEGDataset(data_path, subjects=['sub-01'], train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    i = 80
    eeg_data, label, clip_text_features, clip_img_features, vae_img_features, text, img = test_dataset[i]