import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from layers.Embed import SubjectEmbedding, DataEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
import re
from loss import ClipLoss, soft_clip_loss
import numpy as np
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
import os


# enc_out: [batch_size , channels , d_model]
class TimeEncoder(nn.Module):
    """use patchTST to encode temporal 时域用patchTST 如果用iTransformer的话"""

    def __init__(self, configs, encode_layers, dynamic=False):
        super().__init__()
        self.encode_layers = encode_layers
        self.seq_len = configs.seq_len
        self.channels_num = configs.channels_num
        self.norm1 = nn.LayerNorm(self.seq_len)
        self.dynamic = dynamic

        # Patching
        self.patch_len = configs.patch_len
        self.patch_stride = configs.patch_stride
        self.padding_patch = configs.padding_patch
        self.patch_num = int((self.seq_len - self.patch_len)/self.patch_stride + 1)
        if self.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.patch_stride))
            self.patch_num += 1

        # embedding
        # self.embedding = nn.Sequential(
        #     nn.Linear(self.patch_len, configs.patch_d_ff),
        #     nn.Linear(configs.patch_d_ff, configs.patch_d_model),
        # )
        self.embedding = nn.Linear(self.patch_len, configs.patch_d_model)
        # self.channelsEmbedding = nn.ModuleDict({
        #         str(channels_id + 1): nn.Linear(configs.seq_len, configs.seq_len) for channels_id in range(configs.channels_num)
        #     })
        self.channel_embedding = nn.Embedding(self.channels_num, self.seq_len)
        # self.shared_embedding = nn.Parameter(torch.randn(1, self.seq_len))

        # Residual dropout
        self.dropout = nn.Dropout(configs.dropout)

        # Positional encoding
        print(f'patch_num is {self.patch_num}')
        self.W_pos = torch.empty((self.patch_num, configs.patch_d_model))
        self.W_pos = nn.init.uniform_(self.W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(self.W_pos, requires_grad=True)
        # if self.dynamic:
        #     self.gate_network = nn.Sequential(
        #         nn.Linear(configs.seq_len, d_model),
        #         nn.Conv1d(in_channels=queries_num, out_channels=d_model, kernel_size=1),
        #         nn.LayerNorm(d_keys * n_heads))  # 生成 gating 权重
        #     self.gate_activation = nn.Sigmoid()  # 归一化权重

        self.PatchEncoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.patch_d_model, configs.n_heads,  dynamic=self.dynamic,
                    ),
                    configs.patch_d_model,
                    configs.patch_d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(self.encode_layers)
            ],
            norm_layer= nn.LayerNorm(configs.patch_d_model)
        )
        self.flatten = nn.Flatten(start_dim=-2)
        self.Proj = nn.Linear((self.patch_num * configs.patch_d_model), configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)

    def forward(self, x, gate_scores):

        batch_size = x.size(0)
        channels_num = x.shape[1]
        for channel_id in range(channels_num):
            channels_emb = self.channel_embedding(torch.tensor(channel_id, device=x.device))
            x[:, channel_id, :] = x[:, channel_id, :].clone() + channels_emb

        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        z = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)                                              # z: [batch_size , channels , patch_num , patch_len]
        emb = self.embedding(z)                                                                                          # emb_out: [batch_size , channels , patch_num , patch_d_model]
        emb_out = self.dropout(emb + self.W_pos)                                                                         # emb_out: [batch_size , channels , patch_num , patch_d_model]
        # res
        emb_out = emb + emb_out                                                                                          # emb_out: [batch_size , channels , patch_num , patch_d_model]
        emb = torch.reshape(emb_out, (emb_out.shape[0] , emb_out.shape[1] * emb_out.shape[2], emb_out.shape[3]))   # emb_out: [batch_size , channels * patch_num , patch_d_model]
        #encode

        enc_out, attn = self.PatchEncoder(emb, gate_scores=gate_scores)                                                                           # enc_out: [batch_size , channels * patch_num , patch_d_model]

        enc_out = torch.reshape(enc_out, (batch_size, self.channels_num ,-1, enc_out.shape[-1]))                      # enc_out: [batch_size , channels , patch_num , patch_d_model]
        enc = enc_out + emb_out
        out = self.flatten(enc)                                                                                         # enc_out: [batch_size , channels , patch_num * patch_d_model]
        out = self.Proj(out)                                                                                           # enc_out: [batch_size , channels , d_model]
        out = self.norm2(out)

        return out

class SpaEncoder(nn.Module):
    """空域用attention"""

    def __init__(self, configs, encode_layers, dynamic=False):
        super().__init__()

        self.encode_layers = encode_layers
        self.dynamic = dynamic
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads, dynamic=self.dynamic,
                    ),
                    configs.d_model,
                    configs.spac_d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(self.encode_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.norm2 = nn.LayerNorm(configs.d_model)

    def forward(self, x, gate_scores):
        enc_out, attn = self.encoder(x, attn_mask=None, gate_scores=gate_scores)  # enc_out : [batch_size , channels , d_model]
        out = self.norm2(enc_out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class SemanticProj(nn.Module):
    def __init__(self, configs, channels = 63, clip_text_dim = 77 , embedding_dim = 1024 , proj_clip_dim = 1024):
        super().__init__()
        self.embedding_dim = configs.d_model
        self.proj = nn.Linear(embedding_dim, proj_clip_dim)
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=clip_text_dim, kernel_size=1)
        self.norm = nn.LayerNorm(proj_clip_dim)

    def forward(self, x, **kwargs):
        x = self.proj(x)
        x = self.conv1(x)
        return self.norm(x)

class SubmodalProj(nn.Module):
    def __init__(self, configs, channels = 63, submodal_dim = 256, embedding_dim = 1024 , proj_clip_dim = 1280):
        super().__init__()
        self.submodal_dim = configs.submodal_dim
        self.embedding_dim = configs.d_model
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, proj_clip_dim),
            nn.Conv1d(in_channels=channels, out_channels=submodal_dim, kernel_size=1),
            torch.nn.LayerNorm(proj_clip_dim)
        )

    def forward(self, x, **kwargs):
        x = self.proj(x)
        return x

class ResMLP(nn.Module):
    def __init__(self, h, n_blocks, dropout=0.15):
        super().__init__()
        self.n_blocks = n_blocks
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x = residual + x
            residual = x
        return x

class ReVISProj(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x, **kwargs):
        x = self.flatten(x)
        return x

class AdaptiveAttentionPool(nn.Module):
    def __init__(self, seq_len, proj_clip_dim = 1024):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(seq_len))  # 学习到的池化权重
        self.norm = nn.LayerNorm(proj_clip_dim)

    def forward(self, x):
        attn_weights = torch.softmax(self.weights, dim=0)  # 归一化权重
        pooled = torch.einsum('bls,l->bs', x, attn_weights)  # [bsz, 1024]
        pooled = self.norm(pooled)
        return pooled

class ReVisEncoder(nn.Module):
    def __init__(self, configs, encoder_type, subjects, cross_sub = False, n_blocks=4):
        super().__init__()
        #self.freqEncoder = FreqEncoder(configs)
        self.cross_sub = cross_sub
        self.channels_num = configs.channels_num
        self.patch_d_model = configs.patch_d_model
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        if  self.cross_sub and subjects is not None:
            self.value_embedding = nn.ModuleDict({
                subject_id: nn.Linear(configs.seq_len, configs.seq_len) for _, subject_id in enumerate(subjects)
            })
            self.dynamic = True
            self.time_gate_network =  nn.ModuleDict({
                subject_id: nn.Sequential(
                nn.Linear(configs.seq_len, self.patch_d_model),
                nn.Conv1d(in_channels=self.channels_num, out_channels=self.patch_d_model, kernel_size=1),
                )  for _, subject_id in enumerate(subjects)
            })
            # self.time_gate_activation = nn.ModuleDict({
            #     subject_id: nn.Softmax(dim=0) for _, subject_id in enumerate(subjects)
            # })
            # self.spac_gate_network = nn.ModuleDict({
            #     subject_id: nn.Sequential(
            #         nn.Linear(configs.seq_len, self.d_model),
            #         nn.Conv1d(in_channels=self.channels_num, out_channels=self.d_model, kernel_size=1),
            #     ) for _, subject_id in enumerate(subjects)
            # })
            # self.spac_gate_activation = nn.ModuleDict({
            #     subject_id: nn.Softmax(dim=0) for _, subject_id in enumerate(subjects)
            # })

        else:
            self.value_embedding = nn.Linear(configs.seq_len, configs.seq_len)
            #TODO:False
            self.dynamic = False
            self.time_gate_network = nn.Sequential(
                nn.Linear(configs.seq_len, self.patch_d_model),
                nn.Conv1d(in_channels=self.channels_num, out_channels=self.patch_d_model, kernel_size=1),)
                # # 生成 gating 权重
                # nn.LayerNorm(self.patch_d_model),
                # # 归一化权重
                # nn.Sigmoid())
            # self.time_gate_activation = nn.Softmax(dim=1)
            # self.spac_gate_network = nn.Sequential(
            #     nn.Linear(configs.seq_len, self.d_model),
            #     nn.Conv1d(in_channels=self.channels_num, out_channels=self.d_model, kernel_size=1),)
            #     # # 生成 gating 权重
            #     # nn.LayerNorm(self.patch_d_model),
            #     # # 归一化权重
            #     # nn.Sigmoid())
            # self.spac_gate_activation = nn.Softmax(dim=1)


        if encoder_type == 'semantic':
            # self.proj = SemanticProj(configs)
            self.encode_layers = configs.encode_layers
        elif encoder_type == 'submodality':
            # self.proj = SubmodalProj(configs)
            self.encode_layers = configs.submodal_encode_layers
        self.SubmodalProj = SubmodalProj(configs)
        # self.SemanticProj = SemanticProj(configs)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.spacRncoder = SpaEncoder(configs)
        self.timeEncoder = TimeEncoder(configs, self.encode_layers, dynamic=self.dynamic)
        # self.enc_embedding = nn.Linear(configs.time_d_model, configs.spac_d_model)

        # self.spacRncoder = ResidualAdd(SpaEncoder(configs, self.encode_layers, dynamic=self.dynamic))
        self.translator = ResMLP(configs.d_model, n_blocks)
        self.imagePool = AdaptiveAttentionPool(256)
        self.fc = nn.Linear(1280, 1024)
        # self.textPool = AdaptiveAttentionPool(77)
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.loss_func = soft_clip_loss()
        # self.loss_func = soft_clip_loss()

    def forward(self, x, subject_id):
        with torch.autograd.set_detect_anomaly(True):
            if self.cross_sub:
        #batch_size = x.size(0)
        # freqEncoder use attention channels-related
        #freq_feature,attn = self.freqEncoder(x, sub_ids)

            # 使用针对每个subject的特定value embedding
                x = self.value_embedding[subject_id](x)
                time_gate_scores = self.time_gate_network[subject_id](x.clone())
                time_gate_scores = time_gate_scores.reshape(-1, self.n_heads, (self.patch_d_model // self.n_heads), self.patch_d_model)
                time_gate_scores = F.gelu(time_gate_scores)
                # time_gate_scores = self.time_gate_activation[subject_id](time_gate_scores)
                # spac_gate_scores = self.spac_gate_network[subject_id](x.clone())
                # spac_gate_scores = self.spac_gate_network[subject_id](x.clone())
                # spac_gate_scores = spac_gate_scores.reshape(-1, self.n_heads, (self.d_model // self.n_heads),
                #                                         self.d_model)
                # spac_gate_scores = self.spac_gate_activation[subject_id](spac_gate_scores)

            else:
                x = self.value_embedding(x)
                time_gate_scores = self.time_gate_network(x.clone())
                time_gate_scores = time_gate_scores.reshape(-1, self.n_heads, (self.patch_d_model // self.n_heads), self.patch_d_model).squeeze(0)
                time_gate_scores = F.gelu(time_gate_scores)
                # spac_gate_scores = self.spac_gate_network(x.clone())
                # spac_gate_scores = self.spac_gate_network(x.clone())
                # spac_gate_scores = spac_gate_scores.reshape(-1, self.n_heads, (self.d_model // self.n_heads),
                #                                             self.d_model).squeeze(0)
                # spac_gate_scores = self.spac_gate_activation(spac_gate_scores)

            # timeEncoder is channels-independent
            time_feature = self.timeEncoder(x, gate_scores=time_gate_scores.contiguous())
            # mid = self.enc_embedding(time_feature)

            # spac_time_feature = self.spacRncoder(time_feature, gate_scores=spac_gate_scores.contiguous()) #[bsz, channels, d_model=256]

            # timeEncoder is channels-independent
            # time_feature = self.timeEncoder(x)
            # features = torch.cat((spac_feature, time_feature), dim=-1) # TODO : cat or add?
            # TODO : 1.15 分别用一个conv
            spac_time_feature = self.translator(time_feature)
            # text_feature = self.SemanticProj(spac_time_feature)
            image_hidden_state = self.SubmodalProj(spac_time_feature)
            # print(f'text_feature shape is {text_feature}')
            # print(f'image_feature shape is {image_feature}')
            # text_out = self.textPool(text_feature)
            image_hidden_state_proj = self.fc(image_hidden_state)
            image_out= self.imagePool(image_hidden_state_proj)
            # TODO : 1.15 分别加入ip_adapter训练中
            # TODO : 1.15 训一个epoch后放 autodl 上
            # TODO : 1.15 写推理 generation 代码
            return image_hidden_state, image_out #, text_out


class teModel(nn.Module):
    def __init__(self, configs, embedding_dim = 100, proj_clip_dim = 768,  channels = 63, clip_text_dim = 77):
        super().__init__()
        self.proj = nn.Linear(embedding_dim , proj_clip_dim)
        self.conv1 = nn.Conv1d(in_channels= channels, out_channels= clip_text_dim, kernel_size=1)
        self.norm = nn.LayerNorm(proj_clip_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
    def forward(self, x, subject_id):
        x = self.proj(x)
        out = self.conv1(x)
        return self.norm(out)

