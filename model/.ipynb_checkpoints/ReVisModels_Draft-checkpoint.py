import torch
import torch.nn as nn
import torch.fft as fft
from layers.Embed import SubjectEmbedding, DataEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import re


class FreqEncoder(nn.Module):
    """频域用attention 也相当于融合了spaial的信息"""

    def __init__(self, configs):
        super().__init__()
        self.norm = nn.LayerNorm(configs.freq_seq_len)
        #self.freq_seq_len = configs.freq_seq_len
        self.enc_embedding = DataEmbedding(configs.freq_seq_len, configs.d_model, dropout=configs.dropout, joint_train=configs.joint_train, num_subjects=configs.num_subjects)
        # encoder 经过 attention projection(Linear) Conv1d * 2 Norm
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.encode_layers)
            ],
            norm_layer= nn.LayerNorm(configs.d_model)
        )

    def forward(self, x, sub_ids = None):
        batch_size = x.size(0)
        fourier_input = fft.rfft(x, dim=-1)
        magnitudes = fourier_input.abs()
        spectrum = magnitudes[:, :, 1:]  # Spectrum without DC component
        power = spectrum ** 2

        x_enc = power

        x_enc = self.norm(x_enc)
        # embedding
        emb_out = self.enc_embedding(x_enc, None, subject_ids = sub_ids)                                                 # emb_out : [batch_size , (channels + 1) , d_model]
        #encode EncoderLayer中做了类似Residual的处理
        enc_out, attn = self.encoder(emb_out, attn_mask=None)                                                            # enc_out : [batch_size , (channels + 1) , d_model]
        return enc_out, attn


class SpaEncoder(nn.Module):
    """空域用attention"""

    def __init__(self):
        super().__init__()


class TimeEncoder(nn.Module):
    """use patchTST to encode temporal 时域用patchTST 如果用iTransformer的话"""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.norm1 = nn.LayerNorm(self.seq_len)
        self.subject_embedding = SubjectEmbedding(configs.num_subjects, self.seq_len) if configs.num_subjects is not None else None
        #self.enc_embedding = DataEmbedding(self.seq_len, configs.patch_d_model, dropout=configs.dropout,
        #                                   joint_train=configs.joint_train, num_subjects=configs.num_subjects)

        # Patching
        self.patch_len = configs.patch_len
        self.patch_stride = configs.patch_stride
        self.padding_patch = configs.padding_patch
        self.patch_num = int((self.seq_len - self.patch_len)/self.patch_stride + 1)
        if self.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.patch_stride))
            self.patch_num += 1

        # embedding
        self.embedding = nn.Linear(self.patch_len, configs.patch_d_model)

        # Residual dropout
        self.dropout = nn.Dropout(configs.dropout)

        # Positional encoding
        print(f'patch_num is {self.patch_num}')
        self.W_pos = torch.empty((self.patch_num, configs.patch_d_model))
        self.W_pos = nn.init.uniform_(self.W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(self.W_pos, requires_grad=True)

        self.PatchEncoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.patch_d_model, configs.n_heads
                    ),
                    configs.patch_d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.encode_layers)
            ],
            norm_layer= nn.LayerNorm(configs.patch_d_model)
        )
        self.flatten = nn.Flatten(start_dim=-2)
        self.finalProj = nn.Linear((self.patch_num * configs.patch_d_model), configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)

    def forward(self, x, sub_ids = None):
        batch_size = x.size(0)


        # embedding
        # emb_out = self.enc_embedding(x, None, subject_ids=sub_ids)                                                          # emb_out : [batch_size , (channels + 1) , patch_d_model]
        subject_emb = self.subject_embedding(subject_ids = sub_ids)                                                           # subject_emb : [batch_size, 1, seq_len]
        x = torch.cat([subject_emb, x], dim=1)                                                                        # x : [batch_size, (channels + 1), seq_len]

        x = self.norm1(x)  # TODO : 在cat subject_emb前还是后比较好？

        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        z = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)                                              # z: [batch_size , (channels + 1) , patch_num , patch_len]
        emb_out = self.embedding(z)                                                                                          # emb_out: [batch_size , (channels + 1) , patch_num , patch_d_model]
        emb_out = torch.reshape(emb_out, (emb_out.shape[0] * emb_out.shape[1], emb_out.shape[2], emb_out.shape[3]))   # emb_out: [batch_size * (channels + 1) , patch_num , patch_d_model]
        emb_out = self.dropout(emb_out + self.W_pos)                                                                         # emb_out: [batch_size * (channels + 1) , patch_num , patch_d_model]
        #encode
        enc_out, attn = self.PatchEncoder(emb_out)                                                                           # enc_out: [batch_size * (channels + 1) , patch_num , patch_d_model]
        enc_out = torch.reshape(enc_out, (batch_size, -1 ,enc_out.shape[-2],enc_out.shape[-1]))                       # enc_out: [batch_size , (channels + 1) , patch_num , patch_d_model]
        out = self.flatten(enc_out)                                                                                         # enc_out: [batch_size , (channels + 1) , patch_num * patch_d_model]
        out = self.finalProj(out)                                                                                           # enc_out: [batch_size , (channels + 1) , d_model]
        out = self.norm2(out)
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
    def __init__(self, configs, channels = 64, clip_text_dim = 77 , embedding_dim = 512 , proj_clip_dim = 768):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, proj_clip_dim)
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=clip_text_dim, kernel_size=1)
        self.norm = nn.LayerNorm(proj_clip_dim)

    def forward(self, x, **kwargs):
        x = self.proj(x)
        x = self.conv1(x)
        return self.norm(x)

class SubmodalProj(nn.Module):
    def __init__(self, configs, channels = 64, clip_text_dim = 77 , embedding_dim = 512 , proj_clip_dim = 768):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, proj_clip_dim)
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=clip_text_dim, kernel_size=1)
        self.norm = nn.LayerNorm(proj_clip_dim)

    def forward(self, x, **kwargs):
        x = self.proj(x)
        x = self.conv1(x)
        return self.norm(x)

class ReVisEncoder(nn.Module):
    def __init__(self, configs, encoder_type):
        super().__init__()
        self.freqEncoder = FreqEncoder(configs)
        self.timeEncoder = TimeEncoder(configs)
        if encoder_type == 'semantic':
            self.proj = SemanticProj(configs)
        elif encoder_type == 'submodality':
            self.proj = SubmodalProj(configs)

    def forward(self, x, sub_ids):
        batch_size = x.size(0)
        # freqEncoder use attention channels-related
        freq_feature,attn = self.freqEncoder(x, sub_ids)
        # timeEncoder is channels-independent
        time_feature = self.timeEncoder(x, sub_ids)
        features = torch.cat((freq_feature, time_feature), dim=-1) # TODO : cat or add?
        # TODO : 1.15 分别用一个conv
        print(f'features shape is{features.shape}')
        out = self.proj(features)
        # TODO : 1.15 分别加入ip_adapter训练中
        # TODO : 1.15 训一个epoch后放 autodl 上
        # TODO : 1.15 写推理 generation 代码
        return out

    def extract_id_from_string(self, s):
        match = re.search(r'\d+$', s)
        if match:
            return int(match.group())
        return None

# rfft = torch.fft.rfft(x, dim=dim)
# magnitudes = rfft.abs()
# spectrum = magnitudes[:, :, 1:]  # Spectrum without DC component
# power = spectrum ** 2
#
# # Frequency
# freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)
#
# # Amplitude
# amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range
#
# # Offset
# offset = rfft.real[:, :, 0] / self.time_range  # DC component
#
# # phase
# phase = rfft.angle()