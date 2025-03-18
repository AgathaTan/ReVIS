import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

class SubjectEmbedding(nn.Module):
    def __init__(self, num_subjects, d_model):
        super(SubjectEmbedding, self).__init__()
        self.subject_embedding = nn.Embedding(num_subjects, d_model)
        self.shared_embedding = nn.Parameter(torch.randn(1, d_model))  # Shared token for unknown subjects
        self.mask_embedding = nn.Parameter(torch.randn(1, d_model))  # Mask token embedding

    def forward(self, subject_ids):

        if subject_ids[0] is None or torch.any(subject_ids >= self.subject_embedding.num_embeddings):
            batch_size = subject_ids.size(0)
            return self.shared_embedding.expand(batch_size, 1, -1)
        else:
            return self.subject_embedding(subject_ids).unsqueeze(1)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, joint_train=False, num_subjects=None):
        super(DataEmbedding, self).__init__()
        if joint_train and num_subjects is not None:
            self.value_embedding = nn.ModuleDict({
                str(subject_id): nn.Linear(c_in, d_model) for subject_id in range(num_subjects)
            })
        else:
            self.value_embedding = nn.Linear(c_in, d_model)  # 如果没有指定subjects，则使用单一的value embedding

        # self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding \
        #     (d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.subject_embedding = SubjectEmbedding(num_subjects, d_model) if num_subjects is not None else None
        self.mask_token = nn.Parameter(torch.randn(1, d_model))  # Mask token embedding
        self.joint_train = joint_train

    def forward(self, x, x_mark, subject_ids=None, mask=None):
        if self.joint_train:
            # 使用针对每个subject的特定value embedding
            x = torch.stack \
                ([self.value_embedding[str(subject_id.item())](x[i]) for i, subject_id in enumerate(subject_ids)])
        else:
            x = self.value_embedding(x)

        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark) + self.position_embedding(x)

        if mask is not None:
            x = x * (~mask.bool()) + self.mask_token * mask.float()

        if self.subject_embedding is not None:
            subject_emb = self.subject_embedding(subject_ids)  # (batch_size, 1, d_model)
            x = torch.cat([subject_emb, x], dim=1)  # 在序列维度上拼接 (batch_size, channels + 1, d_model)

        return self.dropout(x)