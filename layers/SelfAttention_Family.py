import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.Mask import TriangularCausalMask
import torch.nn.functional as F

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,  d_keys=None,
                 d_values=None, dynamic =True ):
        super(AttentionLayer, self).__init__()
        self.dynamic = dynamic
        # self.queries_num = queries_num

        self.d_keys = d_keys or (d_model // n_heads)
        self.d_values = d_values or (d_model // n_heads)
        self.d_model = d_model
        self.n_heads = n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(self.d_model, self.d_keys * self.n_heads)
        self.key_projection = nn.Linear(self.d_model, self.d_keys * self.n_heads)
        self.value_projection = nn.Linear(self.d_model, self.d_values * self.n_heads)
        self.out_projection = nn.Linear(self.d_values * self.n_heads, self.d_model)


        # if self.dynamic:
        #     self.query_gate_network = nn.Linear(self.queries_num, self.d_keys * self.n_heads)
        #         # nn.Sequential(
        #         # nn.Linear(d_model, n_heads),
        #         # nn.Conv1d(in_channels=channel_nums, out_channels=d_model, kernel_size=1))  # 生成 gating 权重
        #     self.query_gate_activation = nn.Softmax(dim=0)  # 归一化权重
        #     self.key_gate_network = nn.Linear(self.queries_num, self.d_keys * self.n_heads)
        #     self.key_gate_activation = nn.Softmax(dim=0)  # 归一化权重
        #     self.value_gate_network = nn.Linear(self.queries_num, self.d_keys * self.n_heads)
        #     self.value_gate_activation = nn.Softmax(dim=0)  # 归一化权重

    def forward(self, queries, keys, values,  gate_scores = None, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if self.dynamic:
            # gate_scores = self.gate_network(queries)  # (B, L, d_keys)
            # gate_weights = self.gate_activation(gate_scores)  # (B, L, d_keys)
            #
            # W_q = self.query_projection.weight.view(H, self.d_keys, self.d_model)  # (H, d_keys, d_model)
            # W_k = self.key_projection.weight.view(H, self.d_keys, self.d_model)  # (H, d_keys, d_model)
            # W_v = self.value_projection.weight.view(H, self.d_values, self.d_model)  # (H, d_values, d_model)
            #
            # bias_q = self.query_projection.bias.view(H, -1) if self.query_projection.bias is not None else None
            # bias_k = self.key_projection.bias.view(H, -1) if self.key_projection.bias is not None else None
            # bias_v = self.value_projection.bias.view(H, -1) if self.value_projection.bias is not None else None
            # # print(f'gate_weights shape is : {gate_weights.unsqueeze(-1).shape}')
            # # print(f'W_q shape is : {W_q.shape}')
            #
            # W_q = torch.einsum('bhk,hkm->bhkm', gate_weights, W_q).sum(dim=1).view(-1, self.d_model)  # (B, L, d_keys, d_model)
            # W_k = torch.einsum('bhk,hkm->bhkm', gate_weights, W_k).sum(dim=1).view(-1, self.d_model)  # (B, L, d_keys, d_model)
            # W_v = torch.einsum('bhk,hkm->bhkm', gate_weights, W_v).sum(dim=1).view(-1, self.d_model)  # (B, L, d_values, d_model)
            #
            # if bias_q is not None:
            #     bias_q = (gate_weights @ bias_q).view(B, L, -1)  # (B, L, d_keys)
            # if bias_k is not None:
            #     bias_k = (gate_weights @ bias_k).view(B, L, -1)  # (B, L, d_keys)
            # if bias_v is not None:
            #     bias_v = (gate_weights @ bias_v).view(B, L, -1)  # (B, L, d_values)
            #
            # # **Step 4: 计算投影后的 Query/Key/Value**
            # queries = F.linear(queries, W_q) + (bias_q if bias_q is not None else 0)
            # Q = queries.view(B, L, H, -1)  # (B, L, H, d_keys)
            #
            # keys = F.linear(keys, W_k) + (bias_k if bias_k is not None else 0)
            # K = keys.view(B, S, H, -1)  # (B, S, H, d_keys)
            #
            # values = F.linear(values, W_v) + (bias_v if bias_v is not None else 0)
            # V = values.view(B, S, H, -1)  # (B, S, H, d_values)


                # query_gate_weights = self.query_gate_network(query.t().float())
                # query_gate_weights = query_gate_weights.t().reshape((H, self.d_keys, self.d_model))
                # query_gate_weights = self.query_gate_activation(query_gate_weights)
            for i in range(queries.shape[0]):
                query = queries[i]
                bias_q = self.query_projection.bias
                W_q = self.query_projection.weight.view(H, self.d_keys, self.d_model)
                W_q = torch.einsum('hkm,hkm->hkm', gate_scores[i], W_q).view(-1, self.d_model)
                query = F.linear(query.clone(), W_q, (bias_q if bias_q is not None else 0))
                queries[i] = query.contiguous()

                # key_gate_weights = self.key_gate_network(key.t().float())
                # key_gate_weights = key_gate_weights.t().reshape((H, self.d_keys, self.d_model))
                # key_gate_weights = self.key_gate_activation(key_gate_weights)
                key = keys[i]
                bias_k = self.key_projection.bias
                W_k = self.key_projection.weight.view(H, self.d_keys, self.d_model)
                W_k = torch.einsum('hkm,hkm->hkm', gate_scores[i], W_k).view(-1, self.d_model)
                key = F.linear(key.clone(), W_k, (bias_k if bias_k is not None else 0))
                keys[i] = key.contiguous()

                # value_gate_weights = self.value_gate_network(value.t().float())
                # value_gate_weights = value_gate_weights.t().reshape((H, self.d_keys, self.d_model))
                # value_gate_weights = self.value_gate_activation(value_gate_weights)
                value = values[i]
                bias_v = self.value_projection.bias
                W_v = self.value_projection.weight.view(H, self.d_keys, self.d_model)
                W_v = torch.einsum('hkm,hkm->hkm', gate_scores[i], W_v).view(-1, self.d_model)
                value = F.linear(value.clone(), W_v, (bias_v if bias_v is not None else 0))
                values[i] = value.contiguous()
            queries = queries.view(B, L, H, -1).contiguous()
            keys = keys.view(B, S, H, -1).contiguous()
            values = values.view(B, S, H, -1).contiguous()

                # W_q = self.query_projection.weight
            # W_k = self.key_projection.weight
            # W_v = self.value_projection.weight
            # bias_q = self.query_projection.bias if self.query_projection.bias is not None else None
            # bias_k = self.key_projection.bias if self.key_projection.bias is not None else None
            # bias_v = self.value_projection.bias if self.value_projection.bias is not None else None
            # gate_weights = gate_scores.clone().mean(dim=0).contiguous().transpose(0, 1)
            # W_q = gate_weights * W_q
            # W_k = gate_weights * W_k
            # W_v = gate_weights * W_v
            # queries = F.linear(queries, W_q) + (bias_q if bias_q is not None else 0)
            # Q = queries.view(B, L, H, -1)  # (B, L, H, d_keys)
            # # queries = self.query_projection(queries).reshape(B, L, H, -1)
            #
            # keys = F.linear(keys, W_k) + (bias_k if bias_k is not None else 0)
            # K = keys.view(B, S, H, -1)  # (B, S, H, d_keys)
            # # keys = self.key_projection(keys).contiguous().reshape(B, S, H, -1)
            #
            # values = F.linear(values, W_v) + (bias_v if bias_v is not None else 0)
            # V = values.view(B, S, H, -1)  # (B, S, H, d_values)

        else:
            queries = self.query_projection(queries).view(B, L, H, -1)
            keys = self.key_projection(keys).view(B, S, H, -1)
            values = self.value_projection(values.contiguous()).view(B, S, H, -1)


        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out.contiguous()), attn

# class CrossAttention(nn.Module):
#     # input from VAE embedding []