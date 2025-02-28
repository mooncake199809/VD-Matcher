import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

from .linear_attention import LinearAttention, FullAttention


class RepeatedModuleList(nn.Module):
    def __init__(self, instances, repeated_times):
        super().__init__()
        assert len(instances) == repeated_times
        self.instances = nn.ModuleList(instances)
        self.repeated_times = repeated_times

    def forward(self, *args, **kwargs):
        r = self._repeated_id
        return self.instances[r](*args, **kwargs)

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(repeated_times={self.repeated_times})'
        return msg


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Position_Encoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feature_dim = dim

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    @staticmethod
    def embed_pos(x, pe):
        return Position_Encoding.embed_rotary(x, pe[..., 0], pe[..., 1])

    def forward(self, feature):
        bsize, npoint, _ = feature.shape
        position = torch.arange(npoint, device=feature.device).unsqueeze(dim=0).repeat(bsize, 1).unsqueeze(dim=-1)
        div_term = torch.exp(torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=feature.device) * (
                -math.log(10000.0) / self.feature_dim)).view(1, 1, -1)
        sinx = torch.sin(position * div_term)  # [B, N, d//2]
        cosx = torch.cos(position * div_term)
        sinx, cosx = map(lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1), [sinx, cosx])
        position_code = torch.stack([cosx, sinx], dim=-1)
        if position_code.requires_grad:
            position_code = position_code.detach()
        return position_code


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        if N == 4800:
            H, W = 60, 80
        elif N == 105 * 105:
            H, W = 105, 105
        else:
            H, W = 5, 5
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SeedEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 ffn_ratio=2,
                 isdw=False,
                 repeat_times=2):
        super(SeedEncoderLayer, self).__init__()

        self.dim = d_model // nhead  # 256/8
        self.nhead = nhead  # 8
        self.repeat_times = repeat_times

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.isdw = isdw

        if isdw:
            self.lpu = RepeatedModuleList([DWConv(d_model) for _ in range(repeat_times)], repeat_times)
            self.merge_conv = RepeatedModuleList([DWConv(d_model) for _ in range(repeat_times)], repeat_times)
            self.mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model * ffn_ratio, bias=False),
                RepeatedModuleList([DWConv(d_model * ffn_ratio) for _ in range(repeat_times)], repeat_times),
                nn.GELU(),
                nn.Linear(d_model * ffn_ratio, d_model, bias=False),
            )
            self.rotary_emb = Position_Encoding(dim=d_model)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model * ffn_ratio, bias=False),
                nn.GELU(),
                nn.Linear(d_model * ffn_ratio, d_model, bias=False),
            )

        self.norm1 = RepeatedModuleList([nn.LayerNorm(d_model) for _ in range(repeat_times)], repeat_times)
        self.norm2 = RepeatedModuleList([nn.LayerNorm(d_model) for _ in range(repeat_times)], repeat_times)

        self.drop_path = DropPath(0.1)
        self.aug_shortcut = nn.Sequential(
            RepeatedModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(repeat_times)], repeat_times),
            RepeatedModuleList([nn.LayerNorm(d_model) for _ in range(repeat_times)], repeat_times))

    def forward(self, x, source, x_mask=None, source_mask=None):
        bs = x.size(0)

        if self.isdw:
            query, key, value = self.lpu(x), self.lpu(source), self.lpu(source)
        else:
            query, key, value = x, source, source  # [B N C] [2 2048 256]

        if self.isdw:
            mixed_query_layer = self.q_proj(query)
            mixed_key_layer = self.k_proj(key)
            que_pe = self.rotary_emb(mixed_query_layer)
            key_pe = self.rotary_emb(mixed_key_layer)
            query = Position_Encoding.embed_pos(mixed_query_layer, que_pe).view(bs, -1, self.nhead, self.dim)
            key = Position_Encoding.embed_pos(mixed_key_layer, key_pe).view(bs, -1, self.nhead, self.dim)
        else:
            query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [B, N, (H, D)]
            key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [B, M, (H, D)]

        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [B, N, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [B, N, C]
        if self.isdw:
            message = self.merge_conv(message)
        message = self.norm1(message)
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.drop_path(self.norm2(message))

        return x + message + self.aug_shortcut(x)


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config, repeat_times=2):
        super(LocalFeatureTransformer, self).__init__()

        self.repeat_times = repeat_times
        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.isall = config['isall']
        self.ffn_ratio = config['ffn_ratio']
        self.layers = nn.ModuleList([copy.deepcopy(SeedEncoderLayer(config['d_model'],
                                                                     config['nhead'],
                                                                     config['attention'],
                                                                     config['ffn_ratio'],
                                                                     self.isall[index],
                                                                     repeat_times=self.repeat_times))
                                     for index in range(len(self.layer_names))])
        
        def set_repeated_times_fn(m):
            m._repeated_times = self.repeat_times
        self.apply(set_repeated_times_fn)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def calcul_assign_matrix_1(self, feat0, feat1, data):
        sim_matrix = torch.einsum("nlc,nsc->nls", feat0, feat1) / 0.1
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data["conf_matrix_1200_1"] = conf_matrix

    def calcul_assign_matrix_2(self, feat0, feat1, data):
        sim_matrix = torch.einsum("nlc,nsc->nls", feat0, feat1) / 0.1
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data["conf_matrix_1200_2"] = conf_matrix

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                for i, t in enumerate(range(self.repeat_times)):
                    def set_repeated_id(m):
                        m._repeated_id = i
                    layer.apply(set_repeated_id)
                    feat0 = layer(feat0, feat0, mask0, mask0)
                    feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                for i, t in enumerate(range(self.repeat_times)):
                    def set_repeated_id(m):
                        m._repeated_id = i
                    layer.apply(set_repeated_id)
                    feat0 = layer(feat0, feat1, mask0, mask1)
                    feat1 = layer(feat1, feat0, mask1, mask0)

        return feat0, feat1


class LocalFeatureTransformer_point(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config, repeat_times=2):
        super(LocalFeatureTransformer_point, self).__init__()

        self.repeat_times = repeat_times
        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.isall = config['isall']
        self.ffn_ratio = config['ffn_ratio']
        self.layers = nn.ModuleList([copy.deepcopy(SeedEncoderLayer(config['d_model'],
                                                                     config['nhead'],
                                                                     config['attention'],
                                                                     config['ffn_ratio'],
                                                                     self.isall[index],
                                                                     repeat_times=self.repeat_times))
                                     for index in range(len(self.layer_names))])
        
        def set_repeated_times_fn(m):
            m._repeated_times = self.repeat_times
        self.apply(set_repeated_times_fn)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0_edge, feat1_edge, feat0, feat1, mask0=None, mask1=None, mask0_edge=None, mask1_edge=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                for i, t in enumerate(range(self.repeat_times)):
                    def set_repeated_id(m):
                        m._repeated_id = i
                    layer.apply(set_repeated_id)
                    feat0_edge = layer(feat0_edge, feat0_edge, mask0_edge, mask0_edge)
                    feat1_edge = layer(feat1_edge, feat1_edge, mask1_edge, mask1_edge)
                    feat0 = layer(feat0, feat0_edge, mask0, mask0_edge)
                    feat1 = layer(feat1, feat1_edge, mask1, mask1_edge)
            if name == 'cross':
                for i, t in enumerate(range(self.repeat_times)):
                    def set_repeated_id(m):
                        m._repeated_id = i
                    layer.apply(set_repeated_id)
                    feat0_edge = layer(feat0_edge, feat1_edge, mask0_edge, mask1_edge)
                    feat1_edge = layer(feat1_edge, feat0_edge, mask1_edge, mask0_edge)
                    feat0 = layer(feat0, feat0_edge, mask0, mask0_edge)
                    feat1 = layer(feat1, feat1_edge, mask1, mask1_edge)
        return feat0, feat1, feat0_edge, feat1_edge
