# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
warnings.filterwarnings("ignore")
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from typing import Sequence, Optional, Callable
from mmcv.cnn import Conv2d
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import BaseModule, ModuleList, Sequential

from ..builder import BACKBONES
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

class DepthWiseConvModule(BaseModule):
    """An implementation of one Depth-wise Conv Module of LEFormer.

    Args:
        embed_dims (int): The feature dimension.
        feedforward_channels (int): The hidden dimension for FFNs.
        output_channels (int): The output channles of each cnn encoder layer.
        kernel_size (int): The kernel size of Conv2d. Default: 3.
        stride (int): The stride of Conv2d. Default: 2.
        padding (int): The padding of Conv2d. Default: 1.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default: 0.0.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(DepthWiseConvModule, self).__init__(init_cfg)
        self.activate = build_activation_layer(act_cfg)
        fc1 = Conv2d(
            in_channels=embed_dims,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MixFFN(BaseModule):
    """An implementation of MixFFN of LEFormer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class Pooling(nn.Module):
    """Pooling module.

    Args:
        pool_size (int): Pooling size. Defaults: 3.
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """Mlp implemented by with 1*1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolingBlock(BaseModule):
    """Pooling Block.

    Args:
        embed_dims (int): The feature dimension.
        pool_size (int): Pooling size. Defaults to 3.
        mlp_ratio (float): Mlp expansion ratio. Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='GN', num_groups=1)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-5.
    """

    def __init__(self,
                 embed_dims,
                 pool_size=3,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 drop_path=0.,
                 layer_scale_init_value=1e-5):
        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dims,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((embed_dims)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((embed_dims)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        return x


class EfficientMultiheadAttention(MultiheadAttention):
    """An implementation of Efficient Multi-head Attention of LEFormer.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of LEFormer. Default: 1.
        pool_size (int): Pooling size. Default: 3.
        pool (bool): Whether to use Pooling Transformer Layer. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 pool_size=3,
                 pool=False
                 ):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.pool = pool
        if not self.pool:
            self.sr_ratio = sr_ratio

            if sr_ratio > 1:
                self.sr = Conv2d(
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    kernel_size=sr_ratio,
                    stride=sr_ratio)
                # The ret[0] of build_norm_layer is norm name.
                self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.pool_former_block = PoolingBlock(
                embed_dims=embed_dims,
                pool_size=pool_size
            )

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmseg import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'EfficientMultiheadAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):
        if self.pool:
            out = nlc_to_nchw(x, hw_shape)
            out = self.pool_former_block(out)
            out = nchw_to_nlc(out)
            return out
        else:
            x_q = x
            if identity is None:
                identity = x_q
            if self.sr_ratio > 1:
                x_kv = nlc_to_nchw(x, hw_shape)
                x_kv = self.sr(x_kv)
                x_kv = nchw_to_nlc(x_kv)
                x_kv = self.norm(x_kv)
            else:
                x_kv = x

            # Because the dataflow('key', 'query', 'value') of
            # ``torch.nn.MultiheadAttention`` is (num_query, batch,
            # embed_dims), We should adjust the shape of dataflow from
            # batch_first (batch, num_query, embed_dims) to num_query_first
            # (num_query ,batch, embed_dims), and recover ``attn_output``
            # from num_query_first to batch_first.
            if self.batch_first:
                x_q = x_q.transpose(0, 1)
                x_kv = x_kv.transpose(0, 1)

            out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

            if self.batch_first:
                out = out.transpose(0, 1)

            return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        """multi head attention forward in mmcv version < 1.3.17."""

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # `need_weights=True` will let nn.MultiHeadAttention
        # `return attn_output, attn_output_weights.sum(dim=1) / num_heads`
        # The `attn_output_weights.sum(dim=1)` may cause cuda error. So, we set
        # `need_weights=False` to ignore `attn_output_weights.sum(dim=1)`.
        # This issue - `https://github.com/pytorch/pytorch/issues/37583` report
        # the error that large scale tensor sum operation may cause cuda error.
        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


# class TransformerEncoderLayer(BaseModule):
#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  feedforward_channels,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  qkv_bias=True,
#                  act_cfg=dict(type='GELU'),
#                  norm_cfg=dict(type='LN'),
#                  batch_first=True,
#                  sr_ratio=1,
#                  with_cp=False,
#                  pool_size=3,
#                  pool=False
#                  ):
#         super(TransformerEncoderLayer, self).__init__()

#         # The ret[0] of build_norm_layer is norm name.
#         self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

#         self.attn = EfficientMultiheadAttention(
#             embed_dims=embed_dims,
#             num_heads=num_heads,
#             attn_drop=attn_drop_rate,
#             proj_drop=drop_rate,
#             dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
#             batch_first=batch_first,
#             qkv_bias=qkv_bias,
#             norm_cfg=norm_cfg,
#             sr_ratio=sr_ratio,
#             pool_size=pool_size,
#             pool=pool
#         )

#         # The ret[0] of build_norm_layer is norm name.
#         self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
#         self.pool = pool
#         if not self.pool:
#             self.ffn = MixFFN(
#                 embed_dims=embed_dims,
#                 feedforward_channels=feedforward_channels,
#                 ffn_drop=drop_rate,
#                 dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
#                 act_cfg=act_cfg)

#         self.with_cp = with_cp

#     def forward(self, x, hw_shape):

#         def _inner_forward(x):
#             x = self.attn(self.norm1(x), hw_shape, identity=x)
#             if not self.pool:
#                 x = self.ffn(self.norm2(x), hw_shape, identity=x)
#             return x

#         if self.with_cp and x.requires_grad:
#             x = cp.checkpoint(_inner_forward, x)
#         else:
#             x = _inner_forward(x)
#         return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()
        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0
        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)
        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)
    
class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """
    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))
    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h, w = self.input_resolution

        # 动态调整 h 和 w，确保 h * w == n
        if h * w != n:
            h = int(n ** 0.5)
            w = n // h
            if h * w != n:
                raise ValueError(f"Cannot reshape input of length {n} into resolution ({h}, {w}).")

        num_heads = self.num_heads
        head_dim = c // num_heads
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # 确保 reshape 操作有效
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z
        x = x.transpose(1, 2).reshape(b, n, c)

        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
    
    
class MLLABlock(nn.Module):
    r""" MLLA Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)
        # Linear Attention
        x = self.attn(x)
        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"
    



class ChannelAttentionModule(BaseModule):
    """An implementation of one Channel Attention Module of LEFormer.

        Args:
            embed_dims (int): The embedding dimension.
    """

    def __init__(self, embed_dims):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            Conv2d(embed_dims, embed_dims // 4, 1, bias=False),
            nn.ReLU(),
            Conv2d(embed_dims // 4, embed_dims, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(BaseModule):
    """An implementation of one Spatial Attention Module of LEFormer.

        Args:
            kernel_size (int): The kernel size of Conv2d. Default: 3.
    """

    def __init__(self, kernel_size=3):
        super(SpatialAttentionModule, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MultiscaleCBAMLayer(BaseModule):
    """An implementation of Multiscale CBAM layer of LEFormer.

        Args:
            embed_dims (int): The feature dimension.
            kernel_size (int): The kernel size of Conv2d. Default: 7.
        """

    def __init__(self, embed_dims, kernel_size=7):
        super(MultiscaleCBAMLayer, self).__init__()
        self.channel_attention = ChannelAttentionModule(embed_dims // 4)
        self.spatial_attention = SpatialAttentionModule(kernel_size)
        self.multiscale_conv = nn.ModuleList()
        for i in range(1, 5):
            self.multiscale_conv.append(
                Conv2d(
                    in_channels=embed_dims // 4,
                    out_channels=embed_dims // 4,
                    kernel_size=3,
                    stride=1,
                    padding=(2 * i + 1) // 2,
                    bias=True,
                    dilation=(2 * i + 1) // 2)
            )

    def forward(self, x):
        outs = torch.split(x, x.shape[1] // 4, dim=1)
        out_list = []
        for (i, out) in enumerate(outs):
            out = self.multiscale_conv[i](out)
            out = self.channel_attention(out) * out
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        out = self.spatial_attention(out) * out
        return out


# class CnnEncoderLayer(BaseModule):
#     """Implements one cnn encoder layer in LEFormer.

#         Args:
#             embed_dims (int): The feature dimension.
#             feedforward_channels (int): The hidden dimension for FFNs.
#             output_channels (int): The output channles of each cnn encoder layer.
#             kernel_size (int): The kernel size of Conv2d. Default: 3.
#             stride (int): The stride of Conv2d. Default: 2.
#             padding (int): The padding of Conv2d. Default: 0.
#             act_cfg (dict): The activation config for FFNs.
#                 Default: dict(type='GELU').
#             ffn_drop (float, optional): Probability of an element to be
#                 zeroed in FFN. Default 0.0.
#             init_cfg (dict, optional): Initialization config dict.
#                 Default: None.
#         """

#     def __init__(self,
#                  embed_dims,
#                  feedforward_channels,
#                  output_channels,
#                  kernel_size=3,
#                  stride=2,
#                  padding=0,
#                  act_cfg=dict(type='GELU'),
#                  ffn_drop=0.,
#                  init_cfg=None):
#         super(CnnEncoderLayer, self).__init__(init_cfg)

#         self.embed_dims = embed_dims
#         self.feedforward_channels = feedforward_channels
#         self.output_channels = output_channels
#         self.act_cfg = act_cfg
#         self.activate = build_activation_layer(act_cfg)

#         self.layers = DepthWiseConvModule(embed_dims=embed_dims,
#                                           feedforward_channels=feedforward_channels // 2,
#                                           output_channels=output_channels,
#                                           kernel_size=kernel_size,
#                                           stride=stride,
#                                           padding=padding,
#                                           act_cfg=dict(type='GELU'),
#                                           ffn_drop=ffn_drop)

#         self.multiscale_cbam = MultiscaleCBAMLayer(output_channels, kernel_size)

#     def forward(self, x):
#         out = self.layers(x)
#         out = self.multiscale_cbam(out)
#         return out

class CnnEncoderLayer(BaseModule):
    """Implements one cnn encoder layer in LEFormer.

        Args:
            embed_dims (int): The feature dimension.
            feedforward_channels (int): The hidden dimension for FFNs.
            output_channels (int): The output channles of each cnn encoder layer.
            kernel_size (int): The kernel size of Conv2d. Default: 3.
            stride (int): The stride of Conv2d. Default: 2.
            padding (int): The padding of Conv2d. Default: 0.
            act_cfg (dict): The activation config for FFNs.
                Default: dict(type='GELU').
            ffn_drop (float, optional): Probability of an element to be
                zeroed in FFN. Default 0.0.
            init_cfg (dict, optional): Initialization config dict.
                Default: None.
        """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=2,
                 padding=0,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(CnnEncoderLayer, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.output_channels = output_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        self.layers = DepthWiseConvModule(embed_dims=embed_dims,
                                          feedforward_channels=feedforward_channels // 2,
                                          output_channels=output_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          act_cfg=dict(type='GELU'),
                                          ffn_drop=ffn_drop)

        self.norm = nn.BatchNorm2d(output_channels)
        self.multiscale_cbam = MultiscaleCBAMLayer(output_channels, kernel_size)

        self.residual = nn.ModuleList([
            # 将 bias = False 改变为 bias = True
            nn.Conv2d(embed_dims, output_channels, kernel_size=7, stride=4, padding=3, bias=True),  # 普通残差  
            nn.Conv2d(embed_dims, output_channels, kernel_size=3, stride=2, padding=1, bias=True),  # 下采样残差
            nn.Conv2d(embed_dims, output_channels, kernel_size=3, stride=2, padding=1, bias=True),  # 下采样残差
            nn.Sequential(  # 带 BN 的残差
                nn.Conv2d(embed_dims, output_channels, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(output_channels)
            )
        ])

    def forward(self, x):

        # [16,3,256,256] -> [16,32,64,64] -> [16,32,64,64]

        out = self.layers(x)
        out = self.multiscale_cbam(out)

        _, _, H, W=out.shape

        if H == 64:
            identity = self.residual[0](x)
        elif H == 32:
            identity = self.residual[1](x)
        elif H == 16:
            identity = self.residual[2](x)
        else:
            identity = self.residual[3](x)

        if identity.shape != out.shape:
            print(f"mismatch: identity={identity.shape}, out={out.shape}")
            identity = F.interpolate(identity, size=out.shape[2:], mode='bilinear', align_corners=False)

        out += identity
        return out

# 原始的
"""
class CrossEncoderFusion(nn.Module):
    def __init__(self):
        super(CrossEncoderFusion, self).__init__()

    def forward(self, x, cnn_encoder_layers, transformer_encoder_layers, fusion_conv_layers, out_indices):
        outs = []
        cnn_encoder_out = x

        for i, (cnn_encoder_layer, transformer_encoder_layer) in enumerate(
                zip(cnn_encoder_layers, transformer_encoder_layers)):
            
            cnn_encoder_out = cnn_encoder_layer(cnn_encoder_out) # [16, 3, 256, 256] → [16, 32, 128, 128] → [16, 64, 64, 64] → [16, 128, 32, 32] → [16, 256, 16, 16]
            
            x, hw_shape = transformer_encoder_layer[0](x)
            for block in transformer_encoder_layer[1]:
                x = block(x, hw_shape)

            
            # 过了block后的x的shape
            # torch.Size([16, 4096, 32])   torch.Size([16, 1024, 64])  torch.Size([16, 256, 160])  torch.Size([16, 64, 192])

            x = transformer_encoder_layer[2](x)
            x = nlc_to_nchw(x, hw_shape)

            x = torch.cat((x, cnn_encoder_out), dim=1)
            
            x = fusion_conv_layers[i](x)

            if i in out_indices:
                outs.append(x)
        return outs

"""

# cross 变体
class DenseLayer(nn.Module):
    """Dense Block 内部的一层"""
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=True)
        
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)  # 维度拼接，增加通道数

class DenseBlock(nn.Module):
    """包含多个 DenseLayer,并使用 1x1 卷积降维"""
    def __init__(self, num_layers, in_channels, growth_rate, out_channels):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels  # 记录当前通道数
        
        for _ in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate  # 每层增加 growth_rate 个通道
        
        # 使用 1x1 卷积降维到目标通道数
        self.conv1x1 = nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # 逐层拼接
        x = self.conv1x1(x)  # 1x1 降维
        x = F.relu(self.bn(x))  # BN + ReLU
        return x

class CrossEncoderFusion(nn.Module):
    def __init__(self, fusion_out_channels):
        super(CrossEncoderFusion, self).__init__()
        
        self.fusion_blocks = nn.ModuleList([
            DenseBlock(num_layers, in_channels, growth_rate, out_channels) 
            for in_channels, growth_rate, num_layers, out_channels in fusion_out_channels
        ])

    def forward(self, x, cnn_encoder_layers, transformer_encoder_layers, out_indices):
        outs = []
        cnn_encoder_out = x

        # 刚来的 x.shape 为 [16, 3, 256, 256]


        for i, (cnn_encoder_layer, transformer_encoder_layer) in enumerate(zip(cnn_encoder_layers, transformer_encoder_layers)):
            # CNN 分支
            cnn_encoder_out = cnn_encoder_layer(cnn_encoder_out)

            # 经过 Transformer 编码器的每一层 : transformer_encoder_layer[0] -> transformer_encoder_layer[1]
            # 1. torch.Size([16, 4096, 32]) (64, 64)   ->   torch.Size([16, 4096, 32]) (64, 64)
            # 2. torch.Size([16, 1024, 64]) (32, 32)   ->   torch.Size([16, 1024, 64]) (32, 32)
            # 3. torch.Size([16, 256, 160]) (16, 16)   ->   torch.Size([16, 256, 160]) (16, 16)
            # 4. torch.Size([16, 64, 192]) (8, 8)   ->   torch.Size([16, 64, 192]) (8, 8)
            # Transformer 分支
            x, hw_shape = transformer_encoder_layer[0](x)


            for block in transformer_encoder_layer[1]:
                x = block(x)
                
            x = transformer_encoder_layer[2](x)
            x = nlc_to_nchw(x, hw_shape)

            # 拼接 CNN 和 Transformer 特征
            fusion_input = torch.cat((cnn_encoder_out, x), dim=1)

            # 经过 DenseBlock 进行融合
            x = self.fusion_blocks[i](fusion_input)

            if i in out_indices:
                outs.append(x)

        return outs

@BACKBONES.register_module()
class LEFormer(BaseModule):
    """The backbone of LEFormer.

    This backbone is the implementation of `LEFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 32.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [2, 2, 2, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 6].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        pool_numbers (int): the number of Pooling Transformer Layers. Default 1.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3, # DDD输入图像的通道数，RGB 图像一般是 3。
                 embed_dims=32, # 基础嵌入维度，用于 Transformer 的输入通道数。
                 num_stages=4, # 总共有 4 个阶段，每个阶段可以有不同的 层。
                 num_layers=(2, 4, 6, 2), # 每个阶段的 Transformer 层数。
                 num_heads=(1, 2, 5, 6), # 每个阶段的 Transformer 多头注意力的头数。
                 patch_sizes=(7, 3, 3, 3), # 每个阶段的 Patch Embedding 卷积核大小。
                 strides=(4, 2, 2, 2), # 每个阶段的 Patch Embedding 步长。
                 sr_ratios=(8, 4, 2, 1), # 每个阶段的 Transformer 编码层的空间缩减率。
                 out_indices=(0, 1, 2, 3), # 注意力缩小比例，用于减少计算量。
                 mlp_ratio=4,  # MLP 隐藏维度与嵌入维度的比例。
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 pool_numbers=1, # 控制使用 PTL 的层数
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(LEFormer, self).__init__(init_cfg=init_cfg)

        # cross 变体
        self.fusion_out_channels_5 = [
            (64, 16, 5, 32),  
            (128, 32, 5, 64),  
            (320, 64, 5, 160),
            (384,128, 5, 192)
        ]
        self.fusion_out_channels_4 = [
            (64, 16, 4, 32),  
            (128, 32, 4, 64),  
            (320, 64, 4, 160),
            (384,128, 4, 192)
        ]
        self.fusion_out_channels_3 = [
            (64, 16, 3, 32),  
            (128, 32, 3, 64),  
            (320, 64, 3, 160),
            (384,128, 3, 192)
        ]
        self.fusion_out_channels_2 = [
            (64, 16, 2, 32),  
            (128, 32, 2, 64),  
            (320, 64, 2, 160),
            (384,128, 2, 192)
        ]

        self.cross_encoder_fusion=CrossEncoderFusion(self.fusion_out_channels_5)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        embed_dims_list = []
        feedforward_channels_list = []
        self.transformer_encoder_layers = ModuleList()
        for i, num_layer in enumerate(num_layers):  # num_layer 是每个阶段的 Transformer 层数
            embed_dims_i = embed_dims * num_heads[i]
            embed_dims_list.append(embed_dims_i)
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            feedforward_channels_list.append(mlp_ratio * embed_dims_i)

            
            if embed_dims_i==32:
                self.input_resolution=(64,64)
            elif embed_dims_i==64:
                self.input_resolution=(32,32)
            elif embed_dims_i==160:
                self.input_resolution=(16,16)
            elif embed_dims_i==192:
                self.input_resolution=(8,8)
                
            # layer = ModuleList([
            #     MLLABlock(
            #         dim=embed_dims_i, 
            #         input_resolution=self.input_resolution,
            #         num_heads=4
            #     ) 
            # ])   
                
            layer = ModuleList([
                MLLABlock(
                    dim=embed_dims_i, 
                    input_resolution=self.input_resolution,
                    num_heads=4
                ) 
                    for idx in range(num_layer)
            ])
           
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            
            self.transformer_encoder_layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.cnn_encoder_layers = nn.ModuleList()

        for i in range(num_stages):
            self.cnn_encoder_layers.append(
                CnnEncoderLayer(
                    embed_dims=self.in_channels if i == 0 else embed_dims_list[i - 1],
                    feedforward_channels=feedforward_channels_list[i],
                    output_channels=embed_dims_list[i],
                    kernel_size=patch_sizes[i],
                    stride=strides[i],
                    padding=patch_sizes[i] // 2,
                    ffn_drop=drop_rate
                )
            )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(LEFormer, self).init_weights()

    def forward(self, x):
             return self.cross_encoder_fusion(
                x,
                cnn_encoder_layers=self.cnn_encoder_layers,
                transformer_encoder_layers=self.transformer_encoder_layers,
                out_indices=self.out_indices
            )