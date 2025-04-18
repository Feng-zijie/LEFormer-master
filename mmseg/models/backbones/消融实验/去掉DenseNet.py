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

class TransformerEncoderLayerCustom(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, dropout=0.1, activation='gelu'):
        super(TransformerEncoderLayerCustom, self).__init__()
        # 使用标准的MultiheadAttention，如果需要可以替换为自定义的LinearAttention
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(int(dim * mlp_ratio), dim)
        
        # 层归一化和Dropout
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Input tensor of shape (B, N, C)
            src_mask: Mask for the src sequence (optional)
            src_key_padding_mask: Mask for the src keys per batch (optional)
        """
        # Multi-head Self Attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed Forward
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, dim)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, C)
        Returns:
            Tensor with positional encoding added.
        """
        B, N, C = x.shape
        return x + self.pe[:N, :, :].to(x.device)

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
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, shortcut):
        H, W = self.input_resolution
        B, L, C = x.shape

        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        # Linear Attention
        x = self.attn(x)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        return x
    
# Mamba-Inspired Linear Attention (MILA)
class MILA(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()
        
        self.mlla_block = MLLABlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **kwargs
        )
        
        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
    def forward(self, x):
        H, W = self.mlla_block.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x + self.cpe(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x
        x = self.norm1(x)
        
        x=self.mlla_block(x,shortcut)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    kernel = kernel.to(img.device)  # 将 kernel 移动到 img 的设备
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out


def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))


def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff


def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))).unsqueeze(-1).unsqueeze(-1)
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))).unsqueeze(-1).unsqueeze(-1)
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

# Edge-Guided Attention Module
class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(in_channels)
        self.pred_conv = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)  # 用于生成 pred

    def forward(self, edge_feature, x):
        residual = x
        xsize = x.size()[2:]

        pred = torch.sigmoid(self.pred_conv(x))

        # reverse attention
        background_att = 1 - pred
        background_x = x * background_att

        # boudary attention
        edge_pred = make_laplace(pred, 1)
        pred_feature = x * edge_pred

        # high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out = self.cbam(out)
        return out

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
        
        self.ega_block = EGA(output_channels)

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

        # self.embed_dims => 3, 32, 64, 160   # self.output_channels => 32, 64, 160, 192
        # 第一次 x[16, 3, 256, 256] -> out1[16, 32, 64, 64] -> out2[16, 32, 64, 64]
        # 第二次 x[16, 32, 64, 64] -> out1[16, 64, 32, 32] -> out2[16, 64, 32, 32]
        # 第三次 x[16, 64, 32, 32] -> out1[16, 160, 16, 16] -> out2[16, 160, 16, 16]
        # 第四次 x[16, 160, 16, 16] -> out1[16, 192, 8, 8] -> out2[16, 192, 8, 8]

        out = self.layers(x)
        
        
        # out = self.multiscale_cbam(out) # 经过后 [16, 32, 64, 64] -> [16, 64, 32, 32] -> [16, 160, 16, 16] -> [16, 192, 8, 8]
        edge_feature = make_laplace(out, channels=self.output_channels)
        out = self.ega_block(edge_feature,out)

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
        
        out = F.relu(out) 
        return out

# 原始的
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
                x = block(x)

            
            # 过了block后的x的shape
            # torch.Size([16, 4096, 32])   torch.Size([16, 1024, 64])  torch.Size([16, 256, 160])  torch.Size([16, 64, 192])

            x = transformer_encoder_layer[2](x)
            x = nlc_to_nchw(x, hw_shape)

            x = torch.cat((x, cnn_encoder_out), dim=1)
            
            x = fusion_conv_layers[i](x)

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
                 in_channels=3, # 输入图像的通道数，RGB 图像一般是 3。
                 embed_dims=32, # 基础嵌入维度，用于 Transformer 的输入通道数。
                 num_stages=4, # 总共有 4 个阶段，每个阶段可以有不同的 层。
                 num_layers=(2, 2, 2, 3), # 每个阶段的 Transformer 层数。
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
        
        self.cross_encoder_fusion=CrossEncoderFusion() 

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
                
            layer = ModuleList([
                MILA(
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
        self.fusion_conv_layers = nn.ModuleList()

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
            self.fusion_conv_layers.append(
                Conv2d(
                    in_channels=embed_dims_list[i] * 2,
                    out_channels=embed_dims_list[i],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
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
                fusion_conv_layers=self.fusion_conv_layers,
                out_indices=self.out_indices
            )