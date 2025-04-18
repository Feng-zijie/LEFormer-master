import torch.nn as nn
import torch.nn.functional as F
import torch
# 论文：Reciprocal Attention Mixing Transformer for Lightweight Image Restoration(CVPR 2024 Workshop)
# 论文地址：https://arxiv.org/abs/2305.11474
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
# H-RAMi(Hierarchical Reciprocal Attention Mixer)
class MobiVari1(nn.Module):  # MobileNet v1 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None):
        super(MobiVari1, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim or dim

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, stride, kernel_size // 2, groups=dim)
        self.pw_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)
        self.act = act()

    def forward(self, x):
        out = self.act(self.pw_conv(self.act(self.dw_conv(x)) + x))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * self.kernel_size * self.kernel_size * self.dim + H * W * 1 * 1 * self.dim * self.out_dim  # self.dw_conv + self.pw_conv
        return flops


class MobiVari2(MobiVari1):  # MobileNet v2 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None, exp_factor=1.2, expand_groups=4):
        super(MobiVari2, self).__init__(dim, kernel_size, stride, act, out_dim)
        self.expand_groups = expand_groups
        expand_dim = int(dim * exp_factor)
        expand_dim = expand_dim + (expand_groups - expand_dim % expand_groups)
        self.expand_dim = expand_dim

        self.exp_conv = nn.Conv2d(dim, self.expand_dim, 1, 1, 0, groups=expand_groups)
        self.dw_conv = nn.Conv2d(expand_dim, expand_dim, kernel_size, stride, kernel_size // 2, groups=expand_dim)
        self.pw_conv = nn.Conv2d(expand_dim, self.out_dim, 1, 1, 0)

    def forward(self, x):
        x1 = self.act(self.exp_conv(x))
        out = self.pw_conv(self.act(self.dw_conv(x1) + x1))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * 1 * 1 * (self.dim // self.expand_groups) * self.expand_dim  # self.exp_conv
        flops += H * W * self.kernel_size * self.kernel_size * self.expand_dim  # self.dw_conv
        flops += H * W * 1 * 1 * self.expand_dim * self.out_dim  # self.pw_conv
        return flops

class HRAMi(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, mv_ver=1, mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(HRAMi, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        # 自动计算输入通道数
        self.in_dim = None  # 运行时动态计算

        self.mv_ver = mv_ver
        self.mv_act = mv_act
        self.exp_factor = exp_factor
        self.expand_groups = expand_groups

    def forward(self, attn_list):
        # 计算实际输入通道
        in_dim = sum([attn.shape[1] for attn in attn_list])

        # 初始化 MobiVari（仅第一次 forward 运行时）
        if self.in_dim is None:
            self.in_dim = in_dim
            if self.mv_ver == 1:
                self.mobivari = MobiVari1(self.in_dim, self.kernel_size, 1, act=self.mv_act, out_dim=self.dim)
            else:
                self.mobivari = MobiVari2(self.in_dim, self.kernel_size, 1, act=self.mv_act, out_dim=self.dim,
                                          exp_factor=self.exp_factor, expand_groups=self.expand_groups)
        
        # 进行特征融合
        for i, attn in enumerate(attn_list[:-1]):
            attn = F.pixel_shuffle(attn, 2 ** i)
            x = attn if i == 0 else torch.cat([x, attn], dim=1)

        x = torch.cat([x, attn_list[-1]], dim=1)
        x = self.mobivari(x)
        return x


    def flops(self, resolutions):
        return self.mobivari.flops(resolutions)


if __name__ == '__main__':
    hrami = HRAMi(dim=160)
    x=torch.randn(16, 160, 64, 64)
    y=torch.randn(16, 160, 64, 64)
    # Create sample input tensors
    # Assume the input tensors have spatial dimensions of 32x32, 16x16, 8x8, etc.
    input = [x,y]

    # Pass the input through HRAMi
    output = hrami(input)

    # Print the shapes of input and output
    print(f"Input shapes: {[attn.shape for attn in input]}")
    print(output.size())