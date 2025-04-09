import torch.nn as nn
import torch

# 定义GAM_Attention类
class GCSA(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GCSA, self).__init__()

        # 通道注意力子模块
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),  # 线性层，将通道数缩减到1/rate
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(int(in_channels / rate), in_channels)  # 线性层，将通道数恢复到原始大小
        )

        # 空间注意力子模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),  # 7x7卷积，通道数缩减到1/rate
            nn.BatchNorm2d(int(in_channels / rate)),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),  # 7x7卷积，恢复到原始通道数
            nn.BatchNorm2d(in_channels)  # 批归一化
        )

    # 通道洗牌操作函数
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 调整形状，分组
        x = x.view(batchsize, groups, channels_per_group, height, width)
        # 转置，打乱组内通道
        x = torch.transpose(x, 1, 2).contiguous()
        # 恢复原始形状
        x = x.view(batchsize, -1, height, width)

        return x

    # 前向传播函数
    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入张量的形状
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  # 调整形状，便于通道注意力操作
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  # 应用通道注意力
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()  # 调整回原始形状，并应用Sigmoid激活函数

        x = x * x_channel_att  # 将输入特征图与通道注意力图逐元素相乘

        x = self.channel_shuffle(x, groups=4) # 添加通道洗牌操作[根据自己的任务设定组数,2也行,大于4也行，看效果选择]

        x_spatial_att = self.spatial_attention(x).sigmoid()  # 应用空间注意力，并应用Sigmoid激活函数

        out = x * x_spatial_att  # 将输入特征图与空间注意力图逐元素相乘

        return out  # 返回输出特征图

# 测试代码
if __name__ == '__main__':
    x = torch.randn(16, 192, 8, 8)  # 创建随机输入张量
    b, c, h, w = x.shape  # 获取输入张量的形状
    net = GCSA(in_channels=c)  # 初始化GAM_Attention模块
    y = net(x)  # 前向传播
    print(y.size())  # 打印输出张量的形状



