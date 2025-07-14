import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from .conv import Conv
from einops import rearrange
from .RFA import RFAConv_Group

__all__ = (
    "PFM",
    "SED",
    "SEG",
    "DE",
    "ASFD",
    "CoordinateAttention",
    "EMA",
    "FCDN_MultiAttention",
    "ACFM",
    "LSKA",
    "SPPF_LSKA",
    "SE_Block",
    "MSCAAttention",
    "ECA",
    "deformable_LKA_Attention",
    "DeformConv",
    "RFEA",
)

class CausalDilatedConv1d(nn.Module):
    """因果扩张卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation_rate
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation_rate, padding=0
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class ForgetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rate):
        super().__init__()
        self.causal_conv = CausalDilatedConv1d(
            channels, channels, kernel_size, dilation_rate
        )

    def forward(self, x):
        conv_out = self.causal_conv(x)
        forget_gate = torch.sigmoid(conv_out)
        return forget_gate * x


class UpdateBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation_rate):
        super().__init__()
        self.causal_conv = CausalDilatedConv1d(
            channels, channels, kernel_size, dilation_rate
        )

    def forward(self, x):
        conv_out = self.causal_conv(x)
        new_features = torch.tanh(conv_out)
        return new_features * x


class FEB(nn.Module):
    """遗忘增强块"""

    def __init__(self, channels, kernel_size, dilation_rate):
        super().__init__()
        self.forget_block = ForgetBlock(channels, kernel_size, dilation_rate)
        self.update_block = UpdateBlock(channels, kernel_size, dilation_rate)

    def forward(self, x):
        x_f = self.forget_block(x)
        x_u = self.update_block(x)
        return x_f + x_u


class PFM(nn.Module):
    """YOLOv8适配版特征提升模块"""

    def __init__(self, c1, num_feb=3, kernel_size=3, dilation_rates=[1, 2, 4]):
        super().__init__()
        self.channels = c1
        self.febs = nn.ModuleList([
            FEB(c1, kernel_size, dilation_rates[i % len(dilation_rates)])
            for i in range(num_feb)
        ])

    def forward(self, x):
        # 输入形状: [B, C, H, W]
        B, C, H, W = x.shape

        # 展平空间维度: [B, C, H*W]
        x_flat = x.view(B, C, -1)

        # 通过FEB序列处理
        for feb in self.febs:
            x_flat = feb(x_flat)

        # 恢复原始形状: [B, C, H, W]
        return x_flat.view(B, C, H, W)

class SEG(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.fusion_conv = nn.Conv2d(128, 64, 1)

        # 修改这里：输入通道改为 64
        self.conv6 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))
        c4 = F.relu(self.bn4(self.conv4(c3)))
        c5 = F.relu(self.bn5(self.conv5(c4)))

        fused = torch.cat([c1, c5], dim=1)  # [B, 128, H, W]
        fused = F.relu(self.fusion_conv(fused))  # [B, 64, H, W]
        out = torch.tanh(self.conv6(fused))
        return out

class SED(nn.Module):
    def __init__(self):
        super(SED, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(64 * 8 * 8, 2)  # 根据输入尺寸调整

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class DE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DE, self).__init__()
        # 多尺度卷积
        self.conv3 = nn.Conv2d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels // 3, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, (out_channels // 3)+1, kernel_size=7, padding=3)

        self.relu = nn.ReLU(inplace=True)
        # 融合后卷积
        self.fuse_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        feat3 = self.relu(self.conv3(x))
        feat5 = self.relu(self.conv5(x))
        feat7 = self.relu(self.conv7(x))

        # 特征拼接
        fused_feat = torch.cat([feat3, feat5, feat7], dim=1)
        # 融合特征
        out = self.relu(self.fuse_conv(fused_feat))
        return out


class ASFD(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, reduction=4):
        super().__init__()
        self.scale = scale_factor
        self.s2 = scale_factor ** 2

        # Spatial Perception分支
        self.spatial_perception = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
            nn.Conv2d(2 * in_channels, self.s2, 7, padding=3)
        )

        # Feature Activation分支
        self.feature_activation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 通道调整
        self.channel_adjust = nn.Conv2d(in_channels * self.s2, out_channels, 1)

    def spatial_separation(self, x):
        """空间分离阶段"""
        B, C, H, W = x.shape
        x = x.unfold(2, self.scale, self.scale).unfold(3, self.scale, self.scale)
        x = x.contiguous().view(B, C, self.s2, H // self.scale, W // self.scale)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, self.s2 * C, H // self.scale, W // self.scale)
        return x.chunk(self.s2, dim=1)  # 分割为s²个特征图

    def forward(self, x):
        # 空间分离阶段
        F_prime = self.spatial_separation(x)  # list of s² tensors

        # 空间感知分支
        sum_F = torch.stack(F_prime).sum(dim=0)
        F_p3 = self.spatial_perception(sum_F)
        H_p = F.softmax(F_p3, dim=1)  # [B, s2, H/s, W/s]

        # 特征激活分支
        H_f = self.feature_activation(sum_F)  # [B, C, 1, 1]

        # 融合过程
        outputs = []
        for i in range(self.s2):
            # 空间注意力分量
            H_p_i = H_p[:, i:i + 1, :, :]  # [B,1,H/s,W/s]

            # 通道注意力分量
            W_i = H_p_i * H_f  # [B,C,H/s,W/s]

            # 特征图加权
            weighted = F_prime[i] * W_i
            outputs.append(weighted)

        # 拼接并调整通道
        F_down = torch.cat(outputs, dim=1)
        F_down = self.channel_adjust(F_down)

        return F_down


# CA CVPR 2021 Coordinate Attentions
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    def __init__(self, inc, outc, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 横向池化
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))  # 纵向池化

        # Reduction层 减小特征图的通道数
        reduced_channel = max(8, inc // reduction)
        self.conv1 = nn.Conv2d(inc, reduced_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(reduced_channel)
        self.act = h_swish()
        # 分别在水平和垂直方向卷积
        self.conv_h = nn.Conv2d(reduced_channel, outc, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(reduced_channel, outc, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x  # 保留输入特征图作为最后的输出
        n, c, h, w = x.size()

        x_h = self.pool_h(x)  # N*C*H*1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # N*C*1*W -> N*C*W*1

        # 拼接后进行卷积和激活
        y = torch.cat([x_h, x_w], dim=2)  # N*C*(H+W)*1
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分别提取横向和纵向特征
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成横向和纵向注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 输出特征图与注意力权重相乘
        out = identity * a_w * a_h
        return out

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class FCDN_MultiAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2, kernel_size=7):
        """
        FCDN多重注意力模块
        :param in_channels: 输入通道数
        :param reduction_ratio: 通道压缩比例
        :param kernel_size: 空间注意力卷积核大小
        """
        super(FCDN_MultiAttention, self).__init__()

        # 通道注意力分支
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)

        # 空间注意力分支
        self.spatial_attention = SpatialAttention(kernel_size)

        # 特征交叉连接模块
        self.cross_link = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        # 输出融合卷积
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)

        # 上采样层（用于小目标特征增强）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 原始输入特征保存
        identity = x

        # 通道注意力分支
        ca = self.channel_attention(x)
        ca_out = x * ca

        # 空间注意力分支
        sa = self.spatial_attention(x)
        sa_out = x * sa

        # 特征交叉连接
        cross_feature = torch.cat([ca_out, sa_out], dim=1)
        cross_feature = self.cross_link(cross_feature)

        # 特征融合
        fused_feature = torch.cat([cross_feature, identity], dim=1)
        fused_feature = self.fusion_conv(fused_feature)

        # 上采样增强小目标特征
        output = self.upsample(fused_feature)

        return output


class ChannelAttention(nn.Module):
    """通道注意力子模块"""

    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x).view(b, c))

        # 特征融合
        channel_att = avg_out + max_out
        channel_att = torch.sigmoid(channel_att).view(b, c, 1, 1)

        return channel_att


class SpatialAttention(nn.Module):
    """空间注意力子模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度上的最大池化和平均池化
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # 拼接特征
        spatial_feat = torch.cat([max_pool, avg_pool], dim=1)

        # 卷积生成空间注意力图
        spatial_att = self.conv(spatial_feat)
        spatial_att = self.sigmoid(spatial_att)

        return spatial_att


class MSCA(nn.Module):
    """
    Multi-Scale Channel Attention (MSCA)
    双分支：全局分支（Global AvgPool->1x1 conv）和
    局部分支（直接1x1 conv），最终融合并输出对每个通道的注意力权重。
    """
    def __init__(self, channels, reduction=16):
        super(MSCA, self).__init__()
        mid = channels // reduction
        # 全局分支
        self.glob_pool = nn.AdaptiveAvgPool2d(1)
        self.glob_conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.glob_relu = nn.ReLU(inplace=True)
        self.glob_conv2 = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        # 局部分支
        self.local_conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.local_relu = nn.ReLU(inplace=True)
        self.local_conv2 = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        # 输出激活
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        # 全局分支
        g = self.glob_pool(x)           # [B, C, 1, 1]
        g = self.glob_conv1(g)          # [B, C//r, 1, 1]
        g = self.glob_relu(g)
        g = self.glob_conv2(g)          # [B, C, 1, 1]

        # 局部分支
        l = self.local_conv1(x)         # [B, C//r, H, W]
        l = self.local_relu(l)
        l = self.local_conv2(l)         # [B, C, H, W]

        # 将全局分支的输出 broadcast 到 [B,C,H,W] 后与局部分支相加
        g_up = g.expand_as(x)
        a = g_up + l                     # [B, C, H, W]
        a = self.sigmoid(a)
        return a                         # 注意力权重 [0,1]

class ACFM(nn.Module):
    """
    Attention-induced Cross-level Fusion Module
    融合两路特征 Fa, Fb：
      1) 对空间更小的那一路上采样到 Fa 的大小
      2) F_sum = Fa + Fb_up
      3) α = MSCA(F_sum)
      4) F_fuse = α * Fa + (1-α) * Fb_up
      5) 3x3 conv + BN + ReLU -> 输出 F_out
    """
    def __init__(self, channels, reduction=16):
        super(ACFM, self).__init__()
        self.channels = channels
        self.msca = MSCA(channels, reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 如果 x 是列表或元组，直接展开；否则认为是 concat 后的张量
        if isinstance(x, (list, tuple)):
            Fa, Fb = x
        else:
            # x 的通道数应该是 2*channels
            c = x.shape[1] // 2
            Fa, Fb = x[:, :c], x[:, c:]
        if Fa.size()[2:] == Fb.size()[2:]:
            Fb_up = Fb
        else:
            Fb_up = F.interpolate(Fb, size=Fa.size()[2:], mode='bilinear', align_corners=False)

        # 融合
        F_sum = Fa + Fb_up              # 元素级加
        alpha = self.msca(F_sum)        # [B,C,H,W]
        F_fuse = alpha * Fa + (1 - alpha) * Fb_up

        # 最后再做一次 conv+BN+ReLU
        out = self.conv(F_fuse)
        return out


class LSKA(nn.Module):
    # Large-Separable-Kernel-Attention
    # https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention/tree/main
    def __init__(self, dim, k_size=7):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                            dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                            dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=dim,
                                            dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=dim,
                                            dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1, 1), padding=(0, 18), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1, 1), padding=(18, 0), groups=dim,
                                            dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 24), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1, 1), padding=(24, 0), groups=dim,
                                            dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


class SPPF_LSKA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = MSCAAttention(c_ * 4)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))

'''-------------一、SE模块-----------------------------'''
# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


class MSCAAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class deformable_LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class PolarizedAttention(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PolarizedAttention, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()
        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)
        mask_ch = self.sigmoid(context)
        out = x * mask_ch
        return out

    def channel_pool(self, x):
        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def forward(self, x):
        # 并联
        # context_channel = self.spatial_pool(x)
        # context_spatial = self.channel_pool(x)
        # out = context_spatial + context_channel

        # 串联
        out = self.spatial_pool(x)
        out = self.channel_pool(out)

        return out

class MLCA(nn.Module):
    def __init__(self, in_size,local_size=5,gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)


        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        # y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)
        y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1)  # 代码修正
        # print(y_global_transpose.size())
        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        # print(att_local.size())
        # print(att_global.size())
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])
        # print(att_all.size())
        x=x*att_all
        return x

# class RFAConv_Group(nn.Module):
#     """ 基于 Group Conv 的 RFAConv，支持自定义 padding 和 dilation """
#     def __init__(self,
#                  in_channel: int,
#                  out_channel: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  padding: int = None,
#                  dilation: int = 1):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#         # 如果用户没传 padding，就沿用 classic 的 kernel_size//2
#         self.padding = padding if padding is not None else kernel_size // 2
#         self.dilation = dilation
#
#         # 1) 用来生成每个位置的注意力权重：先 pad + avg_pool，再 1×1 depthwise conv
#         self.get_pw = nn.Conv2d(
#             in_channel,
#             in_channel * (kernel_size ** 2),
#             kernel_size=1,
#             groups=in_channel,
#             bias=False
#         )
#
#         # 2) 用来提取感受野空间特征：group conv + BN + ReLU
#         self.gen_feat = nn.Sequential(
#             nn.Conv2d(
#                 in_channel,
#                 in_channel * (kernel_size ** 2),
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=self.padding,
#                 dilation=self.dilation,
#                 groups=in_channel,
#                 bias=False
#             ),
#             nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
#             nn.ReLU(inplace=True)
#         )
#
#         # 3) 最后的融合卷积：常规 conv + BN + ReLU
#         #    stride = kernel_size 用于把 h',w' -> h'*k, w'*k
#         self.fuse = nn.Sequential(
#             nn.Conv2d(
#                 in_channel,
#                 out_channel,
#                 kernel_size=kernel_size,
#                 stride=kernel_size,
#                 padding=0,
#                 bias=False
#             ),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         k2 = self.kernel_size ** 2
#
#         # —— 1) 生成权重 ——
#         # 手工 pad，避免 AvgPool2d padding 越界
#         if self.padding > 0:
#             x_pad = F.pad(x,
#                           pad=(self.padding, self.padding, self.padding, self.padding),
#                           mode='constant', value=0)
#         else:
#             x_pad = x
#         weight = F.avg_pool2d(
#             x_pad,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=0
#         )
#         # b, c*k2, h', w'
#         weight = self.get_pw(weight)
#         h2, w2 = weight.shape[2:]
#         weight = weight.view(b, c, k2, h2, w2).softmax(dim=2)
#
#         # —— 2) 生成感受野特征 ——
#         feat = self.gen_feat(x)              # b, c*k2, h', w'
#         feat = feat.view(b, c, k2, h2, w2)
#
#         # —— 3) 加权、重排成大图 ——
#         weighted = feat * weight             # b, c, k2, h', w'
#         conv_data = rearrange(
#             weighted,
#             'b c (n1 n2) h w -> b c (h n1) (w n2)',
#             n1=self.kernel_size,
#             n2=self.kernel_size
#         )
#
#         # —— 4) 最后一层融合 conv+bn+act ——
#         out = self.fuse(conv_data)
#         return out


class TridentBlock(nn.Module):
    def __init__(self, c1, c2, stride=1, e=0.5, padding=[1,2,3], dilate=[1,2,3]):
        super().__init__()
        self.stride = stride
        c_ = int(c2 * e)
        self.padding = padding
        self.dilate  = dilate

        # 用 RFAConv_Group 替代 1×1 conv+BN+SiLU
        self.rfa1 = RFAConv_Group(in_channel=c1,
                                  out_channel=c_,
                                  kernel_size=1,
                                  stride=1)

        # 用三个共享权重的 RFAConv_Group 替代 3×3 conv+BN+SiLU，
        # 只是在不同分支上设置不同的 padding/dilation
        self.rfa2_small  = RFAConv_Group(c_, c2, kernel_size=3, stride=stride)
        self.rfa2_middle = RFAConv_Group(c_, c2, kernel_size=3, stride=stride)
        self.rfa2_big    = RFAConv_Group(c_, c2, kernel_size=3, stride=stride)

        # 最后的残差激活
        self.act = nn.SiLU()

    def forward_for_small(self, x):
        res = x
        out = self.rfa1(x)
        # RFAConv_Group 自带 BN+ReLU
        # 但它不支持 dilation/padding 调节，这里 hack 一下
        # 你也可以在 RFAConv_Group 中加 dilation/padding 参数
        out = nn.functional.conv2d(
            out,
            weight=self.rfa2_small.conv[0].weight,
            bias=self.rfa2_small.conv[0].bias,
            stride=self.stride,
            padding=self.padding[0],
            dilation=self.dilate[0],
        )
        out = self.rfa2_small.conv[1](out)  # BN
        out = self.rfa2_small.conv[2](out)  # ReLU

        out = out + res
        return self.act(out)

    def forward_for_middle(self, x):
        res = x
        out = self.rfa1(x)
        out = nn.functional.conv2d(
            out,
            weight=self.rfa2_middle.conv[0].weight,
            bias=self.rfa2_middle.conv[0].bias,
            stride=self.stride,
            padding=self.padding[1],
            dilation=self.dilate[1],
        )
        out = self.rfa2_middle.conv[1](out)
        out = self.rfa2_middle.conv[2](out)
        out = out + res
        return self.act(out)

    def forward_for_big(self, x):
        res = x
        out = self.rfa1(x)
        out = nn.functional.conv2d(
            out,
            weight=self.rfa2_big.conv[0].weight,
            bias=self.rfa2_big.conv[0].bias,
            stride=self.stride,
            padding=self.padding[2],
            dilation=self.dilate[2],
        )
        out = self.rfa2_big.conv[1](out)
        out = self.rfa2_big.conv[2](out)
        out = out + res
        return self.act(out)

    def forward(self, x):
        # 如果你希望三条分支同时作用于同一个输入
        x1 = self.forward_for_small(x)
        x2 = self.forward_for_middle(x)
        x3 = self.forward_for_big(x)
        return [x1, x2, x3]


class RFEA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, stride=1):
        super().__init__()
        # 第一个 TridentBlock 开启三路并行
        layers = [ TridentBlock(c1, c2, stride=stride, e=e) ]
        for _ in range(1, n):
            # 后续如果只需要单路，也可以复用上面的 block
            layers.append(TridentBlock(c2, c2, e=e))
        self.layers = nn.Sequential(*layers)
        self.bn  = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # 这里 x 直接就是一个 Tensor
        feats = self.layers(x)      # 返回 [f1,f2,f3]
        out = feats[0] + feats[1] + feats[2] + x
        return self.act(self.bn(out))