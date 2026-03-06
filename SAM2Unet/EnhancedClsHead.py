import torch
import torch.nn as nn
import torch.nn.functional as F


# 补充 h_swish 的标准实现 (PyTorch 1.6+ 原生支持 nn.Hardswish)
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3.) / 6.


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class EnhancedClsHead(nn.Module):
    """
    单输入增强分类头 (基于 CoordAtt + Avg/Max 混合池化)
    仅接收最深层特征 x4
    """

    def __init__(self, in_channels, hidden_dim, num_classes):
        super(EnhancedClsHead, self).__init__()
        # 1. Attention Module
        self.attention = CoordAtt(in_channels, in_channels)

        # 2. Classifier (维度为 in_channels * 2，因为拼接了 avg_pool 和 max_pool)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: [B, C, H, W] (对应传入的 x4)
        x = self.attention(x)

        # Mixed Pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        max_pool = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)

        # Concatenate
        flat = torch.cat([avg_pool, max_pool], dim=1)

        return self.classifier(flat)