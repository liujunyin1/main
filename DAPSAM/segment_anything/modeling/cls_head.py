import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMBlock(nn.Module):
    """简化的通道+空间注意力模块"""

    def __init__(self, in_channels, reduction=16):
        super(CBAMBlock, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 注意：使用降维再升维，减少参数量
        mid_channels = max(in_channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        x = x * self.sigmoid(out)

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv_spatial(out)
        return x * self.sigmoid(out)


class GeMPooling(nn.Module):
    """Generalized Mean Pooling for robust feature aggregation in medical imaging."""

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class SAMAdvancedClsHead(nn.Module):
    """
    专门为 SAM 适配的高级分类头:
    1. CBAM (Channel + Spatial Attention)
    2. Adaptive GeM + Avg + Max Pooling (Triple Pooling)
    3. LayerNorm + GELU 现代化 MLP 分类器
    """

    def __init__(self, in_channels=256, hidden_dim=256, num_classes=7, extra_channels=[]):
        super().__init__()

        # 1. 空间与通道注意力 (处理 SAM 输出的 [B, 256, H, W])
        self.attn = CBAMBlock(in_channels)

        # 2. 池化层
        self.gem = GeMPooling()

        # 3. 处理额外多尺度特征（当前 SAM 默认不用，保留接口）
        self.extra_projs = nn.ModuleList()
        total_extra_dim = 0
        for ch in extra_channels:
            self.extra_projs.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(ch, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                )
            )
            total_extra_dim += 512

        # 4. Multi-stage Classifier
        # 输入维度 = (3 * in_channels) [Avg+Max+GeM] + total_extra_dim
        cls_in_dim = (in_channels * 3) + total_extra_dim

        # 升级点：将 BatchNorm 改为 LayerNorm，适配小 Batch Size 和 ViT 特征
        # 升级点：将 ReLU 改为 GELU
        self.classifier = nn.Sequential(
            nn.Linear(cls_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, extra_feats=None):
        # 输入 x 是 SAM 的 bottleneck 特征: [B, 256, H, W]

        # 1. CBAM 注意力强化
        x_attn = self.attn(x)

        # 2. Triple Pooling Path (提取极值、均值和广义平均特征)
        avg_f = F.adaptive_avg_pool2d(x_attn, 1).view(x.size(0), -1)
        max_f = F.adaptive_max_pool2d(x_attn, 1).view(x.size(0), -1)
        gem_f = self.gem(x_attn).view(x.size(0), -1)

        # 拼接成为 [B, 768] 的丰富表征
        feat = torch.cat([avg_f, max_f, gem_f], dim=1)

        # 3. 额外特征（如果传入的话）
        if extra_feats is not None:
            extra_out = []
            for i, ef in enumerate(extra_feats):
                if i < len(self.extra_projs):
                    extra_out.append(self.extra_projs[i](ef))
            if extra_out:
                feat = torch.cat([feat] + extra_out, dim=1)

        # 4. 输出 Logits
        return self.classifier(feat)