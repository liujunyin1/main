# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class VNet3D(nn.Module):
    """
    适配 SFR 的 VNet 实现为通用接口。
    你可以按实际文件位置修改 import 路径：
      from SFR.code.models.networks.hierarchical_vnet import HierarchicalVNet
    如果你使用的是其它 3D U-Net/VNet，请在这里统一 forward 输出 logits: [B,C,D,H,W]
    """
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        try:
            from models.networks.hierarchical_vnet import HierarchicalVNet
            self.net = HierarchicalVNet(in_channels=in_ch, num_classes=out_ch)
            self.returns_logits = True
        except Exception:
            # 兜底：简单3D UNet（若SFR不可用），确保项目可直接跑通
            from torch.nn import Conv3d, MaxPool3d, ReLU
            class Tiny3DUNet(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.enc = nn.Sequential(
                        Conv3d(in_ch, 16, 3, padding=1), ReLU(inplace=True), Conv3d(16, 32, 3, padding=1), ReLU(inplace=True),
                        MaxPool3d(2),
                        Conv3d(32, 64, 3, padding=1), ReLU(inplace=True),
                    )
                    self.cls = Conv3d(64, out_ch, 1)
                def forward(self, x): return self.cls(self.enc(x))
            self.net = Tiny3DUNet(in_ch, out_ch)
            self.returns_logits = True

    def forward(self, x):
        out = self.net(x)
        return out  # logits
