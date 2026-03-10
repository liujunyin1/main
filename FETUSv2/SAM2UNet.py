import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from EnhancedClsHead import EnhancedClsHead
from model.overlock_lite_backbone import (
    LayerNorm2d as OverLoCKLayerNorm2d,
    load_pretrained_weights,
    overlock_lite_b,
    overlock_lite_s,
    overlock_lite_t,
    overlock_lite_xt,
)

# ==================== 基础模块 (保持你的原样) ====================


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.up(x))


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim), nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        dilation: Union[int, tuple[int, int]] = 1,
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class OverLoCKSkipGenerator(nn.Module):
    _VARIANTS = {
        "xt": [56, 112, 256, 336],
        "t": [64, 128, 256, 512],
        "s": [64, 128, 320, 512],
        "b": [80, 160, 384, 576],
    }

    def __init__(
        self,
        in_chans=3,
        variant: str = "t",
        pretrained: bool = False,
        weights_path: Optional[str] = None,
    ):
        super().__init__()

        if in_chans != 3:
            raise ValueError("OverLoCKSkipGenerator expects 3-channel input.")
        if variant not in self._VARIANTS:
            raise ValueError(
                f"Unknown OverLoCK variant: {variant}. Choose from {sorted(self._VARIANTS.keys())}"
            )

        self.variant = variant
        embed_dim: List[int] = self._VARIANTS[variant]
        self.embed_dim = embed_dim

        lite_builders = {
            "xt": overlock_lite_xt,
            "t": overlock_lite_t,
            "s": overlock_lite_s,
            "b": overlock_lite_b,
        }
        self.backbone = lite_builders[variant]()
        norm_layer = OverLoCKLayerNorm2d

        if weights_path:
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"OverLoCK weights not found: {weights_path}\n"
                    f"Please provide a valid path or set weights_path=None for random init."
                )
            load_pretrained_weights(self.backbone, weights_path)
        elif pretrained:
            print(
                "Warning: pretrained=True not supported for aux_encoder. Use weights_path instead."
            )

        self.extra_norm = nn.ModuleList(
            [
                norm_layer(embed_dim[0]),
                norm_layer(embed_dim[1]),
                norm_layer(embed_dim[2]),
                norm_layer(embed_dim[3]),
            ]
        )

    def _forward_lite(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []

        x = self.backbone.patch_embed1(x)
        for blk in self.backbone.blocks1:
            x = blk(x)
        x0 = self.extra_norm[0](x)
        feats.append(x0)

        x = self.backbone.patch_embed2(x)
        for blk in self.backbone.blocks2:
            x = blk(x)
        x1 = self.extra_norm[1](x)
        feats.append(x1)

        x = self.backbone.patch_embed3(x)
        for blk in self.backbone.blocks3:
            x = blk(x)
        x2 = self.extra_norm[2](x)
        feats.append(x2)

        x = self.backbone.patch_embed4(x)
        for blk in self.backbone.blocks4:
            x = blk(x)
        x3 = self.extra_norm[3](x)
        feats.append(x3)

        return feats

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self._forward_lite(x)


# ==================== 统一网络架构 ====================


class SAM2UNet(nn.Module):
    def __init__(
        self,
        checkpoint_path=None,
        overlock_weights_path: Optional[str] = None,
        device: str | torch.device = "cuda",
        in_chns=1,
        seg_class_num=15,
        cls_class_num=7,
        img_size=352,
        channel_mode="replicate",
        model_size="tiny",
        seg_mode="legacy_concat",
        use_overlock: bool = True,
    ):
        super().__init__()

        self.seg_class_num = seg_class_num
        self.cls_class_num = cls_class_num
        self.img_size = img_size
        self.channel_mode = channel_mode
        self.seg_mode = seg_mode
        self.use_overlock = use_overlock
        sam_device = str(device)

        # 1. 骨干网络初始化
        model_configs = {
            "large": ("sam2_hiera_l.yaml", [144, 288, 576, 1152]),
            "base_plus": ("sam2_hiera_b+.yaml", [112, 224, 448, 896]),
            "small": ("sam2_hiera_s.yaml", [96, 192, 384, 768]),
            "tiny": ("sam2_hiera_t.yaml", [96, 192, 384, 768]),
        }
        model_cfg, self.feat_dims = model_configs[model_size]

        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path, device=sam_device)
        else:
            model = build_sam2(model_cfg, device=sam_device)

        if hasattr(model, "memory_encoder"):
            del model.memory_encoder
        if hasattr(model, "memory_attention"):
            del model.memory_attention

        self.encoder = model.image_encoder.trunk
        self.image_encoder_neck = model.image_encoder.neck
        self.image_encoder_scalp = model.image_encoder.scalp
        self.sam_prompt_encoder = model.sam_prompt_encoder
        self.sam_mask_decoder = model.sam_mask_decoder
        self._sam_forward_sam_heads = model._forward_sam_heads
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder_neck.parameters():
            param.requires_grad = False
        for param in self.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam_mask_decoder.parameters():
            param.requires_grad = seg_mode == "hybrid_x4"

        # Adapter
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(Adapter(block))
        self.encoder.blocks = nn.Sequential(*blocks)

        # 2. 输入通道适配
        if channel_mode == "conv":
            self.channel_adapter = nn.Conv2d(1, 3, 1)
        else:
            self.channel_adapter = None

        # 3. 分割分支
        if self.seg_mode not in {"legacy_concat", "hybrid_x4"}:
            raise ValueError(f"Unknown seg_mode: {self.seg_mode}")

        self.rfb1 = RFB_modified(self.feat_dims[0], 64)
        self.rfb2 = RFB_modified(self.feat_dims[1], 64)
        self.rfb3 = RFB_modified(self.feat_dims[2], 64)
        self.rfb4 = RFB_modified(self.feat_dims[3], 64)

        self.cnn_skip_dims = [56, 112, 256, 336]
        if self.use_overlock:
            self.cnn_skips = OverLoCKSkipGenerator(
                in_chans=3,
                variant="xt",
                weights_path=overlock_weights_path,
            )
        else:
            self.cnn_skips = None

        self.hybrid_x4_adapter = nn.Sequential(
            nn.Conv2d(
                256, self.feat_dims[3], kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.feat_dims[3]),
            nn.ReLU(inplace=True),
        )

        self.up1 = Up(720, 320)
        self.up2 = Up(496, 176)
        self.up3 = Up(296, 120)
        self.seg_head = nn.Conv2d(120, 120, 1)

        self.seg_decoder = nn.Sequential(
            nn.Conv2d(120, 120, 3, padding=1),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
            nn.Conv2d(120, seg_class_num, 1),
        )

        # 4. 增强分类分支 (连接编码器最深层)
        # feat_dims[3] 是 x4 的通道数(例如 768)，feat_dims[2] 是 x3(例如 384)
        self.cls_decoder = EnhancedClsHead(
            in_channels=self.feat_dims[3],  # 对应 x4 的通道数
            hidden_dim=256,
            num_classes=cls_class_num,
        )

        # 5. 半监督扰动
        self.dropout = nn.Dropout2d(0.5)

    def _adapt_input(self, x):
        if x.shape[1] == 1:
            if self.channel_adapter:
                x = self.channel_adapter(x)
            else:
                x = x.repeat(1, 3, 1, 1)

        orig_size = x.shape[2:]
        if orig_size != (self.img_size, self.img_size):
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return x, orig_size

    def _forward_task(self, x):
        """执行编码器并分别走分割和分类头"""
        x1, x2, x3, x4 = self.encoder(x)

        # ========== 分类路径 ==========
        # 喂给分类器的是最高级语义特征 x4，并将 x3 作为多尺度补充
        cls_out = self.cls_decoder(x4)

        if self.seg_mode == "hybrid_x4":
            x4 = self._compute_hybrid_x4(x1, x2, x3, x4)

        r1, r2, r3, r4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        c1, c2, c3, c4 = self._get_cnn_skips(x1, x2, x3, x4, x)
        f1 = torch.cat([r1, c1], dim=1)
        f2 = torch.cat([r2, c2], dim=1)
        f3 = torch.cat([r3, c3], dim=1)
        f4 = torch.cat([r4, c4], dim=1)

        up_x = self.up1(f4, f3)
        up_x = self.up2(up_x, f2)
        up_x = self.up3(up_x, f1)

        seg_feat = self.seg_head(up_x)
        seg_out = self.seg_decoder(seg_feat)

        return seg_out, cls_out

    def forward(self, x, return_fp=False):
        x_proc, orig_size = self._adapt_input(x)

        if return_fp:
            x_fp = self.dropout(x_proc)
            x_cat = torch.cat([x_proc, x_fp], dim=0)

            seg_cat, cls_cat = self._forward_task(x_cat)
            seg_cat = F.interpolate(
                seg_cat, size=orig_size, mode="bilinear", align_corners=False
            )

            seg, seg_fp = seg_cat.chunk(2)
            cls, cls_fp = cls_cat.chunk(2)
            return (seg, seg_fp), (cls, cls_fp)
        else:
            seg_out, cls_out = self._forward_task(x_proc)
            seg_out = F.interpolate(
                seg_out, size=orig_size, mode="bilinear", align_corners=False
            )
            return seg_out, cls_out

    def get_trainable_params(self):
        adapter_params = []
        for name, p in self.encoder.blocks.named_parameters():
            if "prompt_learn" in name:
                adapter_params.append(p)

        head_params = []
        for name, p in self.named_parameters():
            if (
                not name.startswith("encoder.")
                and not name.startswith("image_encoder_neck.")
                and not name.startswith("sam_prompt_encoder.")
                and "prompt_learn" not in name
            ):
                head_params.append(p)

        return {"adapter": adapter_params, "head": head_params}

    def _get_cnn_skips(self, x1, x2, x3, x4, x):
        if self.cnn_skips is not None:
            return self.cnn_skips(x)

        zeros = []
        for feat, channels in zip((x1, x2, x3, x4), self.cnn_skip_dims):
            zeros.append(
                torch.zeros(
                    feat.shape[0],
                    channels,
                    feat.shape[2],
                    feat.shape[3],
                    device=feat.device,
                    dtype=feat.dtype,
                )
            )
        return tuple(zeros)

    def _compute_hybrid_x4(self, x1, x2, x3, x4):
        neck_features, _ = self.image_encoder_neck([x1, x2, x3, x4])
        if self.image_encoder_scalp > 0:
            neck_features = neck_features[: -self.image_encoder_scalp]
        neck_features[0] = self.sam_mask_decoder.conv_s0(neck_features[0])
        neck_features[1] = self.sam_mask_decoder.conv_s1(neck_features[1])

        image_embeddings = neck_features[-1]
        high_res_features = neck_features[:-1]
        sam_outputs = self._sam_forward_sam_heads(
            backbone_features=image_embeddings,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=False,
            return_refined_features=True,
        )
        refined_bottleneck = sam_outputs[-1]
        hybrid_delta = self.hybrid_x4_adapter(refined_bottleneck)
        return x4 + hybrid_delta
