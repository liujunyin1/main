import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from EnhancedClsHead import EnhancedClsHead

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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
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
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
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






# ==================== 统一网络架构 ====================

class SAM2UNet(nn.Module):
    def __init__(
            self,
            checkpoint_path=None,
            in_chns=1,
            seg_class_num=15,
            cls_class_num=7,
            img_size=352,
            channel_mode="replicate",
            model_size="tiny"
    ):
        super().__init__()

        self.seg_class_num = seg_class_num
        self.cls_class_num = cls_class_num
        self.img_size = img_size
        self.channel_mode = channel_mode

        # 1. 骨干网络初始化
        model_configs = {
            "large": ("sam2_hiera_l.yaml", [144, 288, 576, 1152]),
            "base_plus": ("sam2_hiera_b+.yaml", [112, 224, 448, 896]),
            "small": ("sam2_hiera_s.yaml", [96, 192, 384, 768]),
            "tiny": ("sam2_hiera_t.yaml", [96, 192, 384, 768]),
        }
        model_cfg, self.feat_dims = model_configs[model_size]

        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)

        # 删除不需要的模块释放显存
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck

        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False

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

        # 3. 分割分支 (RFB + Decoder)
        self.rfb1 = RFB_modified(self.feat_dims[0], 64)
        self.rfb2 = RFB_modified(self.feat_dims[1], 64)
        self.rfb3 = RFB_modified(self.feat_dims[2], 64)
        self.rfb4 = RFB_modified(self.feat_dims[3], 64)

        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        self.seg_head = nn.Conv2d(64, 64, 1)

        self.seg_decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, seg_class_num, 1)
        )

        # 4. 增强分类分支 (连接编码器最深层)
        # feat_dims[3] 是 x4 的通道数(例如 768)，feat_dims[2] 是 x3(例如 384)
        self.cls_decoder = EnhancedClsHead(
            in_channels=self.feat_dims[3],  # 对应 x4 的通道数
            hidden_dim=256,
            num_classes=cls_class_num
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
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return x, orig_size

    def _forward_task(self, x):
        """执行编码器并分别走分割和分类头"""
        x1, x2, x3, x4 = self.encoder(x)

        # ========== 分类路径 ==========
        # 喂给分类器的是最高级语义特征 x4，并将 x3 作为多尺度补充
        cls_out = self.cls_decoder(x4)
        # ========== 分割路径 ==========
        r1, r2, r3, r4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)

        up_x = self.up1(r4, r3)
        up_x = self.up2(up_x, r2)
        up_x = self.up3(up_x, r1)

        seg_feat = self.seg_head(up_x)
        seg_out = self.seg_decoder(seg_feat)

        return seg_out, cls_out

    def forward(self, x, return_fp=False):
        x_proc, orig_size = self._adapt_input(x)

        if return_fp:
            x_fp = self.dropout(x_proc)
            x_cat = torch.cat([x_proc, x_fp], dim=0)

            seg_cat, cls_cat = self._forward_task(x_cat)

            # 恢复原始分辨率
            seg_cat = F.interpolate(seg_cat, size=orig_size, mode="bilinear", align_corners=False)

            seg, seg_fp = seg_cat.chunk(2)
            cls, cls_fp = cls_cat.chunk(2)
            return (seg, seg_fp), (cls, cls_fp)
        else:
            seg_out, cls_out = self._forward_task(x_proc)
            seg_out = F.interpolate(seg_out, size=orig_size, mode="bilinear", align_corners=False)
            return seg_out, cls_out

    def get_trainable_params(self):
        adapter_params = []
        for name, p in self.encoder.blocks.named_parameters():
            if "prompt_learn" in name:
                adapter_params.append(p)

        head_params = []
        for name, p in self.named_parameters():
            if "encoder" not in name:  # 只要不是冻结的 encoder，都加入训练
                head_params.append(p)

        return {
            "adapter": adapter_params,
            "head": head_params
        }
