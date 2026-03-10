"""
OverLoCK aux_encoder backbone (lite implementation)

This module provides a NATTEN-free implementation of the aux_encoder
backbone. It imports only the components needed for the lite backbone
(stem, blocks1-4, patch_embeds) and skips the NATTEN-dependent sub_blocks.

Components extracted from OverLoCK:
- stem (patch_embed1)
- patch_embed2, patch_embed3, patch_embed4
- blocks1, blocks2, blocks3, blocks4 (RepConvBlock)
- LayerNorm2d (for extra_norm)

NOT included (require NATTEN):
- sub_blocks3, sub_blocks4 (DynamicConvBlock)
- patch_embedx, high_level_proj
"""

import importlib
import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.drop import DropPath


def _pair(value: Union[int, tuple[int, int]]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def get_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Optional[Union[int, tuple[int, int]]],
    dilation: Union[int, tuple[int, int]],
    groups: int,
    bias: bool,
    attempt_use_lk_impl: bool = True,
):
    """Get Conv2d with optional large kernel implementation."""
    kernel_size = _pair(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = _pair(padding)
    need_large_impl = (
        kernel_size[0] == kernel_size[1]
        and kernel_size[0] > 5
        and padding == (kernel_size[0] // 2, kernel_size[1] // 2)
    )

    if attempt_use_lk_impl and need_large_impl:
        try:
            depthwise_module = importlib.import_module("depthwise_conv2d_implicit_gemm")
            DepthWiseConv2dImplicitGEMM = getattr(
                depthwise_module, "DepthWiseConv2dImplicitGEMM", None
            )

            if (
                DepthWiseConv2dImplicitGEMM is not None
                and need_large_impl
                and in_channels == out_channels
                and out_channels == groups
                and stride == 1
                and dilation == 1
            ):
                return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
        except ImportError:
            pass

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def convert_dilated_to_nondilated(kernel, dilate_rate):
    """Convert dilated kernel to non-dilated."""
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(
                kernel[:, i : i + 1, :, :], identity_kernel, stride=dilate_rate
            )
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    """Merge dilated kernel into large kernel."""
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


def fuse_bn(conv, bn):
    """Fuse BatchNorm into Conv."""
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return (
        conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1),
        bn.bias + (conv_bias - bn.running_mean) * bn.weight / std,
    )


def stem(in_chans=3, embed_dim=96):
    """Stem network for initial feature extraction."""
    return nn.Sequential(
        nn.Conv2d(
            in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False
        ),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(
            embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False
        ),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
    )


def downsample(in_dim, out_dim):
    """Downsample block."""
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
    )


class LayerNorm2d(nn.LayerNorm):
    """2D Layer Normalization."""

    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, input):
        x = rearrange(input, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x.contiguous()


class SEModule(nn.Module):
    """Squeeze-and-Excitation module."""

    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        return x * self.proj(x)


class LayerScale(nn.Module):
    """Layer Scale module."""

    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(dim, 1, 1, 1) * init_value, requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])


class GRN(nn.Module):
    """Global Response Normalization layer (from ConvNeXt V2)."""

    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class ResDWConv(nn.Conv2d):
    """Depthwise convolution with residual connection."""

    def __init__(self, dim, kernel_size=3):
        super().__init__(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )

    def forward(self, input):
        return input + super().forward(input)


class DilatedReparamBlock(nn.Module):
    """Dilated Reparam Block from UniRepLKNet."""

    def __init__(
        self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True
    ):
        super().__init__()
        self.lk_origin = get_conv2d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=channels,
            bias=deploy,
            attempt_use_lk_impl=attempt_use_lk_impl,
        )
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}")

        self.deploy = deploy
        if not deploy:
            self.origin_bn = nn.BatchNorm2d(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = get_conv2d(
                    channels,
                    channels,
                    k,
                    stride=1,
                    padding=(r * (k - 1) + 1) // 2,
                    dilation=r,
                    groups=channels,
                    bias=False,
                    attempt_use_lk_impl=attempt_use_lk_impl,
                )
                bn = nn.BatchNorm2d(channels)
                self.add_module(f"dil_conv_k{k}_{r}", conv)
                self.add_module(f"dil_bn_k{k}_{r}", bn)

    def forward(self, x):
        if self.deploy:
            return self.lk_origin(x)

        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f"dil_conv_k{k}_{r}")
            bn = getattr(self, f"dil_bn_k{k}_{r}")
            out = out + bn(conv(x))
        return out


class RepConvBlock(nn.Module):
    """Reparameterizable Convolution Block (local feature extraction)."""

    def __init__(
        self,
        dim=64,
        kernel_size=7,
        mlp_ratio=4,
        ls_init_value=None,
        res_scale=False,
        drop_path: float = 0,
        norm_layer=LayerNorm2d,
        use_gemm=False,
        deploy=False,
        use_checkpoint=False,
    ):
        super().__init__()

        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint
        mlp_dim = int(dim * mlp_ratio)

        self.dwconv = ResDWConv(dim, kernel_size=3)

        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(
                dim,
                kernel_size=kernel_size,
                deploy=deploy,
                use_sync_bn=False,
                attempt_use_lk_impl=use_gemm,
            ),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )

        self.ls = (
            LayerScale(dim, init_value=ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

    def forward_features(self, x):
        x = self.dwconv(x)
        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))
        return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            from torch.utils.checkpoint import checkpoint

            x = checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)
        return x


class OverLoCKLiteBackbone(nn.Module):
    """
    OverLoCK aux_encoder backbone (no NATTEN dependency).

    This is a simplified aux_encoder that only includes:
    - stem (patch_embed1)
    - patch_embed2, patch_embed3, patch_embed4
    - blocks1, blocks2, blocks3, blocks4 (RepConvBlock)

    Excluded (require NATTEN):
    - sub_blocks3, sub_blocks4 (DynamicConvBlock)
    - patch_embedx, high_level_proj
    """

    def __init__(
        self,
        depth: List[int] = [2, 2, 2, 2],
        in_chans: int = 3,
        embed_dim: List[int] = [64, 128, 256, 512],
        kernel_size: List[int] = [7, 7, 7, 7],
        mlp_ratio: List[int] = [4, 4, 4, 4],
        ls_init_value: List[Optional[float]] = [None, None, 1, 1],
        res_scale: bool = True,
        deploy: bool = False,
        use_gemm: bool = True,
        drop_path_rate: float = 0,
        norm_layer=LayerNorm2d,
        use_checkpoint: List[int] = [0, 0, 0, 0],
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.patch_embed1 = stem(in_chans, embed_dim[0])
        self.patch_embed2 = downsample(embed_dim[0], embed_dim[1])
        self.patch_embed3 = downsample(embed_dim[1], embed_dim[2])
        self.patch_embed4 = downsample(embed_dim[2], embed_dim[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        self.blocks3 = nn.ModuleList()
        self.blocks4 = nn.ModuleList()

        for i in range(depth[0]):
            self.blocks1.append(
                RepConvBlock(
                    dim=embed_dim[0],
                    kernel_size=kernel_size[0],
                    mlp_ratio=mlp_ratio[0],
                    ls_init_value=ls_init_value[0],
                    res_scale=res_scale,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[0]),
                )
            )

        for i in range(depth[1]):
            self.blocks2.append(
                RepConvBlock(
                    dim=embed_dim[1],
                    kernel_size=kernel_size[1],
                    mlp_ratio=mlp_ratio[1],
                    ls_init_value=ls_init_value[1],
                    res_scale=res_scale,
                    drop_path=dpr[i + depth[0]],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[1]),
                )
            )

        for i in range(depth[2]):
            self.blocks3.append(
                RepConvBlock(
                    dim=embed_dim[2],
                    kernel_size=kernel_size[2],
                    mlp_ratio=mlp_ratio[2],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depth[:2])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[2]),
                )
            )

        for i in range(depth[3]):
            self.blocks4.append(
                RepConvBlock(
                    dim=embed_dim[3],
                    kernel_size=kernel_size[3],
                    mlp_ratio=mlp_ratio[3],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depth[:3])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[3]),
                )
            )

    def forward(self, x):
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
        out1 = x

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        out2 = x

        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        out3 = x

        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        out4 = x

        return out1, out2, out3, out4


def overlock_lite_xt(**kwargs) -> OverLoCKLiteBackbone:
    return OverLoCKLiteBackbone(
        depth=[2, 2, 3, 2],
        embed_dim=[56, 112, 256, 336],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        ls_init_value=[None, None, 1, 1],
        res_scale=True,
        **kwargs,
    )


def overlock_lite_t(**kwargs) -> OverLoCKLiteBackbone:
    return OverLoCKLiteBackbone(
        depth=[4, 4, 6, 2],
        embed_dim=[64, 128, 256, 512],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        ls_init_value=[None, None, 1, 1],
        res_scale=True,
        **kwargs,
    )


def overlock_lite_s(**kwargs) -> OverLoCKLiteBackbone:
    return OverLoCKLiteBackbone(
        depth=[6, 6, 8, 3],
        embed_dim=[64, 128, 320, 512],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        ls_init_value=[None, None, 1, 1],
        res_scale=True,
        **kwargs,
    )


def overlock_lite_b(**kwargs) -> OverLoCKLiteBackbone:
    return OverLoCKLiteBackbone(
        depth=[8, 8, 10, 4],
        embed_dim=[80, 160, 384, 576],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        ls_init_value=[None, None, 1, 1],
        res_scale=True,
        **kwargs,
    )


def load_pretrained_weights(
    model: OverLoCKLiteBackbone, weights_path: str, strict: bool = False
):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    filtered_state = {}
    skipped_keys = []
    for k, v in state.items():
        if any(skip in k for skip in ["sub_blocks", "patch_embedx", "high_level_proj"]):
            skipped_keys.append(k)
            continue
        filtered_state[k] = v

    msg = model.load_state_dict(filtered_state, strict=strict)
    print(f"Loaded pretrained weights from {weights_path}")
    print(f"  Matched keys: {len(filtered_state)}")
    print(f"  Skipped keys (NATTEN-dependent): {len(skipped_keys)}")
    if msg.missing_keys:
        print(f"  Missing keys: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

    return msg
