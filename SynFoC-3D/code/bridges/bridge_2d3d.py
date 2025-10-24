"""Utilities that bridge a 3D volume and the 2D tiled representation used by SAM.

The previous draft only reshaped a volume by stacking slices without any padding
or channel conversion.  SAM expects three-channel 1024×1024 RGB images, while
medical volumes are typically single-channel and have spatial sizes that do not
divide 1024 exactly.  Here we reproduce the tiling strategy employed in SFR: we
resize slices to a regular grid, place them into a centred canvas, and keep all
metadata needed for an invertible mapping back to 3D logits.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def _choose_grid(depth: int, height: int, width: int, canvas: int, tile_cols: int | None) -> tuple[int, int]:
    """Choose a (rows, cols) grid that can host ``depth`` slices on a canvas."""

    if tile_cols is not None and tile_cols > 0:
        cols = tile_cols
        rows = int(math.ceil(depth / cols))
    else:
        rows = max(1, canvas // max(1, height))
        cols = max(1, canvas // max(1, width))
        if rows * cols == 0:
            rows = cols = 1
        while rows * cols < depth:
            if rows <= cols:
                rows += 1
            else:
                cols += 1
    return max(rows, 1), max(cols, 1)


@torch.no_grad()
def slice_volume_to_tiles(
    vol3d: torch.Tensor,
    sam_input_size: int = 1024,
    tile_cols: int | None = None,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, int]]:
    """Convert ``[B, C, D, H, W]`` volumes into SAM-compatible 2D tiles.

    Returns a batch of three-channel images ``[B, 3, sam_input_size, sam_input_size]``
    together with metadata required for ``tiles_to_volume``.
    """

    if vol3d.dim() != 5:
        raise ValueError(f"vol3d shape expected [B,C,D,H,W], got {tuple(vol3d.shape)}")

    B, C, D, H, W = vol3d.shape
    rows, cols = _choose_grid(D, H, W, sam_input_size, tile_cols)
    tile_h = max(1, sam_input_size // rows)
    tile_w = max(1, sam_input_size // cols)
    grid_h = tile_h * rows
    grid_w = tile_w * cols
    pad_top = max((sam_input_size - grid_h) // 2, 0)
    pad_bottom = sam_input_size - grid_h - pad_top
    pad_left = max((sam_input_size - grid_w) // 2, 0)
    pad_right = sam_input_size - grid_w - pad_left

    # Collapse modalities to a single channel for SAM and keep intensity statistics.
    if C == 1:
        gray = vol3d
    else:
        gray = vol3d.mean(dim=1, keepdim=True)

    resized = F.interpolate(
        gray,
        size=(D, tile_h, tile_w),
        mode="trilinear",
        align_corners=False,
    )

    canvas = vol3d.new_full((B, 3, sam_input_size, sam_input_size), pad_value)
    total_tiles = rows * cols
    for idx in range(total_tiles):
        if idx >= D:
            break
        r, c = divmod(idx, cols)
        tile = resized[:, :, idx]
        tile_rgb = tile.repeat(1, 3, 1, 1)
        top = pad_top + r * tile_h
        left = pad_left + c * tile_w
        canvas[:, :, top : top + tile_h, left : left + tile_w] = tile_rgb

    meta = {
        "depth": D,
        "orig_h": H,
        "orig_w": W,
        "rows": rows,
        "cols": cols,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "pad_bottom": pad_bottom,
        "pad_right": pad_right,
        "sam_size": sam_input_size,
    }
    return canvas, meta


@torch.no_grad()
def tiles_to_volume(big_pred: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
    """Invert :func:`slice_volume_to_tiles` for SAM logits."""

    if big_pred.dim() != 4:
        raise ValueError(f"big_pred shape expected [B,C,H,W], got {tuple(big_pred.shape)}")

    B, C, _HH, _WW = big_pred.shape
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    tile_h = int(meta["tile_h"])
    tile_w = int(meta["tile_w"])
    depth = int(meta["depth"])
    pad_top = int(meta["pad_top"])
    pad_left = int(meta["pad_left"])
    grid_h = tile_h * rows
    grid_w = tile_w * cols

    crop = big_pred[:, :, pad_top : pad_top + grid_h, pad_left : pad_left + grid_w]
    vol = crop.new_zeros((B, C, rows * cols, tile_h, tile_w))

    idx = 0
    for r in range(rows):
        for c in range(cols):
            tile = crop[:, :, r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w]
            vol[:, :, idx] = tile
            idx += 1

    vol = vol[:, :, :depth]
    orig_h = int(meta["orig_h"])
    orig_w = int(meta["orig_w"])
    if tile_h != orig_h or tile_w != orig_w:
        vol = F.interpolate(vol, size=(depth, orig_h, orig_w), mode="trilinear", align_corners=False)

    return vol