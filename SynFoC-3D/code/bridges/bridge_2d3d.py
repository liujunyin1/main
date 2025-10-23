# -*- coding: utf-8 -*-
import torch, math

@torch.no_grad()
def slice_volume_to_tiles(vol3d: torch.Tensor, tile_cols=None):
    """
    vol3d: [B, C, D, H, W]
    返回:
      big_img: [B, C, rows*H, cols*W]
      meta: dict 记录回填坐标
    """
    assert vol3d.dim()==5, f"vol3d shape expected [B,C,D,H,W], got {vol3d.shape}"
    B,C,D,H,W = vol3d.shape
    cols = tile_cols or int(math.ceil(math.sqrt(D)))
    rows = int(math.ceil(D/cols))
    big = torch.zeros(B, C, rows*H, cols*W, device=vol3d.device, dtype=vol3d.dtype)
    idx_map = []
    for k in range(D):
        r, c = k // cols, k % cols
        big[:, :, r*H:(r+1)*H, c*W:(c+1)*W] = vol3d[:, :, k]
        idx_map.append((r,c,k))
    meta = {'D':D,'H':H,'W':W,'rows':rows,'cols':cols,'idx_map':idx_map}
    return big, meta

@torch.no_grad()
def tiles_to_volume(big_pred: torch.Tensor, meta: dict):
    """
    big_pred: [B, C, rows*H, cols*W]
    返回: vol3d: [B, C, D, H, W]
    """
    B,C,HH,WW = big_pred.shape
    D,H,W,rows,cols,idx_map = meta['D'], meta['H'], meta['W'], meta['rows'], meta['cols'], meta['idx_map']
    vol = torch.zeros(B, C, D, H, W, device=big_pred.device, dtype=big_pred.dtype)
    for (r,c,k) in idx_map:
        vol[:, :, k] = big_pred[:, :, r*H:(r+1)*H, c*W:(c+1)*W]
    return vol
