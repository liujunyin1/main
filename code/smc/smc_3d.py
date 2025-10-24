# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

@torch.no_grad()
def smc_fuse_3d(logits_vnet, logits_sam3d, T_v=1.0, T_s=1.0, mode='voxel'):
    """
    输入: 两个教师在同一 Uw 上的 3D logits
    输出: 融合后的概率 p_en 以及 α 权重
    """
    pv = F.softmax(logits_vnet / T_v, dim=1)  # [B,C,D,H,W]
    ps = F.softmax(logits_sam3d / T_s, dim=1)
    # 自置信
    cv = pv.max(dim=1).values
    cs = ps.max(dim=1).values
    # 互置信（hard 一致）
    yv = pv.argmax(dim=1); ys = ps.argmax(dim=1)
    agree = (yv == ys).float()
    # α
    wv, ws = cv*agree, cs*agree + 1e-6
    alpha = wv / (wv + ws)
    # 融合
    p_en = alpha.unsqueeze(1)*pv + (1.0-alpha.unsqueeze(1))*ps
    return p_en, alpha
