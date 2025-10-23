# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def _sobel_kernels_3d(device):
    # 3D Sobel (approx.) for edge/gradient
    kx = torch.tensor([[[ -1, 0, 1],
                        [ -3, 0, 3],
                        [ -1, 0, 1]],
                       [[ -3, 0, 3],
                        [ -6, 0, 6],
                        [ -3, 0, 3]],
                       [[ -1, 0, 1],
                        [ -3, 0, 3],
                        [ -1, 0, 1]]], dtype=torch.float32, device=device)
    ky = kx.transpose(0,1).contiguous()
    kz = kx.transpose(0,2).contiguous()
    kx = kx.view(1,1,3,3,3); ky = ky.view(1,1,3,3,3); kz = kz.view(1,1,3,3,3)
    return kx, ky, kz

def edge_grad_3d(x):
    """
    x: [B,1,D,H,W] (前景概率/边界图)
    return: gradient magnitude [B,1,D,H,W]
    """
    B,_,D,H,W = x.shape
    kx,ky,kz = _sobel_kernels_3d(x.device)
    gx = F.conv3d(x, kx, padding=1)
    gy = F.conv3d(x, ky, padding=1)
    gz = F.conv3d(x, kz, padding=1)
    g = torch.sqrt(gx*gx + gy*gy + gz*gz + 1e-8)
    return g

def cdcr_loss_3d(p_sam3d, p_vnet3d, w_edge=1.0, w_kl=1.0):
    """
    p_*: [B,C,D,H,W] 概率
    - 分布一致（双向KL）
    - 边界梯度一致（前景通道）
    """
    # 分布一致
    kl_sv = F.kl_div(p_sam3d.log().clamp_min(-100), p_vnet3d, reduction='batchmean')
    kl_vs = F.kl_div(p_vnet3d.log().clamp_min(-100), p_sam3d, reduction='batchmean')
    l_kl = kl_sv + kl_vs

    # 边界一致
    ps = p_sam3d[:,1:2]   # 假定二类时用前景通道；多类可对每类取 max 边界或求和
    pv = p_vnet3d[:,1:2]
    es = edge_grad_3d(ps)
    ev = edge_grad_3d(pv)
    l_edge = F.l1_loss(es, ev)

    return w_edge*l_edge + w_kl*l_kl
