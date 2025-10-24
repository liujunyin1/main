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

def _edge_grad_single(x):
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
    sam_log = p_sam3d.clamp_min(1e-8).log()
    vnet_log = p_vnet3d.clamp_min(1e-8).log()

    kl_sv = F.kl_div(sam_log, p_vnet3d, reduction='none')
    kl_vs = F.kl_div(vnet_log, p_sam3d, reduction='none')
    l_kl = (kl_sv + kl_vs).mean()

    # 边界一致
    grads_s = []
    grads_v = []
    for c in range(p_sam3d.shape[1]):
        grads_s.append(_edge_grad_single(p_sam3d[:, c:c+1]))
        grads_v.append(_edge_grad_single(p_vnet3d[:, c:c+1]))
    es = torch.cat(grads_s, dim=1)
    ev = torch.cat(grads_v, dim=1)
    l_edge = F.l1_loss(es, ev)

    return w_edge*l_edge + w_kl*l_kl
