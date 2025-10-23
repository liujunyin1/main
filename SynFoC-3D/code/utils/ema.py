# -*- coding: utf-8 -*-
import torch

@torch.no_grad()
def update_ema(ema_model, online_model, ema_m=0.999):
    for p_ema, p_online in zip(ema_model.parameters(), online_model.parameters()):
        p_ema.data.mul_(ema_m).add_(p_online.data, alpha=1.0-ema_m)
    # 若模型含有 buffers（如BN均值），也同步
    for b_ema, b_online in zip(ema_model.buffers(), online_model.buffers()):
        b_ema.data.copy_(b_online.data)
