# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

@torch.no_grad()
def dice_per_class(prob, target, num_classes: int, eps=1e-6):
    # prob: [B,C,D,H,W]; target: [B,D,H,W]
    pred = prob.argmax(1)
    dices = []
    for c in range(num_classes):
        pc = (pred==c).float()
        tc = (target==c).float()
        inter = (pc*tc).sum()
        denom = pc.sum() + tc.sum()
        d = (2*inter + eps)/(denom + eps)
        dices.append(d)
    return torch.stack(dices)  # [C]
