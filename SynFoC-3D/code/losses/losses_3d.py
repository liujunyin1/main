# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def dice_loss_3d(prob, target, eps=1e-6):
    # prob: [B,C,D,H,W], target: [B,D,H,W]
    num_classes = prob.shape[1]
    target_1hot = F.one_hot(target, num_classes).permute(0,4,1,2,3).float()
    inter = (prob * target_1hot).sum(dim=(0,2,3,4))
    union = (prob + target_1hot).sum(dim=(0,2,3,4))
    dice = (2*inter + eps)/(union + eps)
    return 1 - dice.mean()

def ce_dice_3d(logits, target):
    ce = F.cross_entropy(logits, target)
    prob = torch.softmax(logits, dim=1)
    dc = dice_loss_3d(prob, target)
    return ce + dc
