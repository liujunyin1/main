# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import random

def rand_flip_3d(x, y=None):
    # x: [B,C,D,H,W]
    if random.random()<0.5: x = torch.flip(x, dims=[2]);  y = torch.flip(y, dims=[1]) if y is not None else None
    if random.random()<0.5: x = torch.flip(x, dims=[3]);  y = torch.flip(y, dims=[2]) if y is not None else None
    if random.random()<0.5: x = torch.flip(x, dims=[4]);  y = torch.flip(y, dims=[3]) if y is not None else None
    return x, y

def rand_gamma(x, gamma_range=(0.8,1.2)):
    g = random.uniform(*gamma_range)
    x = torch.clamp(x, min=x.min(), max=x.max())
    x = x - x.min()
    x = x/(x.max()+1e-6)
    x = x**g
    return x

def weak_aug_3d(x, y=None):
    x, y = rand_flip_3d(x, y)
    x = rand_gamma(x, (0.9,1.1))
    return x, y

def strong_aug_3d(x, y=None):
    x, y = rand_flip_3d(x, y)
    x = rand_gamma(x, (0.7,1.3))
    # 可加随机噪声/遮挡
    if random.random()<0.3:
        noise = torch.randn_like(x)*0.05
        x = x + noise
    x = torch.clamp(x, -5.0, 5.0)
    return x, y
