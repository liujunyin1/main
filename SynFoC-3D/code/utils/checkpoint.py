# -*- coding: utf-8 -*-
import torch

def save_ckpt(path, **states):
    torch.save(states, path)

def load_ckpt(path, map_location=None):
    return torch.load(path, map_location=map_location)
