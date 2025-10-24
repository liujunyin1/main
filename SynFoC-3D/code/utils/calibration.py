# utils/calibration.py
import torch

def temperature_scale(logits, T):
    return logits / T

def percentile_calibrate(prob, q=0.95):
    # 把每次 batch 的概率按分位数归一，缓解 2D/3D 校准差异
    th = torch.quantile(prob.detach(), q)
    return torch.clamp(prob / (th + 1e-6), 0, 1)
