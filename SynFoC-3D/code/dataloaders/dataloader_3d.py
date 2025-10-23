# -*- coding: utf-8 -*-
import os, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import nibabel as nib

from .augs_3d import weak_aug_3d, strong_aug_3d

def _read_npy(path): return np.load(path)
def _read_nii(path): return nib.load(path).get_fdata()

def _read_vol(path):
    if path.endswith(".npy"):
        v = _read_npy(path)
    else:
        v = _read_nii(path)  # 支持 .nii / .nii.gz
    if v.ndim==3:
        v = v[None,...]  # [C=1,D,H,W]
    elif v.ndim==4 and v.shape[0] not in [1,3]:
        # 若是 [D,H,W,C] 则转置
        v = np.transpose(v, (3,0,1,2))
    return v

def _center_crop_or_pad(vol: np.ndarray, size: Tuple[int,int,int]):
    # vol: [C,D,H,W] ; size: (D,H,W)
    C,D,H,W = vol.shape
    d,h,w = size
    out = np.zeros((C,d,h,w), dtype=vol.dtype)
    sd, sh, sw = max(0, (d-D)//2), max(0, (h-H)//2), max(0, (w-W)//2)
    td, th, tw = max(0, (D-d)//2), max(0, (H-h)//2), max(0, (W-w)//2)
    ds = min(d, D); hs = min(h, H); ws = min(w, W)
    out[:, sd:sd+ds, sh:sh+hs, sw:sw+ws] = vol[:, td:td+ds, th:th+hs, tw:tw+ws]
    return out

class LabeledSet3D(Dataset):
    def __init__(self, index_file: str, patch_size: Tuple[int,int,int], image_suffix: str):
        self.items = []  # (image_path, label_path)
        with open(index_file, 'r') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                img_path, lab_path = line.split(",")
                self.items.append((img_path.strip(), lab_path.strip()))
        self.patch = patch_size
        self.suffix = image_suffix

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img_path, lab_path = self.items[i]
        img = _read_vol(img_path).astype(np.float32)
        lab = _read_vol(lab_path).astype(np.int64)
        # 归一化
        img = (img - np.mean(img))/ (np.std(img)+1e-6)
        img = _center_crop_or_pad(img, self.patch)
        lab = _center_crop_or_pad(lab, self.patch)
        x = torch.from_numpy(img)            # [C,D,H,W]
        y = torch.from_numpy(lab[0])         # [D,H,W] (假定单通道标签)
        x = x.unsqueeze(0)                   # [B=1,C,D,H,W]
        y = y.unsqueeze(0)
        xw, yw = weak_aug_3d(x, y)           # 有标-弱增
        return xw.squeeze(0).float(), yw.squeeze(0).long()

class UnlabeledSet3D(Dataset):
    def __init__(self, index_file: str, patch_size: Tuple[int,int,int], image_suffix: str):
        self.items = []
        with open(index_file, 'r') as f:
            for line in f:
                p=line.strip()
                if p: self.items.append(p)
        self.patch = patch_size
        self.suffix = image_suffix

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img_path = self.items[i]
        img = _read_vol(img_path).astype(np.float32)
        img = (img - np.mean(img))/ (np.std(img)+1e-6)
        img = _center_crop_or_pad(img, self.patch)
        x = torch.from_numpy(img).unsqueeze(0)   # [1,C,D,H,W]
        uw,_ = weak_aug_3d(x, None)              # 教师-弱增
        uc,_ = strong_aug_3d(x, None)            # 学生-强/中增
        return uw.squeeze(0).float(), uc.squeeze(0).float()

def build_semi_loaders(labeled_list: str, unlabeled_list: str, val_list: str,
                       image_suffix: str, in_channels: int, patch_size: Tuple[int,int,int], spacing: Tuple[float,float,float],
                       batch_size_l: int, batch_size_u: int, batch_size_v: int):
    ds_l = LabeledSet3D(labeled_list, patch_size, image_suffix)
    ds_u = UnlabeledSet3D(unlabeled_list, patch_size, image_suffix)
    dl_l = DataLoader(ds_l, batch_size=batch_size_l, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    dl_u = DataLoader(ds_u, batch_size=batch_size_u, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    dl_v = None
    if val_list and os.path.isfile(val_list):
        ds_v = LabeledSet3D(val_list, patch_size, image_suffix)
        dl_v = DataLoader(ds_v, batch_size=batch_size_v, shuffle=False, num_workers=2, pin_memory=True)
    return dl_l, dl_u, dl_v
