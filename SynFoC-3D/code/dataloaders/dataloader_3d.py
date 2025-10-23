# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

from .augs_3d import strong_aug_3d, weak_aug_3d


def _read_vol(path: str) -> Tuple[np.ndarray, Optional[Tuple[float, float, float]]]:
    spacing = None
    if path.endswith(".npy"):
        v = np.load(path)
    else:
        nii = nib.load(path)
        v = nii.get_fdata()
        spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    if v.ndim == 3:
        v = v[None, ...]  # [C=1,D,H,W]
    elif v.ndim == 4 and v.shape[0] not in [1, 3]:
        # 例如 [D,H,W,C]
        v = np.transpose(v, (3, 0, 1, 2))
    return v, spacing


def _resample_to_spacing(
    vol: np.ndarray,
    current_spacing: Optional[Tuple[float, float, float]],
    target_spacing: Optional[Tuple[float, float, float]],
    order: int,
) -> np.ndarray:
    if current_spacing is None or target_spacing is None:
        return vol
    factors = [current_spacing[i] / target_spacing[i] for i in range(3)]
    if np.allclose(factors, 1.0):
        return vol

    if vol.ndim == 4:
        resampled = [zoom(vol[c], factors, order=order) for c in range(vol.shape[0])]
        return np.stack(resampled, axis=0)
    return zoom(vol, factors, order=order)


def _center_crop_or_pad(vol: np.ndarray, size: Tuple[int, int, int]):
    # vol: [C,D,H,W] ; size: (D,H,W)
    C,D,H,W = vol.shape
    d,h,w = size
    out = np.zeros((C,d,h,w), dtype=vol.dtype)
    sd, sh, sw = max(0, (d-D)//2), max(0, (h-H)//2), max(0, (w-W)//2)
    td, th, tw = max(0, (D-d)//2), max(0, (H-h)//2), max(0, (W-w)//2)
    ds = min(d, D); hs = min(h, H); ws = min(w, W)
    out[:, sd:sd+ds, sh:sh+hs, sw:sw+ws] = vol[:, td:td+ds, th:th+hs, tw:tw+ws]
    return out


def _normalise(img: np.ndarray) -> np.ndarray:
    mean = img.mean(axis=(1, 2, 3), keepdims=True)
    std = img.std(axis=(1, 2, 3), keepdims=True)
    return (img - mean) / (std + 1e-6)

class LabeledSet3D(Dataset):
    def __init__(
        self,
        index_file: str,
        patch_size: Tuple[int, int, int],
        image_suffix: str,
        spacing: Optional[Tuple[float, float, float]],
        augment: bool = True,
    ):
        self.items = []  # (image_path, label_path)
        with open(index_file, 'r') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                img_path, lab_path = line.split(",")
                self.items.append((img_path.strip(), lab_path.strip()))
        self.patch = patch_size
        self.suffix = image_suffix
        self.spacing = spacing
        self.augment = augment

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img_path, lab_path = self.items[i]
        img, img_spacing = _read_vol(img_path)
        lab, lab_spacing = _read_vol(lab_path)

        img = _resample_to_spacing(img, img_spacing, self.spacing, order=1).astype(np.float32)
        lab = _resample_to_spacing(lab, lab_spacing, self.spacing, order=0).astype(np.int64)

        img = _normalise(img)
        img = _center_crop_or_pad(img, self.patch)
        lab = _center_crop_or_pad(lab, self.patch)

        if lab.shape[0] > 1:
            lab = np.argmax(lab, axis=0)
        else:
            lab = lab[0]

        x = torch.from_numpy(img)
        y = torch.from_numpy(lab)

        if self.augment:
            x, y = weak_aug_3d(x.unsqueeze(0), y.unsqueeze(0))
            x = x.squeeze(0)
            y = y.squeeze(0)

        return x.float(), y.long()

class UnlabeledSet3D(Dataset):
    def __init__(
        self,
        index_file: str,
        patch_size: Tuple[int, int, int],
        image_suffix: str,
        spacing: Optional[Tuple[float, float, float]],
    ):
        self.items = []
        with open(index_file, 'r') as f:
            for line in f:
                p=line.strip()
                if p: self.items.append(p)
        self.patch = patch_size
        self.suffix = image_suffix
        self.spacing = spacing

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img_path = self.items[i]
        img, img_spacing = _read_vol(img_path)
        img = _resample_to_spacing(img, img_spacing, self.spacing, order=1).astype(np.float32)
        img = _normalise(img)
        img = _center_crop_or_pad(img, self.patch)
        x = torch.from_numpy(img)
        uw, _ = weak_aug_3d(x.unsqueeze(0), None)
        uc, _ = strong_aug_3d(x.unsqueeze(0), None)
        return uw.squeeze(0).float(), uc.squeeze(0).float()

def build_semi_loaders(labeled_list: str, unlabeled_list: str, val_list: str,
                       image_suffix: str, in_channels: int, patch_size: Tuple[int,int,int], spacing: Tuple[float,float,float],
                       batch_size_l: int, batch_size_u: int, batch_size_v: int, num_workers: int = 2):
    if spacing is None:
        spacing_t = None
    else:
        spacing_t = tuple(spacing)
        if not all(s > 0 for s in spacing_t):
            spacing_t = None
    ds_l = LabeledSet3D(labeled_list, patch_size, image_suffix, spacing_t, augment=True)
    ds_u = UnlabeledSet3D(unlabeled_list, patch_size, image_suffix, spacing_t)
    dl_l = DataLoader(ds_l, batch_size=batch_size_l, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_u = DataLoader(ds_u, batch_size=batch_size_u, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_v = None
    if val_list and os.path.isfile(val_list):
        ds_v = LabeledSet3D(val_list, patch_size, image_suffix, spacing_t, augment=False)
        dl_v = DataLoader(ds_v, batch_size=batch_size_v, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_l, dl_u, dl_v
