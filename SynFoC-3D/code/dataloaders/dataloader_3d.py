# -*- coding: utf-8 -*-
import glob
import os
from typing import Dict, Iterable, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

from .augs_3d import strong_aug_3d, weak_aug_3d


_PATH_CACHE: Dict[Tuple[str, str, Optional[str]], str] = {}
_MISSING = object()
_STEM_CACHE: Dict[Tuple[str, str], Optional[str]] = {}


def _split_double_ext(filename: str) -> Tuple[str, str]:
    """Split ``filename`` into stem and extension, handling ``.nii.gz`` etc."""

    base, ext = os.path.splitext(filename)
    if ext.lower() == ".gz":
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return base, ext


_SIBLING_HINTS = ("data", "dataset", "datasets", "database")


def _iter_roots(base_dir: str) -> Iterable[str]:
    base_dir = os.path.abspath(base_dir)
    seen = set()

    def _add(candidate: str):
        if not candidate:
            return
        candidate = os.path.abspath(candidate)
        if candidate not in seen:
            seen.add(candidate)
            yield candidate

    base_name = os.path.basename(base_dir)

    # Current directory and a few ancestors are useful when the actual data
    # lives outside the repo checkout (e.g. ``../data`` or a mounted drive).
    cursor = base_dir
    for _ in range(4):
        for item in _add(cursor):
            yield item
        parent = os.path.dirname(cursor)
        if parent == cursor:
            break
        for item in _add(os.path.join(parent, base_name)):
            yield item

        # Some datasets live in sibling directories such as ``../data`` while
        # the index files stay inside the repository checkout.  Walk a small set
        # of heuristically common sibling names so we can discover layouts such
        # as ``../data/ACDC/database`` automatically.
        for hint in _SIBLING_HINTS:
            sibling = os.path.join(parent, hint)
            if not os.path.isdir(sibling):
                continue
            for item in _add(sibling):
                yield item
            for item in _add(os.path.join(sibling, base_name)):
                yield item
        cursor = parent

    # Allow users to override the search root via environment variables.  A
    # colon/semicolon separated list is supported to cover Unix/Windows shells.
    for key in (
        "SYNCF3D_DATA_ROOT",
        "SYNCFOC3D_DATA_ROOT",
        "SYNCFOC_DATA_ROOT",
        "SYNFOC3D_DATA_ROOT",
    ):
        env = os.environ.get(key)
        if not env:
            continue
        for chunk in env.split(os.pathsep):
            chunk = chunk.strip()
            if not chunk:
                continue
            for item in _add(chunk):
                yield item
            for item in _add(os.path.join(chunk, base_name)):
                yield item


def _resolve_path(path: str, base_dir: str, suffix: Optional[str] = None) -> str:
    """Normalise dataset paths so index files work cross-platform.

    Besides fixing Windows separators, dataset indices sometimes omit the file
    suffix (``.nii.gz``).  We try a few suffix candidates so the loader works on
    both Linux and Windows without forcing users to edit the index files.
    """

    key = (os.path.abspath(base_dir), path, suffix)
    cached = _PATH_CACHE.get(key)
    if cached:
        return cached

    cleaned = path.strip().replace("\\", "/")
    if not cleaned:
        _PATH_CACHE[key] = cleaned
        return cleaned

    suffix = suffix.strip() if suffix else ""

    roots = list(_iter_roots(base_dir))
    last = os.path.normpath(cleaned if os.path.isabs(cleaned) else os.path.join(base_dir, cleaned))

    def _candidates_for(root: str):
        # ``cleaned`` may already be absolute; otherwise treat it relative to
        # the candidate root.
        if os.path.isabs(cleaned):
            guess = cleaned
        else:
            guess = os.path.join(root, cleaned)
        yield guess
        if suffix:
            if not guess.endswith(suffix):
                yield guess + suffix
                base_guess, ext_guess = os.path.splitext(guess)
                if ext_guess and suffix.startswith(ext_guess):
                    yield base_guess + suffix

    for root in roots:
        for cand in _candidates_for(root):
            norm = os.path.normpath(cand)
            if os.path.isfile(norm):
                _PATH_CACHE[key] = norm
                return norm
            last = norm

    # Fallback 1: try to locate the relative path anywhere under the candidate
    # roots.  This covers layouts such as ``database/training`` that are not
    # encoded in the index file paths.
    if not os.path.isabs(cleaned):
        rel_os = cleaned.replace("/", os.sep)
        rel_variants = [rel_os]
        if suffix and not rel_os.endswith(suffix):
            rel_variants.append(rel_os + suffix)
            base_rel, ext_rel = os.path.splitext(rel_os)
            if ext_rel and suffix.startswith(ext_rel):
                rel_variants.append(base_rel + suffix)
        for root in roots:
            if not os.path.isdir(root):
                continue
            for variant in rel_variants:
                pattern = os.path.join(root, "**", variant)
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    norm = os.path.normpath(matches[0])
                    _PATH_CACHE[key] = norm
                    return norm

    # Fallback 2: search by stem so ``.nii`` ↔ ``.nii.gz`` mismatches are
    # resolved even when the directory hierarchy differs.
    filename = os.path.basename(cleaned)
    stem, _ = _split_double_ext(filename)
    for root in roots:
        cache_key = (root, stem)
        cached_stem = _STEM_CACHE.get(cache_key, _MISSING)
        if cached_stem is _MISSING:
            if not os.path.isdir(root):
                _STEM_CACHE[cache_key] = None
                continue
            for dirpath, _, filenames in os.walk(root):
                for entry in filenames:
                    entry_stem, _ = _split_double_ext(entry)
                    if entry_stem == stem:
                        resolved = os.path.normpath(os.path.join(dirpath, entry))
                        _STEM_CACHE[cache_key] = resolved
                        _PATH_CACHE[key] = resolved
                        return resolved
            _STEM_CACHE[cache_key] = None
            cached_stem = None
        if cached_stem:
            _PATH_CACHE[key] = cached_stem
            return cached_stem

    _PATH_CACHE[key] = last
    return last


def _read_vol(path: str) -> Tuple[np.ndarray, Optional[Tuple[float, float, float]]]:
    """Read a volume from ``path`` and return channel-first data with spacing.

    The datasets we support may store 3D data in a variety of layouts:

    * ``(D, H, W)`` single-channel volumes.
    * ``(C, D, H, W)`` channel-first tensors saved via ``numpy``.
    * ``(D, H, W, C)`` channel-last tensors produced by ``nibabel`` or other
      medical imaging toolkits.

    The previous implementation only handled the last case when the first
    dimension was not 1 or 3, which broke common multi-modal datasets such as
    BraTS (shape ``(H, W, D, 4)``) because the heuristic mis-identified the
    channel axis.  Here we normalise by explicitly checking both ends and falling
    back to moving the smallest dimension to the channel axis when ambiguous.
    """

    spacing = None
    if path.endswith(".npy"):
        v = np.load(path)
    else:
        nii = nib.load(path)
        v = nii.get_fdata()
        spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])

    v = np.asarray(v)

    if v.ndim == 3:
        return v[None, ...], spacing  # [C=1,D,H,W]

    if v.ndim != 4:
        raise ValueError(f"Unsupported volume shape {v.shape} for {path}")

    # Two common cases: channel already first (C,D,H,W) or last (D,H,W,C).
    if v.shape[0] <= 4:
        # Treat small first dimension as channels. Copy to avoid negative strides
        # from nibabel proxy arrays when later torch.from_numpy is called.
        return np.ascontiguousarray(v), spacing

    if v.shape[-1] <= 4:
        return np.ascontiguousarray(np.moveaxis(v, -1, 0)), spacing

    # Ambiguous case: move the smallest dimension to the front as channels.
    channel_axis = int(np.argmin(v.shape))
    if channel_axis == 0:
        return np.ascontiguousarray(v), spacing
    return np.ascontiguousarray(np.moveaxis(v, channel_axis, 0)), spacing



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
        base_dir = os.path.dirname(os.path.abspath(index_file))
        with open(index_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                img_path, lab_path = line.split(",")
                img_path = _resolve_path(img_path, base_dir, image_suffix)
                lab_path = _resolve_path(lab_path, base_dir, image_suffix)
                self.items.append((img_path, lab_path))
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
        base_dir = os.path.dirname(os.path.abspath(index_file))
        with open(index_file, 'r') as f:
            for line in f:
                resolved = _resolve_path(line, base_dir, image_suffix)
                if resolved:
                    self.items.append(resolved)
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
