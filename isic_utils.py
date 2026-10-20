"""Shared helpers for the ISIC 2020 project (preprocessing, data loading, model, metrics).

Kept in a .py file (not inside the notebook) so that PyTorch DataLoader workers
can import it on Windows. Both notebooks import from here.
"""
from __future__ import annotations

import io
import math
import time

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- preprocessing
def shades_of_gray(rgb: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray color constancy (Finlayson & Trezzi, 2004).

    Rescales each channel so the scene illuminant becomes neutral gray. Reduces
    color cast differences between dermoscopes and clinics.
    """
    img = rgb.astype(np.float32)
    illum = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power)
    illum = illum / (np.linalg.norm(illum) + 1e-6)
    gain = 1.0 / (illum * math.sqrt(3.0) + 1e-6)
    return np.clip(img * gain, 0, 255).astype(np.uint8)


def load_square_rgb(jpeg_bytes: bytes, size: int, color_constancy: bool = False) -> np.ndarray:
    """Decode a JPEG, keep aspect ratio, resize short side to `size`, center-crop to size x size."""
    with Image.open(io.BytesIO(jpeg_bytes)) as im:
        w, h = im.size
        scale = size / min(w, h)
        # draft() lets libjpeg decode at 1/2, 1/4 or 1/8 scale: large speed-up for 6000x4000 images.
        im.draft("RGB", (math.ceil(w * scale), math.ceil(h * scale)))
        im = im.convert("RGB")
        w, h = im.size
        scale = size / min(w, h)
        new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
        im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left, top = (new_w - size) // 2, (new_h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        arr = np.asarray(im, dtype=np.uint8)
    if color_constancy:
        arr = shades_of_gray(arr)
    return arr


# ----------------------------------------------------------------------------- dataset
try:
    import torch
    from torch.utils.data import Dataset
    from torchvision.transforms import v2 as T
except ImportError:  # preprocessing notebook does not need torch
    torch = None
    Dataset = object


class RandomRot90:
    """Rotate by 0/90/180/270 degrees. Dermoscopy images have no natural 'up'."""

    def __call__(self, x):
        return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))


def build_transforms(img_size: int, train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRot90(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(img_size, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CachedISIC(Dataset):
    """Reads uint8 images from the memory-mapped .npy cache written by preprocess_jpeg.ipynb."""

    def __init__(self, cache_path, indices, targets, transform):
        self.cache_path = str(cache_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.transform = transform
        self._images = None  # opened lazily so each worker gets its own memmap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._images is None:
            self._images = np.load(self.cache_path, mmap_mode="r")
        x = torch.from_numpy(np.array(self._images[self.indices[i]])).permute(2, 0, 1)  # HWC -> CHW uint8
        return self.transform(x), torch.tensor(self.targets[i])


# ----------------------------------------------------------------------------- model
def build_model(name: str = "resnet50", pretrained: bool = True):
    import torchvision.models as tvm
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = torch.nn.Linear(m.fc.in_features, 1)
    elif name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = torch.nn.Linear(m.fc.in_features, 1)
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 1)
    elif name == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = torch.nn.Linear(m.classifier.in_features, 1)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def model_size_mb(model) -> float:
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / 1024**2


@torch.no_grad() if torch is not None else (lambda f: f)
def time_per_image_ms(model, img_size, device, batch_size=1, n_iter=100):
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (n_iter * batch_size) * 1000


# ----------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, recall_score, roc_auc_score)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUPRC": average_precision_score(y_true, y_prob),
        "AUROC": roc_auc_score(y_true, y_prob),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "recall_melanoma": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": float(threshold),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def best_f1_threshold(y_true, y_prob) -> float:
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f1[:-1])])


def bootstrap_ci(y_true, y_prob, groups=None, n_boot=1000, seed=0, threshold=0.5):
    """95% bootstrap CIs. If `groups` (patient ids) is given, resample patients, not images."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if groups is None:
        units = [np.array([i]) for i in range(len(y_true))]
    else:
        groups = np.asarray(groups)
        units = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([units[k] for k in pick])
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(compute_metrics(y_true[idx], y_prob[idx], threshold))
    out = {}
    for k in ["AUPRC", "AUROC", "balanced_acc", "recall_melanoma", "F1"]:
        v = np.array([s[k] for s in stats])
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out
