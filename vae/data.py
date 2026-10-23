"""Data loading for MNIST, Fashion-MNIST and any 2D MedMNIST dataset at 28x28.

MNIST and Fashion-MNIST are binarised by default: dynamically (resampled every epoch)
for training, and once with a fixed seed for validation and test so the
reported likelihoods are reproducible and comparable with the literature.
MedMNIST images are kept as grey-level / RGB intensities in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

TORCHVISION_SETS = {"mnist": datasets.MNIST, "fashionmnist": datasets.FashionMNIST}
EVAL_BINARISE_SEED = 1234
N_VAL = 5000  # held out from the 60k MNIST training images


@dataclass
class DataInfo:
    name: str
    channels: int
    n_classes: int
    class_names: list[str] = field(default_factory=list)
    binarised: bool = False


class DynamicBinarise:
    """Sample each pixel from Bernoulli(intensity); a fresh draw on every access."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(x)


class StaticBinarised(Dataset):
    """Binarise a dataset once with a fixed seed and keep it in memory."""

    def __init__(self, base: Dataset, seed: int = EVAL_BINARISE_SEED) -> None:
        xs, ys = zip(*(base[i] for i in range(len(base))))
        g = torch.Generator().manual_seed(seed)
        self.x = torch.bernoulli(torch.stack(xs), generator=g)
        self.y = torch.as_tensor(ys)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class _MedMNISTWrapper(Dataset):
    """Return (image, int label) instead of MedMNIST's (image, array label)."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        y = torch.as_tensor(y).flatten()
        # Multi-label sets (e.g. ChestMNIST) have no single class; use -1.
        return x, int(y[0]) if y.numel() == 1 else -1


def _torchvision_splits(name: str, root: str, seed: int, binarise: bool = True):
    cls = TORCHVISION_SETS[name]
    to_tensor = transforms.ToTensor()
    train_tf = transforms.Compose([to_tensor, DynamicBinarise()]) if binarise else to_tensor
    train_dyn = cls(root, train=True, download=True, transform=train_tf)
    train_plain = cls(root, train=True, download=True, transform=to_tensor)
    test_plain = cls(root, train=False, download=True, transform=to_tensor)

    perm = torch.randperm(len(train_plain), generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:N_VAL].tolist(), perm[N_VAL:].tolist()

    train = Subset(train_dyn, train_idx)
    if binarise:
        val, test = StaticBinarised(Subset(train_plain, val_idx)), StaticBinarised(test_plain)
    else:
        val, test = Subset(train_plain, val_idx), test_plain
    info = DataInfo(name=name, channels=1, n_classes=10,
                    class_names=list(train_plain.classes), binarised=binarise)
    return train, val, test, info


def _medmnist_splits(flag: str, root: str):
    try:
        import medmnist
    except ImportError as e:  # pragma: no cover
        raise ImportError("MedMNIST datasets need `pip install medmnist`.") from e
    if flag not in medmnist.INFO:
        raise ValueError(f"Unknown dataset '{flag}'. Try mnist, fashionmnist or a MedMNIST flag "
                         f"such as pneumoniamnist, bloodmnist, organamnist.")
    meta = medmnist.INFO[flag]
    if "3d" in flag:
        raise ValueError("3D MedMNIST volumes are not supported by this 2D model.")
    cls = getattr(medmnist, meta["python_class"])
    tf = transforms.ToTensor()
    splits = [_MedMNISTWrapper(cls(split=s, transform=tf, download=True, root=root, size=28))
              for s in ("train", "val", "test")]
    labels = meta["label"]
    info = DataInfo(name=flag, channels=meta["n_channels"], n_classes=len(labels),
                    class_names=[labels[str(i)] for i in range(len(labels))], binarised=False)
    return (*splits, info)


def get_datasets(name: str, root: str = "data", seed: int = 0, binarise: bool = True):
    """Return (train, val, test, DataInfo). `binarise` applies to MNIST / Fashion-MNIST only."""
    name = name.lower()
    if name in TORCHVISION_SETS:
        return _torchvision_splits(name, root, seed, binarise)
    import os
    os.makedirs(root, exist_ok=True)
    return _medmnist_splits(name, root)


def get_loaders(name: str, root: str = "data", batch_size: int = 128,
                num_workers: int = 2, seed: int = 0, binarise: bool = True):
    train, val, test, info = get_datasets(name, root, seed, binarise)
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers)
    return (
        DataLoader(train, shuffle=True, drop_last=True, generator=g, **kw),
        DataLoader(val, shuffle=False, **kw),
        DataLoader(test, shuffle=False, **kw),
        info,
    )
