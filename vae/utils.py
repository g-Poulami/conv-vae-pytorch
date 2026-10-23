from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .model import ConvVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(choice: str = "auto") -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: ConvVAE, extra: dict) -> None:
    torch.save({"model_config": model.config(), "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: Path, device) -> tuple[ConvVAE, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ConvVAE(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def write_json(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")
