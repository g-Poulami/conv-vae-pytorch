"""Evaluate a trained VAE on the test set.

Reports, in nats per image:
  * negative ELBO, split into reconstruction NLL and KL
  * importance-weighted estimate of -log p(x) with k samples (tighter than the ELBO)
  * number of active latent units

Usage:
    python -m vae.evaluate --run-dir runs/mnist_z16 --iw-samples 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import get_loaders
from .losses import active_units, elbo_terms, iwae_nll
from .utils import get_device, load_checkpoint, set_seed, write_json


@torch.no_grad()
def elbo_on_loader(model, loader, device) -> dict:
    tot = {"neg_elbo": 0.0, "rec": 0.0, "kl": 0.0}
    n = 0
    for x, _ in loader:
        x = x.to(device)
        _, rec, kl = elbo_terms(model, x)
        tot["rec"] += rec.sum().item()
        tot["kl"] += kl.sum().item()
        tot["neg_elbo"] += (rec + kl).sum().item()
        n += x.shape[0]
    return {k: v / n for k, v in tot.items()}


@torch.no_grad()
def evaluate(model, loader, device, iw_samples: int = 100, seed: int = 0) -> dict:
    set_seed(seed)
    model.eval()
    metrics = {f"test_{k}": v for k, v in elbo_on_loader(model, loader, device).items()}
    if iw_samples > 0:
        total, n = 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            total += iwae_nll(model, x, k=iw_samples).sum().item()
            n += x.shape[0]
        metrics[f"test_nll_iw{iw_samples}"] = total / n
    au, var = active_units(model, loader, device)
    metrics["active_units"] = au
    metrics["latent_dim"] = model.latent_dim
    metrics["posterior_mean_variance_per_dim"] = [round(v, 4) for v in var.tolist()]
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--iw-samples", type=int, default=100)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.run_dir / args.checkpoint, device)
    cfg = ckpt["train_config"]
    _, _, test_loader, _ = get_loaders(cfg["dataset"], args.data_dir, 256, 0, cfg["seed"])
    metrics = evaluate(model, test_loader, device, args.iw_samples)
    metrics["checkpoint_epoch"] = ckpt.get("epoch")
    write_json(args.run_dir / "test_metrics.json", metrics)
    for k, v in metrics.items():
        if k != "posterior_mean_variance_per_dim":
            print(f"{k:>24}: {v:.2f}" if isinstance(v, float) else f"{k:>24}: {v}")


if __name__ == "__main__":
    main()
