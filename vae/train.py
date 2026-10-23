"""Train a convolutional VAE.

Examples:
    python -m vae.train --dataset mnist --latent-dim 16 --epochs 30
    python -m vae.train --dataset pneumoniamnist --latent-dim 16 --epochs 50   # Gaussian decoder (auto)
    python -m vae.train --dataset mnist --latent-dim 2 --epochs 20 --name mnist_z2

Each run writes to runs/<name>/: config.json, history.csv, best.pt
(lowest validation negative ELBO), last.pt, test_metrics.json and figures/.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from .data import get_loaders
from .evaluate import elbo_on_loader, evaluate
from .losses import elbo_terms
from .model import ConvVAE
from .utils import get_device, load_checkpoint, save_checkpoint, set_seed, write_json
from .visualise import make_all_figures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mnist",
                   help="mnist, fashionmnist, or a 2D MedMNIST flag (pneumoniamnist, bloodmnist, ...)")
    p.add_argument("--likelihood", choices=["auto", "bernoulli", "gaussian"], default="auto",
                   help="decoder likelihood; auto = bernoulli for binarised data, gaussian otherwise")
    p.add_argument("--no-binarise", action="store_true",
                   help="keep MNIST / Fashion-MNIST as grey levels instead of binarising")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0, help="weight on the KL term (beta-VAE); 1.0 = standard VAE")
    p.add_argument("--kl-warmup-epochs", type=int, default=0,
                   help="linearly anneal beta from 0 over this many epochs (0 = off)")
    p.add_argument("--iw-samples", type=int, default=100, help="importance samples for the test NLL (0 = skip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--name", default=None, help="run name (default: <dataset>_z<latent-dim>)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def beta_at(epoch: int, beta: float, warmup: int) -> float:
    if warmup <= 0:
        return beta
    return beta * min(1.0, epoch / warmup)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    run_dir = Path(args.out_dir) / (args.name or f"{args.dataset}_z{args.latent_dim}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, info = get_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.seed, not args.no_binarise
    )
    likelihood = args.likelihood
    if likelihood == "auto":
        likelihood = "bernoulli" if info.binarised else "gaussian"
    model = ConvVAE(in_channels=info.channels, latent_dim=args.latent_dim,
                    hidden_dim=args.hidden_dim, likelihood=likelihood).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    train_config = {**vars(args), "n_params": n_params, "device": str(device),
                    "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
                    "n_test": len(test_loader.dataset), "binarised": info.binarised,
                    "likelihood_resolved": likelihood}
    write_json(run_dir / "config.json", train_config)
    print(f"{info.name}: {train_config['n_train']} train / {train_config['n_val']} val / "
          f"{train_config['n_test']} test | {n_params:,} parameters | {likelihood} decoder | device {device}")

    history_path = run_dir / "history.csv"
    fields = ["epoch", "beta", "train_neg_elbo", "train_rec", "train_kl",
              "val_neg_elbo", "val_rec", "val_kl", "seconds"]
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        beta = beta_at(epoch, args.beta, args.kl_warmup_epochs)
        model.train()
        tot_rec = tot_kl = 0.0
        n = 0
        for x, _ in train_loader:
            x = x.to(device)
            loss, rec, kl = elbo_terms(model, x, beta=beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_rec += rec.sum().item()
            tot_kl += kl.sum().item()
            n += x.shape[0]

        model.eval()
        val = elbo_on_loader(model, val_loader, device)
        row = {
            "epoch": epoch, "beta": beta,
            "train_neg_elbo": (tot_rec + tot_kl) / n, "train_rec": tot_rec / n, "train_kl": tot_kl / n,
            "val_neg_elbo": val["neg_elbo"], "val_rec": val["rec"], "val_kl": val["kl"],
            "seconds": time.time() - t0,
        }
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{row[k]:.4f}" if isinstance(row[k], float) else row[k] for k in fields])
        print(f"epoch {epoch:3d} | beta {beta:.2f} | train -ELBO {row['train_neg_elbo']:7.2f} | "
              f"val -ELBO {val['neg_elbo']:7.2f} (rec {val['rec']:6.2f}, KL {val['kl']:5.2f}) | "
              f"{row['seconds']:.0f}s")

        extra = {"epoch": epoch, "train_config": train_config, "data_info": vars(info)}
        save_checkpoint(run_dir / "last.pt", model, extra)
        # Only select on epochs trained with the full KL weight, so the ELBO is a valid bound.
        if beta == args.beta and val["neg_elbo"] < best_val:
            best_val = val["neg_elbo"]
            save_checkpoint(run_dir / "best.pt", model, extra)

    model, ckpt = load_checkpoint(run_dir / "best.pt", device)
    metrics = evaluate(model, test_loader, device, args.iw_samples, args.seed)
    metrics["checkpoint_epoch"] = ckpt["epoch"]
    metrics["best_val_neg_elbo"] = best_val
    write_json(run_dir / "test_metrics.json", metrics)
    make_all_figures(model, test_loader, info, device, run_dir / "figures", history_path)
    print("Test:", {k: (round(v, 2) if isinstance(v, float) else v)
                    for k, v in metrics.items() if k != "posterior_mean_variance_per_dim"})


if __name__ == "__main__":
    main()
