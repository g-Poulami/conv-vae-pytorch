"""Figures: reconstructions, prior samples, interpolations, latent space, training curves.

Usage:
    python -m vae.visualise --run-dir runs/mnist_z16
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torchvision.utils import make_grid, save_image  # noqa: E402

from .data import get_loaders  # noqa: E402
from .utils import get_device, load_checkpoint, set_seed  # noqa: E402


@torch.no_grad()
def save_reconstructions(model, x: torch.Tensor, path: Path, n: int = 16) -> None:
    """Top row: inputs. Bottom row: reconstructions through the posterior mean."""
    x = x[:n]
    recon = model.reconstruct(x)
    save_image(torch.cat([x, recon]).cpu(), path, nrow=n, padding=2, pad_value=1.0)


@torch.no_grad()
def save_samples(model, path: Path, n: int = 64, device="cpu") -> None:
    save_image(model.sample(n, device).cpu(), path, nrow=8, padding=2, pad_value=1.0)


@torch.no_grad()
def save_interpolations(model, x: torch.Tensor, path: Path, pairs: int = 6, steps: int = 10) -> None:
    """Linear interpolation in latent space between pairs of test images."""
    mu, _ = model.encode(x[: 2 * pairs])
    a, b = mu[:pairs], mu[pairs : 2 * pairs]
    t = torch.linspace(0, 1, steps, device=mu.device).view(1, steps, 1)
    z = (1 - t) * a.unsqueeze(1) + t * b.unsqueeze(1)
    imgs = torch.sigmoid(model.decode(z.reshape(-1, mu.shape[1])))
    save_image(imgs.cpu(), path, nrow=steps, padding=2, pad_value=1.0)


@torch.no_grad()
def save_latent_scatter(model, loader, device, path: Path, class_names=None, max_points: int = 10000) -> None:
    """Posterior means projected to 2D (PCA unless the latent is already 2D), coloured by class."""
    mus, ys = [], []
    for x, y in loader:
        mu, _ = model.encode(x.to(device))
        mus.append(mu.cpu())
        ys.append(torch.as_tensor(y))
        if sum(len(m) for m in mus) >= max_points:
            break
    mu = torch.cat(mus)[:max_points]
    y = torch.cat(ys)[:max_points]
    if mu.shape[1] > 2:
        centred = mu - mu.mean(0)
        _, s, v = torch.linalg.svd(centred, full_matrices=False)
        proj = centred @ v[:2].T
        explained = (s[:2] ** 2).sum() / (s ** 2).sum()
        xlabel, ylabel = "PC 1", "PC 2"
        title = f"Posterior means, PCA ({100 * explained:.0f}% variance)"
    else:
        proj, xlabel, ylabel, title = mu, "z1", "z2", "Posterior means"

    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=150)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=y, cmap="tab10", s=3, alpha=0.7, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if class_names and int(y.min()) >= 0:
        handles = [plt.Line2D([], [], marker="o", ls="", color=sc.cmap(sc.norm(i)), label=str(c))
                   for i, c in enumerate(class_names) if (y == i).any()]
        ax.legend(handles=handles, fontsize=7, markerscale=1.2, frameon=False,
                  bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def save_manifold(model, path: Path, n: int = 20, device="cpu") -> None:
    """For a 2D latent: decode a grid of points at standard-normal quantiles."""
    if model.latent_dim != 2:
        return
    q = torch.distributions.Normal(0, 1).icdf(torch.linspace(0.03, 0.97, n))
    zz = torch.stack(torch.meshgrid(q, q.flip(0), indexing="xy"), dim=-1).reshape(-1, 2).to(device)
    imgs = torch.sigmoid(model.decode(zz)).cpu()
    save_image(imgs, path, nrow=n, padding=0)


def save_loss_curves(history_csv: Path, path: Path) -> None:
    rows = list(csv.DictReader(open(history_csv)))
    ep = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=150)
    for ax, key, label in zip(
        axes,
        ["neg_elbo", "rec", "kl"],
        ["Negative ELBO (nats)", "Reconstruction NLL (nats)", "KL (nats)"],
    ):
        ax.plot(ep, [float(r[f"train_{key}"]) for r in rows], label="train")
        ax.plot(ep, [float(r[f"val_{key}"]) for r in rows], label="validation")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("epoch")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_all_figures(model, test_loader, info, device, out: Path, history_csv: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    x, _ = next(iter(test_loader))
    x = x.to(device)
    save_reconstructions(model, x, out / "reconstructions.png")
    save_samples(model, out / "samples.png", device=device)
    save_interpolations(model, x, out / "interpolations.png")
    save_latent_scatter(model, test_loader, device, out / "latent_space.png", info.class_names)
    save_manifold(model, out / "manifold.png", device=device)
    if history_csv and Path(history_csv).exists():
        save_loss_curves(history_csv, out / "loss_curves.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.run_dir / args.checkpoint, device)
    cfg = ckpt["train_config"]
    _, _, test_loader, info = get_loaders(cfg["dataset"], args.data_dir, cfg["batch_size"], 0, cfg["seed"])
    make_all_figures(model, test_loader, info, device, args.run_dir / "figures", args.run_dir / "history.csv")
    print(f"Figures written to {args.run_dir / 'figures'}")


if __name__ == "__main__":
    main()
