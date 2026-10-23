"""ELBO, importance-weighted likelihood bound and posterior diagnostics.

All quantities are in nats per image (summed over pixels, averaged over
the batch where stated).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_LOG_2PI = math.log(2 * math.pi)


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL( N(mu, diag(exp(logvar))) || N(0, I) ), one value per sample."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)


def bernoulli_nll(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """-log p(x|z) under a factorised Bernoulli decoder, one value per sample."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)


def elbo_terms(model, x: torch.Tensor, beta: float = 1.0):
    """Return (loss, reconstruction NLL, KL); loss is the beta-weighted negative ELBO averaged over the batch."""
    logits, mu, logvar = model(x)
    rec = model.pixel_nll(logits, x).flatten(1).sum(1)
    kl = gaussian_kl(mu, logvar)
    loss = (rec + beta * kl).mean()
    return loss, rec, kl


def _log_normal(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (_LOG_2PI + logvar + (z - mu).pow(2) / logvar.exp()).sum(-1)


@torch.no_grad()
def iwae_nll(model, x: torch.Tensor, k: int = 100, chunk: int = 25) -> torch.Tensor:
    """Importance-weighted estimate of -log p(x) (Burda et al., 2016), one value per sample.

    Uses q(z|x) as the proposal with k samples. As k grows the bound
    tightens towards the true negative log-likelihood; k = 1 recovers the
    ordinary negative ELBO.
    """
    mu, logvar = model.encode(x)
    b, d = mu.shape
    log_w = []
    remaining = k
    while remaining > 0:
        kc = min(chunk, remaining)
        remaining -= kc
        eps = torch.randn(kc, b, d, device=x.device)
        z = mu + torch.exp(0.5 * logvar) * eps  # (kc, b, d)
        log_q = _log_normal(z, mu, logvar)
        log_p_z = -0.5 * (_LOG_2PI + z.pow(2)).sum(-1)
        logits = model.decode(z.reshape(kc * b, d)).reshape(kc, b, *x.shape[1:])
        log_p_x = -model.pixel_nll(logits, x.expand_as(logits)).flatten(2).sum(-1)
        log_w.append(log_p_x + log_p_z - log_q)
    log_w = torch.cat(log_w, dim=0)  # (k, b)
    return -(torch.logsumexp(log_w, dim=0) - math.log(k))


@torch.no_grad()
def active_units(model, loader: DataLoader, device, threshold: float = 0.01) -> tuple[int, torch.Tensor]:
    """Count latent dimensions whose posterior mean varies across the data (Burda et al., 2016).

    A dimension with Var_x( E_q[z_d | x] ) below the threshold carries no
    information about x, i.e. it has collapsed to the prior.
    """
    mus = []
    for x, _ in loader:
        mu, _ = model.encode(x.to(device))
        mus.append(mu.cpu())
    var = torch.cat(mus).var(dim=0)
    return int((var > threshold).sum()), var
