import torch
from torch.distributions import Normal, kl_divergence

from vae import ConvVAE, elbo_terms, gaussian_kl, iwae_nll


def _batch(b=8, c=1):
    torch.manual_seed(0)
    return torch.bernoulli(torch.rand(b, c, 28, 28))


def test_shapes_grayscale_and_rgb():
    for c in (1, 3):
        model = ConvVAE(in_channels=c, latent_dim=8)
        x = _batch(c=c)
        logits, mu, logvar = model(x)
        assert logits.shape == x.shape
        assert mu.shape == logvar.shape == (8, 8)
        assert model.sample(5).shape == (5, c, 28, 28)


def test_samples_are_valid_intensities():
    s = ConvVAE().sample(16)
    assert s.min() >= 0 and s.max() <= 1


def test_kl_matches_torch_distributions():
    torch.manual_seed(0)
    mu, logvar = torch.randn(32, 6), torch.randn(32, 6)
    ref = kl_divergence(Normal(mu, (0.5 * logvar).exp()), Normal(0.0, 1.0)).sum(1)
    assert torch.allclose(gaussian_kl(mu, logvar), ref, atol=1e-5)


def test_kl_is_zero_at_the_prior():
    assert torch.allclose(gaussian_kl(torch.zeros(4, 3), torch.zeros(4, 3)), torch.zeros(4))


def test_importance_weighting_tightens_the_bound():
    torch.manual_seed(0)
    model = ConvVAE(latent_dim=8).eval()
    x = _batch(b=32)
    nll_k1 = iwae_nll(model, x, k=1).mean()
    nll_k64 = iwae_nll(model, x, k=64).mean()
    assert nll_k64 < nll_k1


def test_can_overfit_a_single_batch():
    torch.manual_seed(0)
    model = ConvVAE(latent_dim=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = _batch(b=16)
    first, _, _ = elbo_terms(model, x)
    for _ in range(60):
        loss, _, _ = elbo_terms(model, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.8 * first.item()
