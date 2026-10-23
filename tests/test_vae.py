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


def test_gaussian_nll_matches_torch_normal_in_eval_mode():
    torch.manual_seed(0)
    model = ConvVAE(latent_dim=4, likelihood="gaussian").eval()
    x = torch.rand(4, 1, 28, 28)
    logits = torch.randn(4, 1, 28, 28)
    ref = -Normal(torch.sigmoid(logits), model.log_sigma.exp()).log_prob(x)
    assert torch.allclose(model.pixel_nll(logits, x), ref, atol=1e-4)


def test_gaussian_sigma_is_calibrated_during_training():
    torch.manual_seed(0)
    model = ConvVAE(latent_dim=4, likelihood="gaussian").train()
    x = torch.rand(8, 1, 28, 28)
    before = model.log_sigma.clone()
    elbo_terms(model, x)
    assert not torch.equal(before, model.log_sigma)  # running estimate moved


def test_gaussian_decoder_learns_grey_level_images():
    torch.manual_seed(0)
    model = ConvVAE(in_channels=3, latent_dim=8, likelihood="gaussian")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(16, 3, 28, 28)
    mse0 = (model.reconstruct(x) - x).pow(2).mean().item()
    for _ in range(60):
        loss, _, _ = elbo_terms(model, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (model.reconstruct(x) - x).pow(2).mean().item() < mse0


def test_old_checkpoint_config_still_loads():
    cfg = {"in_channels": 1, "latent_dim": 16, "hidden_dim": 256, "image_size": 28}
    model = ConvVAE(**cfg)
    model.load_state_dict(ConvVAE(**cfg).state_dict())
    assert model.likelihood == "bernoulli"
