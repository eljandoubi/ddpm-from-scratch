import pytest
import torch

from sampler import Sampler


@pytest.fixture
def sampler():
    return Sampler(total_timesteps=50, device="cpu")


def test_cumulative_alpha_decreases(sampler):
    cumulative = sampler.alpha_cumulative_prod
    assert (cumulative[1:] < cumulative[:-1]).all()
    assert 0 < cumulative[-1] < cumulative[0] < 1


def test_default_schedule_ends_near_pure_noise():
    cumulative = Sampler(device="cpu").alpha_cumulative_prod
    assert cumulative[-1] < 1e-3


def test_add_noise_follows_the_closed_form(sampler):
    clean = torch.randn(4, 3, 8, 8)
    timesteps = torch.tensor([0, 10, 25, 49])
    noisy, noise = sampler.add_noise(clean, timesteps)
    alpha = sampler.alpha_cumulative_prod[timesteps].view(-1, 1, 1, 1)
    expected = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    assert torch.allclose(noisy, expected, atol=1e-6)


def test_remove_noise_at_step_zero_recovers_the_clean_image(sampler):
    clean = torch.randn(3, 3, 8, 8)
    timesteps = torch.zeros(3, dtype=torch.long)
    noisy, noise = sampler.add_noise(clean, timesteps)
    assert torch.allclose(
        sampler.remove_noise(noisy, timesteps, noise), clean, atol=1e-4
    )


def test_remove_noise_rejects_mismatched_shapes(sampler):
    with pytest.raises(AssertionError):
        sampler.remove_noise(
            torch.zeros(2, 3, 8, 8), torch.tensor([1, 1]), torch.zeros(2, 3, 4, 4)
        )


@pytest.mark.parametrize("t", [0, 1, 17, 49])
def test_clipping_changes_nothing_when_the_estimate_is_in_range(sampler, t):
    clean = torch.rand(3, 3, 8, 8) - 0.5
    timesteps = torch.full((3,), t)
    noisy, noise = sampler.add_noise(clean, timesteps)
    torch.manual_seed(0)
    plain = sampler.remove_noise(noisy, timesteps, noise)
    torch.manual_seed(0)
    clipped = sampler.remove_noise(noisy, timesteps, noise, clip_x0=True)
    assert torch.allclose(plain, clipped, atol=1e-4)


def test_clipping_limits_the_clean_image_estimate(sampler):
    timesteps = torch.zeros(2, dtype=torch.long)
    noisy = torch.tensor([3.0, -3.0]).view(2, 1, 1, 1).expand(2, 3, 4, 4)
    no_noise = torch.zeros_like(noisy)
    clipped = sampler.remove_noise(noisy, timesteps, no_noise, clip_x0=True)
    assert torch.equal(clipped[:, 0, 0, 0], torch.tensor([1.0, -1.0]))
    assert sampler.remove_noise(noisy, timesteps, no_noise).abs().min() > 2.9


def test_clipped_step_is_the_posterior_mean_of_the_clipped_estimate(sampler):
    t = 30
    timesteps = torch.full((1,), t)
    noisy = torch.full((1, 3, 4, 4), 5.0)
    no_noise = torch.zeros_like(noisy)
    torch.manual_seed(0)
    stepped = sampler.remove_noise(noisy, timesteps, no_noise, clip_x0=True)
    a_t, a_prev = sampler.alpha_cumulative_prod[t], sampler.alpha_cumulative_prod[t - 1]
    beta, alpha = sampler.beta_schedule[t], sampler.alpha[t]
    torch.manual_seed(0)
    sigma_z = torch.randn_like(noisy) * (beta * (1 - a_prev) / (1 - a_t)).sqrt()
    mean = (
        a_prev.sqrt() * beta / (1 - a_t) * 1.0
        + alpha.sqrt() * (1 - a_prev) / (1 - a_t) * 5.0
    )
    assert torch.allclose(stepped, mean + sigma_z, atol=1e-5)


def test_ddim_step_with_the_true_noise_follows_the_forward_process(sampler):
    clean = torch.rand(2, 3, 8, 8) * 2 - 1
    noise = torch.randn_like(clean)
    t, prev = 40, 20
    alpha_t = sampler.alpha_cumulative_prod[t]
    alpha_prev = sampler.alpha_cumulative_prod[prev]
    noisy = alpha_t.sqrt() * clean + (1 - alpha_t).sqrt() * noise
    stepped = sampler.ddim_step(noisy, t, prev, noise)
    expected = alpha_prev.sqrt() * clean + (1 - alpha_prev).sqrt() * noise
    assert torch.allclose(stepped, expected, atol=1e-4)


def test_ddim_last_step_returns_the_clean_estimate(sampler):
    clean = torch.rand(2, 3, 8, 8) * 2 - 1
    noise = torch.randn_like(clean)
    alpha_t = sampler.alpha_cumulative_prod[30]
    noisy = alpha_t.sqrt() * clean + (1 - alpha_t).sqrt() * noise
    assert torch.allclose(sampler.ddim_step(noisy, 30, -1, noise), clean, atol=1e-4)


def test_ddim_clipping_recomputes_the_noise_from_the_clipped_estimate(sampler):
    t, prev = 40, 20
    alpha_t = sampler.alpha_cumulative_prod[t]
    alpha_prev = sampler.alpha_cumulative_prod[prev]
    noisy = torch.randn(2, 3, 8, 8) * 3
    predicted = torch.randn_like(noisy) * 0.1  # far too small: the estimate overshoots
    raw = (noisy - (1 - alpha_t).sqrt() * predicted) / alpha_t.sqrt()
    assert raw.abs().max() > 1

    clipped = raw.clamp(-1, 1)
    consistent = (noisy - alpha_t.sqrt() * clipped) / (1 - alpha_t).sqrt()
    expected = alpha_prev.sqrt() * clipped + (1 - alpha_prev).sqrt() * consistent
    assert torch.allclose(
        sampler.ddim_step(noisy, t, prev, predicted), expected, atol=1e-5
    )

    unclipped = alpha_prev.sqrt() * raw + (1 - alpha_prev).sqrt() * predicted
    result = sampler.ddim_step(noisy, t, prev, predicted, clip_x0=False)
    assert torch.allclose(result, unclipped, atol=1e-5)


def test_ddim_noise_is_random_only_when_eta_is_positive(sampler):
    noisy = torch.randn(2, 3, 8, 8)
    noise = torch.randn_like(noisy)
    first = sampler.ddim_step(noisy, 40, 20, noise, eta=0.0)
    second = sampler.ddim_step(noisy, 40, 20, noise, eta=0.0)
    assert torch.equal(first, second)
    third = sampler.ddim_step(noisy, 40, 20, noise, eta=1.0)
    fourth = sampler.ddim_step(noisy, 40, 20, noise, eta=1.0)
    assert not torch.equal(third, fourth)
