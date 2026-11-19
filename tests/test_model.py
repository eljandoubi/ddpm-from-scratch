import torch
from PIL import Image

from model import Diffusion, build_model
from sample import generate
from sampler import Sampler
from utils import image_transform


def tiny_model():
    return Diffusion(
        in_channels=3,
        start_dim=16,
        dim_mults=(1, 2),
        residual_blocks_per_group=1,
        groupnorm_num_groups=4,
        time_embed_dim=16,
    )


def test_unet_keeps_the_image_shape():
    model = tiny_model()
    images = torch.randn(2, 3, 16, 16)
    timesteps = torch.randint(0, 1000, (2,))
    assert model(images, timesteps).shape == images.shape


def test_gradients_reach_the_parameters():
    model = tiny_model()
    images = torch.randn(2, 3, 16, 16)
    loss = model(images, torch.randint(0, 1000, (2,))).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_build_model_uses_four_resolution_levels():
    model = build_model(16)
    assert model.dim_mults == (1, 2, 3, 4)
    assert model(torch.randn(1, 3, 32, 32), torch.tensor([5])).shape == (1, 3, 32, 32)


def test_image_transform_crops_to_a_square_in_minus_one_to_one():
    image = Image.new("RGB", (178, 218), color=(255, 0, 128))
    tensor = image_transform(16)(image)
    assert tensor.shape == (3, 16, 16)
    assert tensor.min() >= -1 and tensor.max() <= 1


def test_generate_returns_clipped_images_for_both_samplers():
    torch.manual_seed(0)
    model = tiny_model().eval()
    sampler = Sampler(total_timesteps=10, device=torch.device("cpu"))
    for method in ("ddpm", "ddim"):
        images = generate(model, sampler, 3, 16, method=method, steps=4, batch_size=2)
        assert images.shape == (3, 3, 16, 16)
        assert images.min() >= -1 and images.max() <= 1
