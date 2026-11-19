import argparse
import os

import torch
from safetensors.torch import load_file
from torchvision.utils import make_grid, save_image

from model import build_model
from sampler import Sampler


def load_model(path_to_checkpoint: str, starting_channels: int, device: torch.device):
    model = build_model(starting_channels)
    state = load_file(os.path.join(path_to_checkpoint, "model.safetensors"))
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def generate(
    model,
    sampler: Sampler,
    num_images: int,
    image_size: int,
    method: str = "ddpm",
    steps: int = 100,
    eta: float = 0.0,
    batch_size: int = 100,
    fp16: bool = False,
    clip_x0: bool = True,
):
    ### Images in [-1, 1], from ancestral DDPM (every timestep) or strided DDIM ###
    device = sampler.device
    total = sampler.total_timesteps
    if method == "ddpm":
        schedule = list(range(total - 1, -1, -1))
    else:
        schedule = torch.linspace(total - 1, 0, steps).round().long().tolist()

    batches = []
    for start in range(0, num_images, batch_size):
        n = min(batch_size, num_images - start)
        images = torch.randn(n, 3, image_size, image_size, device=device)
        for i, t in enumerate(schedule):
            timesteps = torch.full((n,), t, device=device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                noise_pred = model(images, timesteps)
            noise_pred = noise_pred.float()
            if method == "ddpm":
                images = sampler.remove_noise(
                    images, timesteps, noise_pred, clip_x0=clip_x0
                )
            else:
                prev = schedule[i + 1] if i + 1 < len(schedule) else -1
                images = sampler.ddim_step(
                    images, t, prev, noise_pred, eta, clip_x0=clip_x0
                )
        batches.append(images.clamp(-1, 1).cpu())
    return torch.cat(batches)


def save_images(images: torch.Tensor, directory: str):
    os.makedirs(directory, exist_ok=True)
    for idx, image in enumerate(images):
        save_image((image + 1) / 2, os.path.join(directory, f"{idx:05d}.png"))


def main():
    parser = argparse.ArgumentParser(description="Sample images from a DDPM checkpoint")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Folder such as work/base/checkpoint_10000",
    )
    parser.add_argument("--starting_channels", default=224, type=int)
    parser.add_argument("--img_size", default=192, type=int)
    parser.add_argument("--num_images", default=64, type=int)
    parser.add_argument("--method", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--steps", default=100, type=int, help="Number of DDIM steps")
    parser.add_argument(
        "--eta", default=0.0, type=float, help="DDIM noise, 0 is deterministic"
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--fp16", action="store_true", help="Run the network in fp16")
    parser.add_argument(
        "--no_clip_x0",
        dest="clip_x0",
        action="store_false",
        help="Do not clip the clean image estimate of each step to [-1, 1]",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--grid_out", default="samples.png", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, args.starting_channels, device)
    sampler = Sampler(device=device)
    images = generate(
        model,
        sampler,
        args.num_images,
        args.img_size,
        method=args.method,
        steps=args.steps,
        eta=args.eta,
        batch_size=args.batch_size,
        fp16=args.fp16,
        clip_x0=args.clip_x0,
    )
    save_image(make_grid((images + 1) / 2, nrow=8), args.grid_out)
    print(f"Wrote {args.num_images} images to {args.grid_out}")


if __name__ == "__main__":
    main()
