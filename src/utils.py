import os
import argparse
import itertools
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.nn import Module
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from sampler import Sampler


class FolderDataset(Dataset):
    """Every image below a folder, yielding (image, 0) like CelebA does."""

    def __init__(self, folder: str, transform):
        suffixes = {".png", ".jpg", ".jpeg"}
        self.paths = sorted(
            p for p in Path(folder).rglob("*") if p.suffix.lower() in suffixes
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        return self.transform(Image.open(self.paths[idx]).convert("RGB")), 0


class HFImageDataset(Dataset):
    """A Hugging Face image dataset that yields (image, 0), like CelebA does."""

    def __init__(self, name: str, transform):
        from datasets import load_dataset

        self.data = load_dataset(name, split="train")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.transform(self.data[idx]["image"].convert("RGB")), 0


def image_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags that say where the training images come from."""
    parser.add_argument(
        "--path_to_data", help="Path to CelebA Dataset", default="data", type=str
    )
    parser.add_argument(
        "--hf_dataset",
        help="Hugging Face image dataset to use instead of torchvision CelebA, e.g. nielsr/CelebA-faces",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--image_folder",
        help="Folder of images to use instead of CelebA",
        default=None,
        type=str,
    )


def load_celeba(args: argparse.Namespace, transform):
    if args.image_folder is not None:
        return FolderDataset(args.image_folder, transform)
    ### torchvision downloads CelebA from Google Drive, which often hits its quota ###
    if args.hf_dataset is not None:
        return HFImageDataset(args.hf_dataset, transform)
    return CelebA(
        args.path_to_data, download=True, transform=transform, target_type="identity"
    )


@torch.no_grad()
def sample_plot_image(
    step_idx: int,
    total_timesteps: int,
    image_size: int,
    num_channels: int,
    plot_freq: int,
    model: Module,
    num_gens: int,
    path_to_generated_dir: str,
    sampler: Sampler,
):
    ### Conver Tensor back to Image (From Huggingface Annotated Diffusion) ###
    tensor2image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda t: t.squeeze(0))
            if num_channels == 3
            else nn.Identity(),
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    images = torch.randn(
        (num_gens, num_channels, image_size, image_size), device=sampler.device
    )
    num_images_per_gen = total_timesteps // plot_freq

    images_to_vis = [[] for _ in range(num_gens)]
    for t in range(total_timesteps - 1, -1, -1):
        ts = torch.full((num_gens,), t, device=sampler.device)
        noise_pred = model(images, ts)
        images = sampler.remove_noise(images, ts, noise_pred)
        if t % plot_freq == 0:
            for idx, image in enumerate(images):
                images_to_vis[idx].append(tensor2image_transform(image))

    images_to_vis = list(itertools.chain(*images_to_vis))

    fig, axes = plt.subplots(
        nrows=num_gens, ncols=num_images_per_gen, figsize=(num_images_per_gen, num_gens)
    )
    plt.tight_layout()
    for ax, image in zip(axes.ravel(), images_to_vis):
        ax.imshow(image)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(path_to_generated_dir, f"step_{step_idx}.png"))

    return fig
