import argparse
import json
import os
import time

import torch
import torch_fidelity

from sample import generate, load_model, save_images
from sampler import Sampler
from utils import add_data_arguments, image_transform, load_celeba


def main():
    parser = argparse.ArgumentParser(
        description="FID of DDPM samples against real images"
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--starting_channels", default=224, type=int)
    parser.add_argument("--img_size", default=192, type=int)
    parser.add_argument("--num_images", default=5000, type=int)
    parser.add_argument("--method", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--steps", default=100, type=int, help="Number of DDIM steps")
    parser.add_argument("--eta", default=0.0, type=float)
    parser.add_argument("--batch_size", default=250, type=int)
    parser.add_argument("--fp16", action="store_true", help="Run the network in fp16")
    add_data_arguments(parser)
    parser.add_argument("--work_dir", default="fid_work", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--result_file", default="fid.json", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, args.starting_channels, device)
    sampler = Sampler(device=device)

    start = time.time()
    generated = generate(
        model,
        sampler,
        args.num_images,
        args.img_size,
        method=args.method,
        steps=args.steps,
        eta=args.eta,
        batch_size=args.batch_size,
        fp16=args.fp16,
    )
    sampling_seconds = time.time() - start
    generated_dir = os.path.join(args.work_dir, "generated")
    save_images(generated, generated_dir)

    ### The reference set is a random subset of the training images, same preprocessing ###
    dataset = load_celeba(args, image_transform(args.img_size))
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(args.seed)
    )
    real = torch.stack([dataset[i][0] for i in order[: args.num_images].tolist()])
    real_dir = os.path.join(args.work_dir, "real")
    save_images(real, real_dir)

    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        cuda=device.type == "cuda",
        fid=True,
        verbose=False,
    )
    result = {
        "fid": metrics["frechet_inception_distance"],
        "num_images": args.num_images,
        "method": args.method,
        "steps": 1000 if args.method == "ddpm" else args.steps,
        "eta": args.eta,
        "checkpoint": args.checkpoint,
        "img_size": args.img_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "sampling_seconds": round(sampling_seconds, 1),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    with open(args.result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
