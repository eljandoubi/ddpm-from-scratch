import os

from PIL import Image

from train import parse_args, trainer
from utils import FolderDataset, image_transform


def make_images(folder, count=4):
    folder.mkdir()
    for i in range(count):
        Image.new("RGB", (178, 218), color=(40 * i, 80, 120)).save(folder / f"{i}.png")


def test_folder_dataset_reads_images_below_the_folder(tmp_path):
    make_images(tmp_path / "images", count=3)
    dataset = FolderDataset(str(tmp_path / "images"), image_transform(16))
    assert len(dataset) == 3
    image, label = dataset[0]
    assert image.shape == (3, 16, 16) and label == 0


def test_trainer_runs_and_keeps_only_the_latest_checkpoints(tmp_path):
    make_images(tmp_path / "images")
    args = parse_args(
        [
            "--experiment_name",
            "smoke",
            "--working_directory",
            str(tmp_path / "work"),
            "--generated_directory",
            str(tmp_path / "gen"),
            "--image_folder",
            str(tmp_path / "images"),
            "--img_size",
            "16",
            "--starting_channels",
            "16",
            "--batch_size",
            "2",
            "--gradient_accumulation_steps",
            "1",
            "--num_workers",
            "0",
            "--num_training_steps",
            "4",
            "--warmup_steps",
            "1",
            "--evaluation_interval",
            "2",
            "--num_keep_checkpoints",
            "1",
            "--num_diffusion_timesteps",
            "10",
            "--plot_freq_interval",
            "5",
            "--num_generations",
            "2",
        ]
    )
    trainer(args)

    experiment = tmp_path / "work" / "smoke"
    checkpoints = sorted(
        p.name for p in experiment.iterdir() if p.name.startswith("checkpoint_")
    )
    assert checkpoints == ["checkpoint_4"]
    assert os.path.exists(tmp_path / "gen" / "step_4.png")
