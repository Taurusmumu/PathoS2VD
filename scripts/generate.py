#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from PIL import Image

from pathos2vd.datasets import TargetImageDataset
from pathos2vd.utils import load_config
from pathos2vd.utils.config import expand_config


def _load_generation_config(path: str | Path, *, dataset_mode: bool) -> dict:
    if not dataset_mode:
        return load_config(path)
    # Dataset generation has no single-image input, so do not require the
    # inference.input environment variable merely to load the model settings.
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("inference"), dict):
        raise ValueError(f"Expected a top-level inference mapping in {path}")
    config = dict(config)
    config["inference"] = dict(config["inference"])
    config["inference"].pop("input", None)
    return expand_config(config)


def _load_dataset_section(path: str | Path) -> dict:
    """Load only target-dataset settings, avoiding unrelated training env vars."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError(f"Expected a top-level dataset mapping in {path}")
    data = config["dataset"]
    keys = {"target_data_root", "target_patterns", "target_manifest", "image_size", "target_crop_size"}
    return expand_config({key: value for key, value in data.items() if key in keys})


def _load_target_dataset(path: str | Path) -> tuple[TargetImageDataset, Path]:
    data = _load_dataset_section(path)
    if "target_data_root" not in data:
        raise ValueError("Dataset config must define dataset.target_data_root")
    data_root = Path(data["target_data_root"])
    dataset = TargetImageDataset(
        data_root=data["target_data_root"],
        patterns=data.get("target_patterns", ["**/*.jpg"]),
        manifest=data.get("target_manifest"),
        image_size=data.get("image_size", 256),
        crop_size=data.get("target_crop_size", 256),
    )
    return dataset, data_root


def _dataset_item_to_image(item: dict) -> Image.Image:
    pixels = item["pixel_values"]
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError("TargetImageDataset must return pixel_values with shape [3, H, W]")
    array = (
        ((pixels.detach().cpu().float() + 1.0) * 127.5)
        .clamp(0, 255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _dataset_output_dir(output_root: str | Path, data_root: str | Path, image_path: str | Path, index: int) -> Path:
    image_path = Path(image_path)
    try:
        relative = image_path.relative_to(Path(data_root))
    except ValueError:
        relative = Path(f"{index:06d}_{image_path.name}")
    return Path(output_root) / relative.parent / relative.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 33-plane PathoS2VD pseudo z-stack")
    parser.add_argument("--config", default="/home/compu/jiamu/PathoS2VD/configs/inference/default.yaml")
    parser.add_argument(
        "--dataset-config",
        help="Optional Stage-II target config; generate one stack for every TargetImageDataset image.",
        default="/home/compu/jiamu/PathoS2VD/configs/stage2/aggc.yaml"
    )
    parser.add_argument("--start-index", type=int, default=0, help="First dataset image to generate.")
    parser.add_argument("--max-images", type=int, default=3, help="Maximum number of dataset images to generate.")
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    config = _load_generation_config(args.config, dataset_mode=args.dataset_config is not None)
    from pathos2vd.pipelines.pathos2vd_pipeline import (
        generate_33_planes,
        load_pathos2vd_pipeline,
        save_plane_stack,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = load_pathos2vd_pipeline(config, device)

    if args.dataset_config is None:
        input_path = Path(config["inference"]["input"])
        if not input_path.is_file():
            raise FileNotFoundError("inference.input must identify one RGB image")
        with Image.open(input_path) as image:
            frames = generate_33_planes(pipeline, image, config, device)
        save_plane_stack(frames, config["inference"]["output_dir"])
        return

    dataset, data_root = _load_target_dataset(args.dataset_config)
    stop = len(dataset) if args.max_images is None else min(len(dataset), args.start_index + args.max_images)
    if args.start_index >= stop:
        raise ValueError(f"Selected dataset range is empty: start={args.start_index}, size={len(dataset)}")
    output_root = Path(config["inference"]["output_dir"])
    for index in range(args.start_index, stop):
        item = dataset[index]
        image = _dataset_item_to_image(item)
        frames = generate_33_planes(pipeline, image, config, device)
        output_dir = _dataset_output_dir(output_root, data_root, item["path"], index)
        save_plane_stack(frames, output_dir)
        print(f"[{index + 1}/{stop}] {item['path']} -> {output_dir}")


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    main()
