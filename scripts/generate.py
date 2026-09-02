#!/usr/bin/env python3
from pathlib import Path
import argparse
import statistics
import sys
import time

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


def _benchmark_dataset_generation(
    dataset: TargetImageDataset,
    pipeline,
    config: dict,
    device: torch.device,
    *,
    start_index: int,
    warmup_samples: int,
    measured_samples: int,
    generate,
) -> None:
    if device.type != "cuda":
        raise RuntimeError("Inference benchmarking requires CUDA for synchronized timing")
    if warmup_samples < 0 or measured_samples <= 0:
        raise ValueError("Benchmark warm-up must be non-negative and measured samples must be positive")
    stop = start_index + warmup_samples + measured_samples
    if start_index < 0 or stop > len(dataset):
        raise ValueError(
            f"Benchmark needs {warmup_samples + measured_samples} images starting at {start_index}, "
            f"but the dataset contains {len(dataset)} images"
        )

    for index in range(start_index, start_index + warmup_samples):
        image = _dataset_item_to_image(dataset[index])
        frames = generate(pipeline, image, config, device)
        if len(frames) != 33:
            raise RuntimeError(f"Expected 33 output planes, received {len(frames)}")

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    final_frames = 0
    for index in range(start_index + warmup_samples, stop):
        image = _dataset_item_to_image(dataset[index])
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        frames = generate(pipeline, image, config, device)
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
        final_frames = len(frames)
        if final_frames != 33:
            raise RuntimeError(f"Expected 33 output planes, received {final_frames}")

    inference = config["inference"]
    mean_seconds = statistics.mean(timings)
    std_seconds = statistics.stdev(timings) if len(timings) > 1 else 0.0
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / float(1024**3)
    print(
        "\n===== Inference Computational Cost =====\n"
        f"GPU model: {torch.cuda.get_device_name(device)}\n"
        "Batch size: 1\n"
        f"Input resolution: {inference.get('image_size', 256)} x {inference.get('image_size', 256)}\n"
        f"Generated frames: {final_frames}\n"
        f"Diffusion sampling steps: {inference.get('num_inference_steps', 20)}\n"
        f"Warm-up patches: {warmup_samples}\n"
        f"Measured patches: {measured_samples}\n"
        f"Mean generation time / patch: {mean_seconds:.3f} s\n"
        f"Std generation time / patch: {std_seconds:.3f} s\n"
        f"Peak allocated GPU memory: {peak_allocated_gib:.2f} GiB\n"
        "========================================="
    )


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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Benchmark dataset generation without saving outputs.",
    )
    parser.add_argument("--benchmark-warmup", type=int, help="Override profiling.warmup_images.")
    parser.add_argument("--benchmark-samples", type=int, help="Override profiling.measured_images.")
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.benchmark and args.dataset_config is None:
        parser.error("--benchmark requires --dataset-config")
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
    if args.benchmark:
        profile = config.get("profiling", {})
        warmup_samples = profile.get("warmup_images", 10) if args.benchmark_warmup is None else args.benchmark_warmup
        measured_samples = profile.get("measured_images", 100) if args.benchmark_samples is None else args.benchmark_samples
        _benchmark_dataset_generation(
            dataset,
            pipeline,
            config,
            device,
            start_index=args.start_index,
            warmup_samples=int(warmup_samples),
            measured_samples=int(measured_samples),
            generate=generate_33_planes,
        )
        return
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
    main()
