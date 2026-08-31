#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from pathos2vd.pipelines.pathos2vd_pipeline import generate_33_planes, load_pathos2vd_pipeline
from pathos2vd.utils import load_config
from pathos2vd.utils.profiling import summarize_timings


def _image_paths(value: str, count: int) -> list[Path]:
    path = Path(value)
    if path.is_file():
        return [path] * count
    if path.is_dir():
        candidates = sorted(
            item for item in path.iterdir() if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
        if len(candidates) < count:
            raise ValueError(f"Need {count} images, found {len(candidates)} in {path}")
        return candidates[:count]
    raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile 33-plane PathoS2VD inference")
    parser.add_argument("--config", default="configs/inference/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for synchronized inference profiling")
    profile = config.get("profiling", {})
    warmup = int(profile.get("warmup_images", 1))
    measured = int(profile.get("measured_images", 10))
    paths = _image_paths(config["inference"]["input"], warmup + measured)
    device = torch.device("cuda")
    pipeline = load_pathos2vd_pipeline(config, device)
    for path in paths[:warmup]:
        with Image.open(path) as image:
            generate_33_planes(pipeline, image, config, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    for path in paths[warmup:]:
        with Image.open(path) as image:
            torch.cuda.synchronize()
            start = time.perf_counter()
            frames = generate_33_planes(pipeline, image, config, device)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        if len(frames) != 33:
            raise RuntimeError("Profiling received a non-33-plane output")
    report = summarize_timings(timings).to_dict()
    report.update(
        {
            "stacks_measured": measured,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_model": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "visible_cuda_devices": torch.cuda.device_count(),
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
