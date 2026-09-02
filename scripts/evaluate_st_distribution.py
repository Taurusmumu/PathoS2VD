#!/usr/bin/env python3
"""Unpaired dataset-level GT-versus-generated z-axis ST distribution evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathos2vd.evaluation.runners import evaluate_st_distribution
from pathos2vd.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare unpaired GT and generated z-axis ST distributions")
    parser.add_argument("--config", default="/home/compu/jiamu/PathoS2VD/configs/evaluation/st_distribution.yaml")
    parser.add_argument("--sampling-manifest", help="Reuse stack IDs and local crop coordinates from a prior local run")
    parser.add_argument("--save-sampling-manifest", help="Also save this run's sampling manifest at this path")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.sampling_manifest:
        config["sampling_manifest"] = args.sampling_manifest
    if args.save_sampling_manifest:
        config["save_sampling_manifest"] = args.save_sampling_manifest
    evaluate_st_distribution(config)


if __name__ == "__main__":
    main()
