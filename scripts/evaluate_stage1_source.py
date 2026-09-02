#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathos2vd.evaluation.runners import evaluate_source_root
from pathos2vd.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate central PathoS2VD stacks against source GT z-stacks")
    parser.add_argument("--config", default="/home/compu/jiamu/PathoS2VD/configs/evaluation/stage1_source.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate_source_root(config, config["generated_root"], config["output_dir"])


if __name__ == "__main__":
    main()
