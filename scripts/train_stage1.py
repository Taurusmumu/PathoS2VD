#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathos2vd.trainers.stage1_trainer import Stage1Trainer
from pathos2vd.utils import load_config
from pathos2vd.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PathoS2VD Stage-I z prior")
    parser.add_argument("--config", default="configs/stage1/diffusion.yaml")
    args = parser.parse_args()
    configure_logging()
    Stage1Trainer(load_config(args.config)).fit()


if __name__ == "__main__":
    main()
