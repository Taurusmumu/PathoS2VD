#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathos2vd.trainers.stage2_trainer import Stage2Trainer
from pathos2vd.utils import load_config
from pathos2vd.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prior-preserving Stage-II fine-tuning")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    configure_logging()
    Stage2Trainer(load_config(args.config)).fit()


if __name__ == "__main__":
    main()
