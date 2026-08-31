from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_accelerator(config: dict):
    try:
        from accelerate import Accelerator
    except ImportError as error:
        raise RuntimeError("Training requires the 'accelerate' package") from error
    training = config["training"]
    return Accelerator(
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 1),
        mixed_precision=training.get("mixed_precision", "no"),
        log_with=config.get("logging", {}).get("report_to"),
        project_dir=config.get("logging", {}).get("logging_dir"),
    )
