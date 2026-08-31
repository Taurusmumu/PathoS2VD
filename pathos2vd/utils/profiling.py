from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Iterable

import torch


def count_parameters(module: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad),
    }


@dataclass
class TimingSummary:
    mean_seconds: float
    median_seconds: float
    q1_seconds: float
    q3_seconds: float
    iqr_seconds: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize_timings(values: Iterable[float]) -> TimingSummary:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("At least one timing is required")

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    q1, q3 = percentile(0.25), percentile(0.75)
    return TimingSummary(mean(ordered), median(ordered), q1, q3, q3 - q1)
