from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_relative_blur(table: pd.DataFrame, plane_columns: list[str], *, threshold: float = 0.2, min_sharp_planes: int = 5, absolute_min_low: float = 0.6, absolute_min_high: float = 1.0, std_multiplier: float = 2.0) -> pd.DataFrame:
    """Faithful active filtering path in ``blur_data_filter.py``."""
    absolute = table.loc[:, plane_columns].to_numpy(dtype=float)
    minima = np.min(absolute, axis=1)
    calibration = minima[(minima >= absolute_min_low) & (minima <= absolute_min_high)]
    if not len(calibration):
        raise ValueError("No stacks fall in the historical absolute-blur calibration range")
    mean, std = float(np.mean(calibration)), float(np.std(calibration))
    retained = (minima >= mean - std_multiplier * std) & (minima <= mean + std_multiplier * std)
    result = table.loc[retained].copy()
    absolute = result.loc[:, plane_columns].to_numpy(dtype=float)
    minima = np.min(absolute, axis=1)
    relative = np.sqrt(np.maximum(absolute ** 2 - minima[:, None] ** 2, 0.0))
    sharp = relative < threshold
    result["min_index"] = np.argmin(relative, axis=1)
    result["start"] = result["min_index"] - 5
    result["end"] = result["min_index"] + 5
    result["keep_blur"] = sharp.sum(axis=1) >= min_sharp_planes
    for index, plane in enumerate(plane_columns):
        result[f"absolute_blur_{plane}"] = absolute[:, index]
        result[f"relative_blur_{plane}"] = relative[:, index]
        result[f"valid_blur_{plane}"] = sharp[:, index]
        result[plane] = relative[:, index]  # compatibility before motion combines fields
    result["blur_calibration_mean"] = mean
    result["blur_calibration_std"] = std
    return result.loc[result["keep_blur"]].copy()
