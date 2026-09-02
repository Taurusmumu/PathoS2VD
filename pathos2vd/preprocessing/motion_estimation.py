from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def farneback_motion(reference_bgr: np.ndarray, moving_bgr: np.ndarray) -> float:
    reference = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    moving = cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(reference, moving, None, 0.5, 3, 8, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude[np.isinf(magnitude)] = 0
    return float(np.mean(magnitude))


def estimate_motion_from_frames(reference_bgr: np.ndarray, frames: list[np.ndarray | None]) -> list[float]:
    """Final paper Farneback score for already ECC-aligned in-memory frames."""
    values: list[float] = []
    for frame in frames:
        if frame is None:
            values.append(-1.0)
            continue
        try:
            values.append(farneback_motion(reference_bgr, frame))
        except cv2.error:
            values.append(-1.0)
    return values


