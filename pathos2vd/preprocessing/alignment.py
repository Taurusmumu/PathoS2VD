from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def align_to_reference(reference_bgr: np.ndarray, moving_bgr: np.ndarray, *, max_iterations: int = 1000, epsilon: float = 1e-7) -> tuple[np.ndarray, bool]:
    """Historical ECC translation alignment; failure intentionally means identity warp."""
    reference = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    moving = cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2GRAY)
    matrix = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(reference, moving, matrix, cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon))
        succeeded = True
    except cv2.error:
        succeeded = False
    height, width = moving.shape
    return cv2.warpAffine(moving_bgr, matrix, (width, height), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP), succeeded


