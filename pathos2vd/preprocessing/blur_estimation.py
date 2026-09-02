from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage
from skimage import feature


def _g1x(x: np.ndarray, y: np.ndarray, std: float) -> np.ndarray:
    squared = std ** 2
    return -x / (2 * np.pi * squared ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * squared))


def _g1y(x: np.ndarray, y: np.ndarray, std: float) -> np.ndarray:
    squared = std ** 2
    return -y / (2 * np.pi * squared ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * squared))


def estimate_sparse_blur(gray: np.ndarray, edges: np.ndarray, *, std1: float, std2: float, half_window: int = 11, max_blur: float = 5.0) -> np.ndarray:
    """Exact active sparse-map branch from ``defocus_estimate.py``."""
    axis = np.arange(-half_window, half_window + 1)
    xmesh = np.tile(axis, (axis.size, 1))
    ymesh = xmesh.T
    first_x, first_y = _g1x(xmesh, ymesh, std1), _g1y(xmesh, ymesh, std1)
    second_x, second_y = _g1x(xmesh, ymesh, std2), _g1y(xmesh, ymesh, std2)
    magnitude1 = np.hypot(scipy.ndimage.convolve(gray, first_x, mode="nearest"), scipy.ndimage.convolve(gray, first_y, mode="nearest"))
    magnitude2 = np.hypot(scipy.ndimage.convolve(gray, second_x, mode="nearest"), scipy.ndimage.convolve(gray, second_y, mode="nearest"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (magnitude1 / magnitude2) * (edges > 0)
        values = (ratio ** 2 * std1 ** 2 - std2 ** 2) / (1 - ratio ** 2)
        values[values < 0] = 0
        result = np.sqrt(values)
    result[np.isnan(result)] = 0
    result[result > max_blur] = max_blur
    return result


def estimate_blur_values(image_bgr: np.ndarray, *, sigma_c: float = 1.0, std1: float = 1.0, std2: float = 1.5, half_window: int = 11, max_blur: float = 5.0) -> np.ndarray:
    """Return ``np.unique(sparse_bmap)`` exactly as the historical active code."""
    if image_bgr is None:
        raise ValueError("Unreadable image")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) / 255.0
    edges = feature.canny(gray, sigma_c)
    return np.unique(estimate_sparse_blur(gray, edges, std1=std1, std2=std2, half_window=half_window, max_blur=max_blur))


def estimate_blur_score(path: str, **kwargs: float) -> float:
    return float(np.mean(estimate_blur_values(cv2.imread(path), **kwargs)))
