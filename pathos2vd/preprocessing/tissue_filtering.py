from __future__ import annotations

import cv2
import numpy as np


def tissue_fraction(path: str, *, low_s: int = 50, high_v: int = 200) -> float:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    background = cv2.inRange(hsv, np.array([0, 0, high_v]), np.array([179, low_s, 255]))
    return float(cv2.countNonZero(cv2.bitwise_not(background)) / (image.shape[0] * image.shape[1]))
