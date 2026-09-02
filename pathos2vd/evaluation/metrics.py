from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _as_float_rgb(stack: np.ndarray) -> np.ndarray:
    array = np.asarray(stack, dtype=np.float32)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected [Z,H,W,3] uint8 image stack, received {array.shape}")
    return array / 255.0


def legacy_ssim(first: np.ndarray, second: np.ndarray) -> float:
    return float(structural_similarity(first, second, multichannel=True, channel_axis=2, data_range=1.0))


def legacy_psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = np.mean((first - second) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * math.log10(1.0 / math.sqrt(mse)))


class LegacyImageMetrics:
    """LPIPS-VGG plus SSIM/PSNR and legacy temporal/middle-slice definitions."""

    def __init__(self, *, device: str | torch.device = "cpu", use_lpips: bool = True) -> None:
        self.device = torch.device(device)
        self.lpips_fn = None
        if use_lpips:
            try:
                import lpips
            except ImportError as error:
                raise RuntimeError("LPIPS was requested but the lpips package is not installed") from error
            self.lpips_fn = lpips.LPIPS(net="vgg").to(self.device).eval()

    def _lpips(self, first: np.ndarray, second: np.ndarray) -> float:
        if self.lpips_fn is None:
            return float("nan")
        first_tensor = torch.from_numpy(first).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        second_tensor = torch.from_numpy(second).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        first_tensor = first_tensor * 2.0 - 1.0
        second_tensor = second_tensor * 2.0 - 1.0
        with torch.no_grad():
            return float(self.lpips_fn(first_tensor, second_tensor).item())

    @staticmethod
    def temporal_flickering(stack: np.ndarray) -> float:
        """Exact VBench helper: (255 - mean adjacent MAE) / 255."""
        image = np.asarray(stack, dtype=np.float32)
        if image.shape[0] < 2:
            return float("nan")
        return float((255.0 - np.mean(np.abs(image[:-1] - image[1:]))) / 255.0)

    def paired_stack(self, gt_stack: np.ndarray, generated_stack: np.ndarray) -> tuple[list[dict[str, float]], dict[str, float]]:
        gt = _as_float_rgb(gt_stack)
        generated = _as_float_rgb(generated_stack)
        if gt.shape != generated.shape:
            raise ValueError(f"Image stack shape mismatch: {gt.shape} vs {generated.shape}")
        per_frame: list[dict[str, float]] = []
        for index in range(gt.shape[0]):
            per_frame.append(
                {
                    "frame_offset": index,
                    "ssim": legacy_ssim(gt[index], generated[index]),
                    "psnr": legacy_psnr(gt[index], generated[index]),
                    "lpips": self._lpips(gt[index], generated[index]),
                }
            )
        # *_ver is the original comparison of adjacent-slice metric curves.
        gt_ssim = [legacy_ssim(gt[index - 1], gt[index]) for index in range(1, gt.shape[0])]
        generated_ssim = [legacy_ssim(generated[index - 1], generated[index]) for index in range(1, gt.shape[0])]
        gt_psnr = [legacy_psnr(gt[index - 1], gt[index]) for index in range(1, gt.shape[0])]
        generated_psnr = [legacy_psnr(generated[index - 1], generated[index]) for index in range(1, gt.shape[0])]
        gt_lpips = [self._lpips(gt[index - 1], gt[index]) for index in range(1, gt.shape[0])]
        generated_lpips = [self._lpips(generated[index - 1], generated[index]) for index in range(1, gt.shape[0])]
        middle = gt.shape[0] // 2
        summary = {
            "ssim": float(np.mean([row["ssim"] for row in per_frame])),
            "psnr": float(np.mean([row["psnr"] for row in per_frame])),
            "lpips": float(np.mean([row["lpips"] for row in per_frame])),
            "ssim_ver": float(np.mean(np.abs(np.asarray(gt_ssim) - np.asarray(generated_ssim)))),
            "psnr_ver": float(np.mean(np.abs(np.asarray(gt_psnr) - np.asarray(generated_psnr)))),
            "lpips_ver": float(np.mean(np.abs(np.asarray(gt_lpips) - np.asarray(generated_lpips)))),
            "temporal_flickering": self.temporal_flickering(generated_stack),
            "gt_temporal_flickering": self.temporal_flickering(gt_stack),
        }
        # Original ssim_m/psnr_m/lpips_m compares the GT middle slice to every generated frame.
        for index, row in enumerate(per_frame):
            row["ssim_m"] = legacy_ssim(gt[middle], generated[index])
            row["psnr_m"] = legacy_psnr(gt[middle], generated[index])
            row["lpips_m"] = self._lpips(gt[middle], generated[index])
        return per_frame, summary

    def target_central_frame(self, input_image: np.ndarray, generated_central: np.ndarray) -> dict[str, float]:
        """Table-2 fidelity: input image against central generated z16 only."""
        input_stack = _as_float_rgb(np.asarray(input_image)[None, ...])
        generated_stack = _as_float_rgb(np.asarray(generated_central)[None, ...])
        if input_stack.shape != generated_stack.shape:
            raise ValueError(f"Target image shape mismatch: {input_stack.shape} vs {generated_stack.shape}")
        return {
            "ssim": legacy_ssim(input_stack[0], generated_stack[0]),
            "lpips": self._lpips(input_stack[0], generated_stack[0]),
        }


def mean_rows(rows: Iterable[dict[str, float]], columns: Iterable[str]) -> dict[str, float]:
    materialized = list(rows)
    return {
        column: float(np.nanmean([row[column] for row in materialized])) if materialized else float("nan")
        for column in columns
    }
