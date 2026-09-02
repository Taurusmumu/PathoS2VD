from __future__ import annotations


import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import wasserstein_distance


EPS = 1e-8


def _tensor_properties(
    volume: np.ndarray,
    *,
    sigma: float = 1.0,
    rho: float = 2.0,
    spacing: tuple[float, float, float] = (0.3, 0.5, 0.5),
) -> dict[str, np.ndarray]:
    """Mirror st_2.py: grayscale [0,1], reflect filtering, (z,y,x) gradients."""
    image = np.asarray(volume, dtype=np.float32)
    if image.ndim != 3 or min(image.shape) < 2:
        raise ValueError(f"Expected [Z,H,W] volume with dimensions >=2, received {image.shape}")
    smoothed = gaussian_filter(image, sigma=sigma)
    iz, iy, ix = np.gradient(smoothed, *spacing)
    jxx = gaussian_filter(ix * ix, sigma=rho)
    jyy = gaussian_filter(iy * iy, sigma=rho)
    jzz = gaussian_filter(iz * iz, sigma=rho)
    jxy = gaussian_filter(ix * iy, sigma=rho)
    jxz = gaussian_filter(ix * iz, sigma=rho)
    jyz = gaussian_filter(iy * iz, sigma=rho)
    tensor = np.zeros(image.shape + (3, 3), dtype=np.float32)
    tensor[..., 0, 0] = jxx
    tensor[..., 1, 1] = jyy
    tensor[..., 2, 2] = jzz
    tensor[..., 0, 1] = tensor[..., 1, 0] = jxy
    tensor[..., 0, 2] = tensor[..., 2, 0] = jxz
    tensor[..., 1, 2] = tensor[..., 2, 1] = jyz
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    lambda1, lambda3 = eigenvalues[..., 0], eigenvalues[..., 2]
    anisotropy = (lambda3 - lambda1) / (lambda3 + EPS)
    return {
        "tensor": tensor,
        "v1": eigenvectors[..., :, 0],
        "anisotropy": anisotropy,
        "iz_abs": np.abs(iz),
    }


def _valid_mask(properties: dict[str, np.ndarray], anis_thresh: float) -> np.ndarray:
    valid = properties["anisotropy"] > anis_thresh
    # Exact paper-code fallback when no oriented GT voxel is available.
    return valid if valid.sum() else np.ones_like(valid, dtype=bool)


def _corrcoef_safe(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) < EPS or np.std(second) < EPS:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def evaluate_st_pair(
    gt_volume: np.ndarray,
    generated_volume: np.ndarray,
    *,
    sigma: float = 1.0,
    rho: float = 2.0,
    spacing: tuple[float, float, float] = (0.3, 0.5, 0.5),
    anis_thresh: float = 0.3,
) -> dict[str, float]:
    """Exact pair metrics from the paper ST script (st_2.evaluate_pair)."""
    gt_image = np.asarray(gt_volume, dtype=np.float32)
    generated_image = np.asarray(generated_volume, dtype=np.float32)
    if gt_image.shape != generated_image.shape:
        raise ValueError(f"ST shape mismatch: {gt_image.shape} vs {generated_image.shape}")
    gt = _tensor_properties(gt_image, sigma=sigma, rho=rho, spacing=spacing)
    generated = _tensor_properties(generated_image, sigma=sigma, rho=rho, spacing=spacing)
    valid = _valid_mask(gt, anis_thresh)
    dot = np.sum(gt["v1"] * generated["v1"], axis=-1)
    tensor_difference = np.linalg.norm(gt["tensor"] - generated["tensor"], axis=(-2, -1))
    gt_tensor_norm = np.linalg.norm(gt["tensor"], axis=(-2, -1)) + EPS
    gt_vz = np.abs(gt["v1"][..., 2])[valid]
    generated_vz = np.abs(generated["v1"][..., 2])[valid]
    gt_evolution = np.mean(np.linalg.norm(np.diff(gt["tensor"], axis=0), axis=(-2, -1)), axis=(1, 2))
    generated_evolution = np.mean(
        np.linalg.norm(np.diff(generated["tensor"], axis=0), axis=(-2, -1)), axis=(1, 2)
    )
    return {
        "orientation_similarity": float(np.mean(np.abs(dot)[valid])),
        "st_frobenius": float(np.mean((tensor_difference / gt_tensor_norm)[valid])),
        "anisotropy_mae": float(np.mean(np.abs(gt["anisotropy"] - generated["anisotropy"])[valid])),
        "vz_wasserstein": float(wasserstein_distance(gt_vz, generated_vz)),
        "st_evolution_corr": _corrcoef_safe(gt_evolution, generated_evolution),
        # Legacy names retained in per-stack CSVs for direct paper-code comparison.
        "tensor_frobenius_rel": float(np.mean((tensor_difference / gt_tensor_norm)[valid])),
        "tensor_evolution_corr": _corrcoef_safe(gt_evolution, generated_evolution),
    }


def vz_descriptor(
    volume: np.ndarray,
    *,
    sigma: float = 1.0,
    rho: float = 2.0,
    spacing: tuple[float, float, float] = (0.3, 0.5, 0.5),
    anis_thresh: float = 0.3,
) -> np.ndarray:
    """Paper Vz descriptor |v1_z| with the same anisotropy validity rule.

    In unpaired target-domain prior drift, each generated volume supplies its
    own validity mask; no source GT is asserted as target-volume ground truth.
    """
    properties = _tensor_properties(volume, sigma=sigma, rho=rho, spacing=spacing)
    return np.abs(properties["v1"][..., 2])[_valid_mask(properties, anis_thresh)].astype(np.float64)


def local_z_axis_descriptors(
    volume: np.ndarray,
    *,
    sigma: float = 1.0,
    rho: float = 2.0,
    spacing: tuple[float, float, float] = (0.3, 0.5, 0.5),
    anis_thresh: float = 0.3,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Descriptors for one local ``[11,H,W]`` volume.

    ``vz`` is the paper-code valid-voxel descriptor.  The other descriptors
    are one spatially averaged observation per adjacent z-plane pair, avoiding
    uncontrolled pooling of raw tensor elements.
    """
    properties = _tensor_properties(volume, sigma=sigma, rho=rho, spacing=spacing)
    valid = _valid_mask(properties, anis_thresh)
    vz_z_index = np.nonzero(valid)[0].astype(np.int64)
    vz = np.abs(properties["v1"][..., 2])[valid].astype(np.float64)

    tensor_change = np.linalg.norm(np.diff(properties["tensor"], axis=0), axis=(-2, -1))
    evolution = np.mean(tensor_change, axis=(1, 2)).astype(np.float64)

    adjacent_valid = valid[:-1] & valid[1:]
    orientation_dot = np.sum(properties["v1"][:-1] * properties["v1"][1:], axis=-1)
    orientation_angle = np.arccos(np.clip(np.abs(orientation_dot), 0.0, 1.0))
    anisotropy_change_voxels = np.abs(np.diff(properties["anisotropy"], axis=0))
    orientation_change: list[float] = []
    anisotropy_change: list[float] = []
    for z_index in range(orientation_angle.shape[0]):
        pair_mask = adjacent_valid[z_index]
        if not pair_mask.any():
            pair_mask = np.ones_like(pair_mask, dtype=bool)
        orientation_change.append(float(np.mean(orientation_angle[z_index][pair_mask])))
        anisotropy_change.append(float(np.mean(anisotropy_change_voxels[z_index][pair_mask])))

    return {
        "vz": (vz_z_index, vz),
        "st_evolution": (np.arange(evolution.size, dtype=np.int64), evolution),
        "orientation_change": (np.arange(len(orientation_change), dtype=np.int64), np.asarray(orientation_change)),
        "anisotropy_change": (np.arange(len(anisotropy_change), dtype=np.int64), np.asarray(anisotropy_change)),
        "st_evolution_roughness": (
            np.arange(max(evolution.size - 1, 0), dtype=np.int64),
            np.abs(np.diff(evolution)).astype(np.float64),
        ),
    }
