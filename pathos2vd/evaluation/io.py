from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import os


GT_FRAME_INDICES = tuple(range(11))
GENERATED_LAYOUTS = {
    "central_33": tuple(range(11, 22)),
    "legacy_11": tuple(range(11)),
}


def stack_frame_indices(layout: str) -> tuple[int, ...]:
    try:
        return GENERATED_LAYOUTS[layout]
    except KeyError as error:
        choices = ", ".join(sorted(GENERATED_LAYOUTS))
        raise ValueError(f"Unknown generated_frame_layout {layout!r}; choose one of {choices}") from error


def _frame_path(directory: Path, index: int) -> Path:
    return directory / f"z{index:02d}.png"


def _load_paths(paths: Iterable[Path], *, grayscale: bool) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = image.convert("L" if grayscale else "RGB")
            arrays.append(np.asarray(image, dtype=np.uint8))
    stack = np.stack(arrays, axis=0)
    if len({tuple(frame.shape) for frame in arrays}) != 1:
        raise ValueError("All stack frames must have the same image shape")
    return stack


def load_gt_stack(directory: str | Path, *, grayscale: bool = False) -> np.ndarray:
    """Read exactly z00--z10; directory listings are never used for ordering."""
    root = Path(directory)
    # return _load_paths((_frame_path(root, index) for index in GT_FRAME_INDICES), grayscale=grayscale)
    paths = sorted(
        (path for path in root.iterdir()
         if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}),
        key=lambda path: path.name,
    )
    if not paths:
        raise FileNotFoundError(f"No image files found in {root}")

    return _load_paths(paths, grayscale=grayscale)



def load_generated_stack(
    directory: str | Path,
    *,
    generated_frame_layout: str = "central_33",
    grayscale: bool = False,
) -> np.ndarray:
    """Read central z11--z21, or explicitly configured legacy z00--z10."""
    root = Path(directory)
    indices = stack_frame_indices(generated_frame_layout)
    return _load_paths((_frame_path(root, index) for index in indices), grayscale=grayscale)


def generated_middle_index(layout: str) -> int:
    """Physical z-file index of the 11-frame middle condition/output frame."""
    return stack_frame_indices(layout)[5]


def generated_sample_dir(
    root: str | Path,
    slide_name: str,
    patch_name: str,
) -> Path:
    return Path(root) / str(slide_name) / Path(str(patch_name)).stem
