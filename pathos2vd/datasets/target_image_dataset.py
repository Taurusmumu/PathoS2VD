from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .zstack_dataset import pathology_transform


class TargetImageDataset(Dataset):
    """Generic target pathology image dataset backed by paths or glob patterns."""

    def __init__(
        self,
        *,
        paths: Sequence[str | Path] | None = None,
        data_root: str | Path | None = None,
        patterns: Sequence[str] = ("**/*.jpg",),
        manifest: dict[str, Any] | None = None,
        image_size: int = 256,
        crop_size: int = 256,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.data_root = Path(data_root) if data_root is not None else None
        if paths is None and manifest is not None:
            if data_root is None:
                raise ValueError("Manifest-backed target data requires data_root")
            paths = self._paths_from_manifest(Path(data_root), manifest)
        if paths is None:
            if data_root is None:
                raise ValueError("Provide paths or data_root")
            root = Path(data_root)
            discovered: list[Path] = []
            for pattern in patterns:
                discovered.extend(Path(path) for path in glob.glob(str(root / pattern), recursive=True))
            paths = sorted(set(discovered))
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("Target image list is empty")
        self.transform = transform or (
            lambda image: pathology_transform(image, crop_size=crop_size, image_size=image_size)
        )

    @staticmethod
    def _paths_from_manifest(data_root: Path, config: dict[str, Any]) -> list[Path]:
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError("Manifest-backed target datasets require pandas") from error
        manifest_path = Path(config["path"])
        table = pd.read_excel(manifest_path) if manifest_path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(manifest_path)
        if config.get("filter_column") is not None:
            table = table.loc[table[config["filter_column"]] == config["filter_value"]]
        values = table[config["path_column"]].astype(str).tolist()
        suffix_contains = config.get("suffix_contains")
        if suffix_contains:
            values = [value for value in values if suffix_contains in value]
        rules = config.get("path_rules", [])
        default_prefix = config.get("default_prefix", "")
        patterns = config.get(
            "directory_patterns",
            ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"],
        )

        paths: list[Path] = []
        missing: list[Path] = []

        for value in values:
            prefix = default_prefix

            for rule in rules:
                if rule["contains"] in value:
                    prefix = rule["prefix"]
                    break

            candidate = data_root / prefix / value

            if candidate.is_file():
                paths.append(candidate)
            elif candidate.is_dir():
                for pattern in patterns:
                    paths.extend(candidate.rglob(pattern))
            else:
                missing.append(candidate)

        if missing:
            examples = ", ".join(str(path) for path in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} manifest paths do not exist. Examples: {examples}"
            )

        paths = sorted(set(paths))

        if not paths:
            raise ValueError(
                f"No images were discovered from manifest {manifest_path}"
            )

        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image)
        return {"pixel_values": tensor, "path": str(path), "name": path.stem}
