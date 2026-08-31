from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PLANE_NAMES = tuple(f"z{index:02d}" for index in range(19))


def pathology_transform(image: Image.Image, *, crop_size: int = 253, image_size: int = 256) -> torch.Tensor:
    """Center crop, bicubic resize, RGB conversion, and [-1, 1] normalization."""
    image = image.convert("RGB")
    width, height = image.size
    left = max((width - crop_size) // 2, 0)
    top = max((height - crop_size) // 2, 0)
    image = image.crop((left, top, min(left + crop_size, width), min(top + crop_size, height)))
    bicubic = getattr(Image, "Resampling", Image).BICUBIC
    image = image.resize((image_size, image_size), bicubic)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


@dataclass(frozen=True)
class ZStackRecord:
    slide_name: str
    patch_name: str
    start_index: int
    min_index: int
    motion: tuple[float, ...]
    blur: tuple[float, ...]


def _read_split_file(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    assignments: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            fields = raw_line.strip().split(",")
            if len(fields) >= 3:
                assignments[fields[0]] = fields[2]
    return assignments


def _parse_score(value: str) -> tuple[float, float]:
    fields = value.split(";")
    if len(fields) != 2:
        raise ValueError(f"Expected 'motion;blur', got {value!r}")
    return float(fields[0]), float(fields[1])


def read_annotation(path: str | Path) -> list[ZStackRecord]:
    records: list[ZStackRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Annotation has no header: {path}")
        score_columns = [name for name in PLANE_NAMES if name in reader.fieldnames]
        if len(score_columns) != 19:
            # Historical files always place 19 motion;blur fields between IDs and the final indices.
            score_columns = reader.fieldnames[2:-3]
        if len(score_columns) != 19:
            raise ValueError(f"Expected 19 plane score columns, found {len(score_columns)}")
        start_key, min_key = reader.fieldnames[-3], reader.fieldnames[-1]
        for row in reader:
            scores = [_parse_score(row[name]) for name in score_columns]
            records.append(
                ZStackRecord(
                    slide_name=row[reader.fieldnames[0]],
                    patch_name=row[reader.fieldnames[1]],
                    start_index=int(row[start_key]),
                    min_index=int(row[min_key]),
                    motion=tuple(value[0] for value in scores),
                    blur=tuple(value[1] for value in scores),
                )
            )
    return records


class ZStackDataset(Dataset):
    """Consume the final ``blur_motion_data6.csv`` annotation directly."""

    def __init__(
        self,
        data_root: str | Path,
        annotation: str | Path,
        *,
        split_file: str | Path | None = None,
        split: str | None = "train",
        sample_frames: int = 11,
        image_size: int = 256,
        crop_size: int = 253,
        blur_threshold: float = 0.2,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if sample_frames != 11:
            raise ValueError("The final PathoS2VD method uses exactly 11 source planes")
        self.data_root = Path(data_root)
        self.sample_frames = sample_frames
        self.image_size = image_size
        self.blur_threshold = blur_threshold
        self.transform = transform or (lambda image: pathology_transform(image, crop_size=crop_size, image_size=image_size))
        assignments = _read_split_file(split_file)
        self.records = [
            record
            for record in read_annotation(annotation)
            if split is None or not assignments or assignments.get(record.slide_name) == split
        ]
        if not self.records:
            raise ValueError(f"No source z-stacks remain for split {split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def _plane(self, record: ZStackRecord, selected_index: int) -> tuple[Path, float, float]:
        physical_index = min(max(selected_index, 0), 18)
        path = self.data_root / record.slide_name / PLANE_NAMES[physical_index] / record.patch_name
        if selected_index != physical_index:
            return path, float("inf"), 0.0
        return path, record.blur[physical_index], record.motion[physical_index]

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        frames: list[torch.Tensor] = []
        blur: list[float] = []
        motion: list[float] = []
        for offset in range(self.sample_frames):
            path, plane_blur, plane_motion = self._plane(record, record.start_index + offset)
            with Image.open(path) as image:
                frames.append(self.transform(image))
            blur.append(plane_blur)
            motion.append(plane_motion)
        pixel_values = torch.stack(frames)
        mask = torch.tensor([value <= self.blur_threshold for value in blur], dtype=torch.bool)
        motion_value = torch.tensor(sum(value for value, valid in zip(motion, mask.tolist()) if valid), dtype=torch.float32)
        return {
            "pixel_values": pixel_values,
            "mid_frame": pixel_values[5],
            "mask": mask,
            "motion": motion_value,
            "slide_name": record.slide_name,
            "patch_name": record.patch_name,
        }


class SourceVaeDataset(Dataset):
    """Expose the full 11-plane item used by the source VAE trainer."""

    def __init__(self, zstack_dataset: ZStackDataset) -> None:
        self.zstack_dataset = zstack_dataset

    def __len__(self) -> int:
        return len(self.zstack_dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.zstack_dataset[index]
        return {
            "pixel_values": item["pixel_values"],
            "mask": item["mask"],
            "slide_name": item["slide_name"],
            "patch_name": item["patch_name"],
        }
