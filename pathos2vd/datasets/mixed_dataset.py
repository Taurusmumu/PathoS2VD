from __future__ import annotations

import random
from typing import Sequence

import torch
from torch.utils.data import Dataset


class PriorPreservingMixedDataset(Dataset):
    """Faithful alternating concatenation used for Stage-II adaptation.

    The target list is repeated ``floor(S/I) * F`` times and grouped into F
    independent images. This intentionally preserves the paper-run behavior;
    it is not replaced by a cleaner dual-stream sampler.
    """

    def __init__(
        self,
        source_dataset: Dataset,
        target_dataset: Dataset,
        *,
        sample_frames: int = 11,
        target_motion: float = 1.7,
        reverse_source_probability: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.sample_frames = sample_frames
        self.target_motion = target_motion
        self.reverse_source_probability = reverse_source_probability
        self.seed = seed
        source_count, target_count = len(source_dataset), len(target_dataset)
        if target_count == 0:
            raise ValueError("Target dataset is empty")
        repeated = list(range(target_count)) * (source_count // target_count * sample_frames)
        random.Random(seed).shuffle(repeated)
        self.target_groups = [
            repeated[index : index + sample_frames]
            for index in range(0, len(repeated), sample_frames)
            if len(repeated[index : index + sample_frames]) == sample_frames
        ]

    @property
    def target_probability(self) -> float:
        return len(self.target_groups) / len(self) if len(self) else 0.0

    def __len__(self) -> int:
        return len(self.target_groups) + len(self.source_dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < len(self.target_groups):
            frames = torch.stack(
                [self.target_dataset[target_index]["pixel_values"] for target_index in self.target_groups[index]]
            )
            return {
                "pixel_values": frames,
                "mid_frame": frames[5],
                "mask": torch.ones(self.sample_frames, dtype=torch.bool),
                "motion": torch.tensor(self.target_motion, dtype=torch.float32),
                "label": torch.tensor(0, dtype=torch.long),
                "is_video": False,
            }
        source_index = index - len(self.target_groups)
        item = dict(self.source_dataset[source_index])
        # The reference uses Python random at access time. Preserve that stochastic behavior.
        if random.random() < self.reverse_source_probability:
            item["pixel_values"] = torch.flip(item["pixel_values"], dims=(0,))
            item["mask"] = torch.flip(item["mask"], dims=(0,))
            item["mid_frame"] = item["pixel_values"][5]
        item["label"] = torch.tensor(1, dtype=torch.long)
        item["is_video"] = True
        return item
