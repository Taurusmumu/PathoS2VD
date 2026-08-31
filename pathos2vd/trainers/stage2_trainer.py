from __future__ import annotations

import torch

from pathos2vd.datasets import PriorPreservingMixedDataset, TargetImageDataset, ZStackDataset
from pathos2vd.models import load_image_encoder, load_unet, load_vae

from .edm import reshape_target_group
from .stage1_trainer import Stage1Trainer


class Stage2Trainer(Stage1Trainer):
    def __init__(self, config: dict, **overrides):
        data, model, checkpoints = config["dataset"], config["model"], config["checkpoint"]
        source = overrides.pop("source_dataset", None) or ZStackDataset(
            data["source_data_root"], data["source_annotation"], split_file=data.get("split_file"),
            split="train", sample_frames=data.get("num_frames", 11), image_size=data.get("image_size", 256),
            crop_size=data.get("source_crop_size", 253), blur_threshold=data.get("blur_threshold", 0.2),
        )
        target = overrides.pop("target_dataset", None) or TargetImageDataset(
            data_root=data["target_data_root"], patterns=data.get("target_patterns", ["**/*.jpg"]),
            manifest=data.get("target_manifest"),
            image_size=data.get("image_size", 256), crop_size=data.get("target_crop_size", 256),
        )
        mixed = overrides.pop("dataset", None) or PriorPreservingMixedDataset(
            source, target, sample_frames=data.get("num_frames", 11),
            target_motion=data.get("target_motion", 1.7),
            reverse_source_probability=data.get("reverse_source_probability", 0.5),
            seed=config["training"].get("seed", 42),
        )
        if config["training"]["max_train_steps"] is None:
            raise ValueError("Stage-II max_train_steps must be set by the target configuration")
        if config["training"]["train_batch_size"] != 1:
            raise ValueError("Final alternating Stage-II implementation requires train_batch_size=1")
        vae = overrides.pop("vae", None) or load_vae(
            model["pretrained_model"], checkpoint=checkpoints["target_vae_checkpoint"], trainable=False
        )
        processor, encoder = (overrides.pop("image_processor", None), overrides.pop("image_encoder", None))
        if processor is None or encoder is None:
            processor, encoder = load_image_encoder(model["pretrained_model"], trainable=False)
        unet = overrides.pop("unet", None) or load_unet(checkpoints["stage1_checkpoint"], trainable=True)
        super().__init__(
            config, dataset=mixed, vae=vae, image_processor=processor, image_encoder=encoder, unet=unet
        )

    def _standardize_batch(self, batch: dict, dtype: torch.dtype) -> tuple[dict, bool]:
        label = int(batch["label"].item())
        pixel_values = batch["pixel_values"].to(self.accelerator.device, dtype=dtype)
        mask = batch["mask"].to(self.accelerator.device)
        motion = batch["motion"].to(self.accelerator.device)
        is_video = label == 1
        if not is_video:
            pixel_values, mask, motion = reshape_target_group(pixel_values, mask, motion)
        condition = pixel_values[:, (pixel_values.shape[1] - 1) // 2]
        return {"pixel_values": pixel_values, "condition": condition, "mask": mask, "motion": motion}, is_video
