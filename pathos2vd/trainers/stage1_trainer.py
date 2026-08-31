from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pathos2vd.datasets import ZStackDataset
from pathos2vd.models import load_image_encoder, load_unet, load_vae
from pathos2vd.pipelines.pathos2vd_pipeline import _resize_with_antialiasing
from pathos2vd.utils.compute_cost import TrainingComputeProfiler
from pathos2vd.utils.distributed import make_accelerator, seed_everything

from .edm import denoised_from_model, paper_masked_edm_loss, prepare_edm_batch
from .loops import infinite_batches


LOGGER = logging.getLogger(__name__)


class Stage1Trainer:
    def __init__(self, config: dict, *, dataset=None, vae=None, image_processor=None, image_encoder=None, unet=None):
        self.config = config
        self.accelerator = make_accelerator(config)
        seed_everything(int(config["training"].get("seed", 42)))
        model = config["model"]
        data = config["dataset"]
        self.dataset = dataset or ZStackDataset(
            data["source_data_root"], data["source_annotation"], split_file=data.get("split_file"),
            split=data.get("split", "train"), sample_frames=data.get("num_frames", 11),
            image_size=data.get("image_size", 256), crop_size=data.get("crop_size", 253),
            blur_threshold=data.get("blur_threshold", 0.2),
        )
        self.vae = vae or load_vae(model["pretrained_model"], checkpoint=model["source_vae_checkpoint"], trainable=False)
        if image_processor is None or image_encoder is None:
            image_processor, image_encoder = load_image_encoder(model["pretrained_model"], trainable=False)
        self.image_processor, self.image_encoder = image_processor, image_encoder
        # Final paper method: initialize from SVD-XT and DO NOT call weights_init.
        self.unet = unet or load_unet(model["pretrained_model"], trainable=True)

    def _encode_image(self, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        images = _resize_with_antialiasing(images.float(), (224, 224))
        images = (images + 1.0) / 2.0
        processed = self.image_processor(
            images=images, do_normalize=True, do_center_crop=False, do_resize=False,
            do_rescale=False, return_tensors="pt",
        ).pixel_values.to(self.accelerator.device, dtype=dtype)
        return self.image_encoder(processed).image_embeds

    def _standardize_batch(self, batch: dict, dtype: torch.dtype) -> tuple[dict, bool]:
        return {
            "pixel_values": batch["pixel_values"].to(self.accelerator.device, dtype=dtype),
            "condition": batch["mid_frame"].to(self.accelerator.device, dtype=dtype),
            "mask": batch["mask"].to(self.accelerator.device),
            "motion": batch["motion"].to(self.accelerator.device),
        }, True

    def fit(self) -> None:
        training = self.config["training"]
        edm = self.config.get("edm", {})
        checkpoint = self.config["checkpoint"]
        loader = DataLoader(
            self.dataset, batch_size=training["train_batch_size"], shuffle=True,
            num_workers=training.get("num_workers", 0), pin_memory=True,
        )
        if training.get("gradient_checkpointing", False):
            self.unet.enable_gradient_checkpointing()
        if training.get("use_ema", False):
            raise NotImplementedError("EMA was disabled in the final paper runs and is not silently emulated")
        optimizer = torch.optim.AdamW(
            self.unet.parameters(), lr=training["learning_rate"],
            betas=(training.get("adam_beta1", 0.9), training.get("adam_beta2", 0.999)),
            weight_decay=training.get("weight_decay", 0.01), eps=training.get("adam_epsilon", 1e-8),
        )
        try:
            from diffusers.optimization import get_scheduler
        except ImportError as error:
            raise RuntimeError("Stage-I training requires diffusers") from error
        scheduler = get_scheduler(
            training.get("lr_scheduler", "constant_with_warmup"), optimizer=optimizer,
            num_warmup_steps=training.get("warmup_steps", 1500),
            num_training_steps=training["max_train_steps"],
        )
        self.unet, optimizer, loader, scheduler = self.accelerator.prepare(self.unet, optimizer, loader, scheduler)
        self.vae.to(self.accelerator.device)
        self.image_encoder.to(self.accelerator.device)
        self.vae.eval()
        self.image_encoder.eval()
        self.unet.train()
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.accelerator.mixed_precision, torch.float32)
        generator = torch.Generator(device=self.accelerator.device).manual_seed(training.get("seed", 42))
        output_dir = Path(checkpoint["output_dir"])
        profiler = TrainingComputeProfiler(self.accelerator, training, self.config.get("logging"))
        profiler.start()
        for global_step, raw_batch in enumerate(infinite_batches(loader), start=1):
            with self.accelerator.accumulate(self.unet):
                batch, is_video = self._standardize_batch(raw_batch, dtype)
                with torch.no_grad():
                    embeddings = self._encode_image(batch["condition"], dtype)
                    edm_batch = prepare_edm_batch(
                        pixel_values=batch["pixel_values"], condition_pixel_values=batch["condition"],
                        mask=batch["mask"], motion=batch["motion"], embeddings=embeddings,
                        vae=self.vae, is_video=is_video, fps=edm.get("fps", 7),
                        sigma_loc=edm.get("sigma_loc", 0.7), sigma_scale=edm.get("sigma_scale", 1.6),
                        condition_sigma_loc=edm.get("condition_sigma_loc", -3.0),
                        condition_sigma_scale=edm.get("condition_sigma_scale", 0.5),
                        conditioning_dropout_probability=training.get("conditioning_dropout_probability", 0.1),
                        generator=generator,
                    )
                prediction = self.unet(
                    edm_batch.unet_input, edm_batch.timesteps, edm_batch.embeddings,
                    added_time_ids=edm_batch.added_time_ids, is_video=edm_batch.is_video,
                ).sample
                denoised = denoised_from_model(prediction, edm_batch)
                loss = paper_masked_edm_loss(denoised, edm_batch.clean_latents, edm_batch.sigmas, edm_batch.mask)
                self.accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            profiler.record_optimization_step(completed=self.accelerator.sync_gradients)
            if self.accelerator.is_main_process and global_step % 100 == 0:
                LOGGER.info("step=%d loss=%.6f", global_step, loss.detach().float().item())
            if global_step % checkpoint.get("save_every", 2500) == 0:
                path = output_dir / f"checkpoint-{global_step}"
                profiler.time_auxiliary(lambda: self._save(path))
            if global_step >= training["max_train_steps"]:
                break
        if training["max_train_steps"] % checkpoint.get("save_every", 2500) != 0:
            path = output_dir / f"checkpoint-{training['max_train_steps']}"
            profiler.time_auxiliary(lambda: self._save(path))
        profiler.finish()

    def _save(self, path: Path) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            path.mkdir(parents=True, exist_ok=True)
            self.accelerator.unwrap_model(self.unet).save_pretrained(path / "unet")
        self.accelerator.save_state(str(path / "accelerate_state"))
