from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pathos2vd.datasets import PriorPreservingMixedDataset, SourceVaeDataset, TargetImageDataset, ZStackDataset
from pathos2vd.models import load_vae
from pathos2vd.utils.compute_cost import TrainingComputeProfiler
from pathos2vd.utils.distributed import make_accelerator, seed_everything

from .loops import infinite_batches


LOGGER = logging.getLogger(__name__)


class VAETrainer:
    def __init__(self, config: dict, *, target_adaptation: bool = False, dataset=None, vae=None):
        self.config = config
        self.target_adaptation = target_adaptation
        section = config.get("target_vae_training", config["training"])
        accelerator_config = dict(config)
        accelerator_config["training"] = section
        self.training = section
        if section.get("max_train_steps") is None:
            raise ValueError("VAE max_train_steps must be set by the selected configuration")
        self.accelerator = make_accelerator(accelerator_config)
        seed_everything(section.get("seed", 42))
        model, data = config["model"], config["dataset"]
        if dataset is None:
            source_stack = ZStackDataset(
                data["source_data_root"], data["source_annotation"], split_file=data.get("split_file"),
                split=data.get("split", "train"), image_size=data.get("image_size", 256),
                crop_size=data.get("source_crop_size", data.get("crop_size", 253)),
            )
            if target_adaptation:
                target = TargetImageDataset(
                    data_root=data["target_data_root"], patterns=data.get("target_patterns", ["**/*.jpg"]),
                    manifest=data.get("target_manifest"),
                    image_size=data.get("image_size", 256), crop_size=data.get("target_crop_size", 256),
                )
                dataset = PriorPreservingMixedDataset(
                    source_stack, target, sample_frames=data.get("num_frames", 11),
                    target_motion=data.get("target_motion", 1.7),
                    reverse_source_probability=0.0, seed=section.get("seed", 42),
                )
                if section["train_batch_size"] != 1:
                    raise ValueError("Final mixed VAE adaptation requires train_batch_size=1")
            else:
                dataset = SourceVaeDataset(source_stack)
        self.dataset = dataset
        initialization = None
        if target_adaptation:
            initialization = config["checkpoint"].get("source_vae_checkpoint")
        self.vae = vae or load_vae(model["pretrained_model"], checkpoint=initialization, trainable=True)

    def fit(self) -> None:
        training = self.training
        loader = DataLoader(
            self.dataset, batch_size=training["train_batch_size"], shuffle=True,
            num_workers=training.get("num_workers", 0), pin_memory=True,
        )
        optimizer = torch.optim.AdamW(
            self.vae.parameters(), lr=training["learning_rate"],
            betas=(training.get("adam_beta1", 0.9), training.get("adam_beta2", 0.999)),
            weight_decay=training.get("weight_decay", 0.005),
        )
        try:
            import lpips
            from diffusers.optimization import get_scheduler
        except ImportError as error:
            raise RuntimeError("VAE training requires lpips and diffusers") from error
        perceptual = lpips.LPIPS(net="vgg").eval()
        perceptual.requires_grad_(False)
        scheduler = get_scheduler(
            training.get("lr_scheduler", "constant_with_warmup"), optimizer=optimizer,
            num_warmup_steps=training.get("warmup_steps", 1500),
            num_training_steps=training["max_train_steps"],
        )
        self.vae, optimizer, loader, scheduler, perceptual = self.accelerator.prepare(
            self.vae, optimizer, loader, scheduler, perceptual
        )
        output_key = "target_vae_output_dir" if self.target_adaptation else "output_dir"
        output = Path(self.config["checkpoint"][output_key])
        loss_config = self.config.get("loss", {"lpips_weight": 1.0, "kl_weight": 1e-6})
        self.vae.train()
        profiler = TrainingComputeProfiler(self.accelerator, training, self.config.get("logging"))
        profiler.start()
        for step, batch in enumerate(infinite_batches(loader), start=1):
            with self.accelerator.accumulate(self.vae):
                pixels, frames, image_only = self._prepare_pixels(batch)
                model = self.vae.module if hasattr(self.vae, "module") else self.vae
                posterior = model.encode(pixels).latent_dist
                latents = posterior.sample()
                indicator = torch.full(
                    (pixels.shape[0] // frames, frames), float(image_only),
                    device=pixels.device, dtype=pixels.dtype,
                )
                if self.target_adaptation:
                    reconstruction = model.decoder(
                        latents, num_frames=frames, image_only_indicator=indicator
                    )
                else:
                    reconstruction = model.decode(latents, num_frames=frames).sample
                l1 = F.l1_loss(reconstruction.float(), pixels.float())
                lpips_loss = perceptual(reconstruction.float(), pixels.float()).mean()
                kl = posterior.kl().mean()
                loss = l1 + loss_config.get("lpips_weight", 1.0) * lpips_loss + loss_config.get("kl_weight", 1e-6) * kl
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.vae.parameters(), training.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            profiler.record_optimization_step(completed=self.accelerator.sync_gradients)
            if self.accelerator.is_main_process and step % 100 == 0:
                LOGGER.info("step=%d vae_loss=%.6f", step, loss.detach().float().item())
            save_every = training.get("save_every", self.config["checkpoint"].get("save_every", 1000))
            if step % save_every == 0:
                path = output / f"checkpoint_{step}"
                profiler.time_auxiliary(lambda: self._save(path))
            if step >= training["max_train_steps"]:
                break
        if training["max_train_steps"] % save_every != 0:
            path = output / f"checkpoint_{training['max_train_steps']}"
            profiler.time_auxiliary(lambda: self._save(path))
        profiler.finish()

    def _prepare_pixels(self, batch: dict) -> tuple[torch.Tensor, int, bool]:
        pixels = batch["pixel_values"].to(self.accelerator.device)
        if not self.target_adaptation:
            batch_size, frames = pixels.shape[:2]
            return pixels.reshape(batch_size * frames, *pixels.shape[2:]), frames, False
        if pixels.shape[0] != 1:
            raise ValueError("Paper-faithful mixed VAE adaptation requires train_batch_size=1")
        frames = pixels.shape[1]
        image_only = int(batch["label"].item()) == 0
        return pixels.reshape(frames, *pixels.shape[2:]), frames, image_only

    def _save(self, path: Path) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            path.mkdir(parents=True, exist_ok=True)
            state = self.accelerator.unwrap_model(self.vae).state_dict()
            torch.save(state, path / "pytorch_model.bin")
