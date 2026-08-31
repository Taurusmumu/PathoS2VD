from __future__ import annotations

from pathlib import Path

import torch

from pathos2vd.utils.checkpoint import load_state_dict


def load_vae(
    pretrained_model: str,
    *,
    checkpoint: str | Path | None = None,
    trainable: bool = False,
):
    """Load the SVD-XT temporal decoder and optionally replace its state dict."""
    try:
        from diffusers import AutoencoderKLTemporalDecoder
    except ImportError as error:
        raise RuntimeError("Loading the PathoS2VD VAE requires diffusers") from error
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model, subfolder="vae")
    if checkpoint is not None:
        load_state_dict(vae, checkpoint, strict=True)
    vae.requires_grad_(trainable)
    return vae


def encode_video_latents(pixel_values: torch.Tensor, vae, *, scaled: bool = True) -> torch.Tensor:
    """Encode ``[B,F,C,H,W]`` into ``[B,F,4,h,w]`` by sampling the posterior."""
    batch, frames = pixel_values.shape[:2]
    flat = pixel_values.reshape(batch * frames, *pixel_values.shape[2:])
    latents = vae.encode(flat).latent_dist.sample()
    latents = latents.reshape(batch, frames, *latents.shape[1:])
    if scaled:
        latents = latents * vae.config.scaling_factor
    return latents


def decode_video_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    batch, frames = latents.shape[:2]
    flat = (latents / vae.config.scaling_factor).reshape(batch * frames, *latents.shape[2:])
    indicator = torch.zeros(batch, frames, device=flat.device, dtype=flat.dtype)
    decoded = vae.decode(flat, num_frames=frames, image_only_indicator=indicator).sample
    return decoded.reshape(batch, frames, *decoded.shape[1:])
