from __future__ import annotations

from dataclasses import dataclass

import torch

from pathos2vd.models.vae import encode_video_latents


def rand_log_normal(shape, *, loc: float, scale: float, device, dtype) -> torch.Tensor:
    return torch.exp(torch.randn(shape, device=device, dtype=dtype) * scale + loc)


def reshape_target_group(pixel_values: torch.Tensor, mask: torch.Tensor, motion: torch.Tensor):
    """Convert ``[1,F,C,H,W]`` into F independent ``T=1`` examples."""
    if pixel_values.shape[0] != 1:
        raise ValueError("Paper-faithful Stage II requires train_batch_size=1 for alternating item types")
    frames = pixel_values.shape[1]
    reshaped = pixel_values.reshape(frames, 1, *pixel_values.shape[2:])
    return reshaped, mask.reshape(frames, 1), motion.reshape(1).repeat(frames)


def make_added_time_ids(
    fps: float,
    motion: torch.Tensor,
    noise_aug_strength: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    fps_column = torch.full_like(motion, float(fps), dtype=dtype)
    return torch.stack((fps_column, motion.to(dtype), noise_aug_strength.to(dtype)), dim=1)


def apply_conditioning_dropout(
    embeddings: torch.Tensor,
    condition_latents: torch.Tensor,
    probability: float | None,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(1)
    if probability is None:
        return embeddings, condition_latents
    batch = embeddings.shape[0]
    random_p = torch.rand(batch, device=embeddings.device, generator=generator)
    prompt_mask = (random_p < 2 * probability).reshape(batch, 1, 1)
    embeddings = torch.where(prompt_mask, torch.zeros_like(embeddings), embeddings)
    image_mask = 1 - (
        (random_p >= probability).to(condition_latents.dtype)
        * (random_p < 3 * probability).to(condition_latents.dtype)
    )
    condition_latents = condition_latents * image_mask.reshape(batch, 1, 1, 1)
    return embeddings, condition_latents


def expand_loss_mask(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    if mask.shape != latent.shape[:2]:
        raise ValueError(f"Mask {tuple(mask.shape)} does not match latent prefix {tuple(latent.shape[:2])}")
    return mask.to(torch.bool)[:, :, None, None, None].expand_as(latent)


def paper_masked_edm_loss(
    denoised_latents: torch.Tensor,
    target: torch.Tensor,
    sigmas: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Paper-run masked loss, including its full-mask denominator quirk.

    Invalid elements are removed from the numerator, but the denominator is the
    number of *all* expanded elements, not the number of valid elements.
    """
    weighing = (1 + sigmas**2) * sigmas.pow(-2.0)
    expanded = expand_loss_mask(mask, denoised_latents)
    errors = weighing.float() * (denoised_latents.float() - target.float()).pow(2)
    return errors[expanded].sum() / torch.ones_like(expanded).sum()


@dataclass
class EDMBatch:
    unet_input: torch.Tensor
    timesteps: torch.Tensor
    embeddings: torch.Tensor
    added_time_ids: torch.Tensor
    noisy_latents: torch.Tensor
    clean_latents: torch.Tensor
    sigmas: torch.Tensor
    mask: torch.Tensor
    is_video: bool


def prepare_edm_batch(
    *,
    pixel_values: torch.Tensor,
    condition_pixel_values: torch.Tensor,
    mask: torch.Tensor,
    motion: torch.Tensor,
    embeddings: torch.Tensor,
    vae,
    is_video: bool,
    fps: float = 7.0,
    sigma_loc: float = 0.7,
    sigma_scale: float = 1.6,
    condition_sigma_loc: float = -3.0,
    condition_sigma_scale: float = 0.5,
    conditioning_dropout_probability: float | None = 0.1,
    generator: torch.Generator | None = None,
) -> EDMBatch:
    latents = encode_video_latents(pixel_values, vae, scaled=True)
    batch = latents.shape[0]
    noise = torch.randn_like(latents)
    condition_sigmas = rand_log_normal(
        (batch,), loc=condition_sigma_loc, scale=condition_sigma_scale,
        device=latents.device, dtype=latents.dtype,
    )
    augmented = condition_pixel_values[:, None] + torch.randn_like(condition_pixel_values[:, None]) * condition_sigmas[:, None, None, None, None]
    condition_latents = encode_video_latents(augmented, vae, scaled=True)[:, 0]
    # Reference divides the scaled condition latent back to the raw VAE latent.
    condition_latents = condition_latents / vae.config.scaling_factor
    embeddings, condition_latents = apply_conditioning_dropout(
        embeddings, condition_latents, conditioning_dropout_probability, generator=generator
    )
    sigmas = rand_log_normal(
        (batch,), loc=sigma_loc, scale=sigma_scale, device=latents.device, dtype=latents.dtype
    )[:, None, None, None, None]
    noisy_latents = latents + noise * sigmas
    scaled_noisy = noisy_latents / torch.sqrt(sigmas**2 + 1)
    repeated_condition = condition_latents[:, None].repeat(1, latents.shape[1], 1, 1, 1)
    unet_input = torch.cat((scaled_noisy, repeated_condition), dim=2)
    timesteps = 0.25 * torch.log(sigmas[:, 0, 0, 0, 0])
    added_time_ids = make_added_time_ids(
        fps, motion.reshape(batch), condition_sigmas, dtype=embeddings.dtype
    ).to(latents.device)
    return EDMBatch(
        unet_input=unet_input,
        timesteps=timesteps,
        embeddings=embeddings,
        added_time_ids=added_time_ids,
        noisy_latents=noisy_latents,
        clean_latents=latents,
        sigmas=sigmas,
        mask=mask.to(torch.bool),
        is_video=is_video,
    )


def denoised_from_model(model_prediction: torch.Tensor, batch: EDMBatch) -> torch.Tensor:
    c_out = -batch.sigmas / torch.sqrt(batch.sigmas**2 + 1)
    c_skip = 1 / (batch.sigmas**2 + 1)
    return model_prediction * c_out + c_skip * batch.noisy_latents
