from __future__ import annotations

from typing import Sequence, TypeVar


T = TypeVar("T")


def central_latent_shape(
    batch_size: int = 1,
    num_frames: int = 11,
    latent_channels: int = 4,
    image_size: int = 256,
    vae_scale_factor: int = 8,
) -> tuple[int, int, int, int, int]:
    return (
        batch_size,
        num_frames,
        latent_channels,
        image_size // vae_scale_factor,
        image_size // vae_scale_factor,
    )


def compose_three_blocks(upper: Sequence[T], central: Sequence[T], lower: Sequence[T]) -> list[T]:
    if not (len(upper) == len(central) == len(lower) == 11):
        raise ValueError("PathoS2VD final inference requires three 11-plane blocks")
    output = list(upper) + list(central) + list(lower)
    assert len(output) == 33
    return output
