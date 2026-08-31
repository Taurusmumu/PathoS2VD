from __future__ import annotations

from pathlib import Path


def load_unet(pretrained_model_or_checkpoint: str | Path, *, subfolder: str = "unet", trainable: bool = True):
    """Load paper U-Net weights without post-paper ``weights_init`` reinitialization."""
    from .reference.unet_spatiotemporal_impl import UNetSpatioTemporalConditionModel

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        str(pretrained_model_or_checkpoint),
        subfolder=subfolder,
        low_cpu_mem_usage=True,
    )
    unet.requires_grad_(trainable)
    return unet


def load_image_encoder(pretrained_model: str, *, trainable: bool = False):
    try:
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    except ImportError as error:
        raise RuntimeError("Loading the image condition encoder requires transformers") from error
    processor = CLIPImageProcessor.from_pretrained(pretrained_model, subfolder="feature_extractor")
    encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model, subfolder="image_encoder")
    encoder.requires_grad_(trainable)
    return processor, encoder
