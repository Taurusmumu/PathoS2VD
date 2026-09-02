from __future__ import annotations

from pathlib import Path
import math

import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pytorch_fid.inception import InceptionV3


class _Images(Dataset):
    def __init__(self, paths: list[str]): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, index: int):
        array = np.asarray(Image.open(self.paths[index]).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def _frechet(first: np.ndarray, second: np.ndarray) -> float:
    mean1, mean2 = first.mean(0), second.mean(0)
    if len(first) == 1 or len(second) == 1:
        return float(np.square(mean1 - mean2).sum())
    covariance1, covariance2 = np.cov(first, rowvar=False), np.cov(second, rowvar=False)
    product, _ = sqrtm(covariance1 @ covariance2, disp=False)
    if not np.isfinite(product).all():
        offset = np.eye(covariance1.shape[0]) * 1e-6
        product = sqrtm((covariance1 + offset) @ (covariance2 + offset))
    return float(np.real(np.square(mean1 - mean2).sum() + np.trace(covariance1 + covariance2 - 2 * product)))


def fid(real_images: list[str], generated_images: list[str], *, device: str | torch.device = "cuda", batch_size: int = 1, num_workers: int = 0) -> float:
    """The old project's InceptionV3 pool-3 FID implementation."""
    if not real_images or not generated_images:
        raise ValueError("FID requires at least one real and one generated image")
    device = torch.device(device)
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    def activations(paths: list[str]) -> np.ndarray:
        values = []
        for images in DataLoader(_Images(sorted(paths)), batch_size=batch_size, num_workers=num_workers):
            with torch.no_grad():
                features = model(images.to(device))[0]
            if features.shape[-2:] != (1, 1): features = adaptive_avg_pool2d(features, (1, 1))
            values.append(features.squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(values)
    return _frechet(activations(real_images), activations(generated_images))


I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"


def _i3d(weights: str | Path | None, device: torch.device) -> torch.jit.ScriptModule:
    path = Path(weights) if weights else Path.home() / ".cache" / "pathos2vd" / "i3d_torchscript.pt"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(I3D_URL, str(path), progress=True)
    return torch.jit.load(str(path), map_location=device).eval()


def _i3d_input(video: torch.Tensor) -> torch.Tensor:
    # Old StyleGAN-V implementation: C,T,H,W [0,1] -> center-cropped 224 and [-1,1].
    _, _, height, width = video.shape
    scale = 224 / min(height, width)
    size = (224, math.ceil(width * scale)) if height < width else (math.ceil(height * scale), 224)
    video = F.interpolate(video, size=size, mode="bilinear", align_corners=False)
    _, _, height, width = video.shape
    return ((video[:, :, (height-224)//2:(height-224)//2+224, (width-224)//2:(width-224)//2+224] - .5) * 2).contiguous()


def fvd(real_videos: torch.Tensor, generated_videos: torch.Tensor, *, device: str | torch.device = "cuda", i3d_weights: str | Path | None = None, batch_size: int = 10) -> float:
    """Old StyleGAN-V FVD: videos are ``[B,T,C,H,W]`` floats in ``[0,1]``."""
    if real_videos.ndim != 5 or generated_videos.ndim != 5 or real_videos.shape[1] < 10 or generated_videos.shape[1] < 10:
        raise ValueError("FVD expects [B,T,C,H,W] videos with at least 10 frames")
    device = torch.device(device)
    detector = _i3d(i3d_weights, device)
    def features(videos: torch.Tensor) -> np.ndarray:
        videos = videos if videos.shape[2] == 3 else videos.repeat(1, 1, 3, 1, 1)
        values = []
        with torch.no_grad():
            for start in range(0, len(videos), batch_size):
                batch = torch.stack([_i3d_input(video.permute(1, 0, 2, 3)) for video in videos[start:start+batch_size]]).to(device)
                values.append(detector(x=batch, rescale=False, resize=False, return_features=True).cpu().numpy())
        return np.concatenate(values)
    return _frechet(features(real_videos), features(generated_videos))
