"""CLIP text encoder + ridge image projection (douban best-config subset)."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import clip
import numpy as np
import torch

from .config import settings

logger = logging.getLogger(__name__)


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norms + 1e-8)


def apply_image_projection(
    image_matrix: np.ndarray,
    W: np.ndarray,
    projection_mix: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (projected_mixed, original) both L2-normalized."""
    original = _l2_normalize(image_matrix.copy())
    projected = image_matrix @ W.T
    mix = float(projection_mix)
    if mix <= 0.0:
        mixed = original
    elif mix >= 1.0:
        mixed = _l2_normalize(projected)
    else:
        mixed = _l2_normalize((1.0 - mix) * original + mix * projected)
    return mixed, original


class ClipEncoder:
    def __init__(self) -> None:
        self.device = torch.device(settings.device if torch.cuda.is_available() or settings.device == "cpu" else "cpu")
        if settings.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            self.device = torch.device("cpu")

        logger.info("Loading CLIP ViT-B/32 on %s ...", self.device)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        state = torch.load(settings.clip_checkpoint, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "clip_state_dict" in state:
            state = state["clip_state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected[:5])
        self.model.eval()

        proj = torch.load(settings.image_projection_path, map_location="cpu", weights_only=False)
        W = proj["W"] if isinstance(proj, dict) and "W" in proj else proj
        if isinstance(W, torch.Tensor):
            W = W.cpu().numpy()
        self.W = np.asarray(W, dtype=np.float32)
        self.projection_mix = settings.projection_mix
        logger.info("CLIP ready; projection W shape=%s mix=%s", self.W.shape, self.projection_mix)

    @torch.no_grad()
    def encode_text(self, query: str) -> np.ndarray:
        tokens = clip.tokenize([query], truncate=True).to(self.device)
        feat = self.model.encode_text(tokens).float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feat.cpu().numpy().astype(np.float32)[0]

    @torch.no_grad()
    def encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device)
        feat = self.model.encode_image(images).float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feat.cpu().numpy().astype(np.float32)

    def project_images(self, image_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return apply_image_projection(image_matrix, self.W, self.projection_mix)


_encoder: Optional[ClipEncoder] = None


def get_encoder() -> ClipEncoder:
    global _encoder
    if _encoder is None:
        _encoder = ClipEncoder()
    return _encoder
