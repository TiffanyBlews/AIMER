from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths inside container / local defaults
    clip_checkpoint: str = "/models/clip_douban.pt"
    image_projection_path: str = "/models/image_proj_ridge_l1.pt"
    feature_cache_path: str = "/models/test_cache.pt"
    metadata_path: str = "/data/metadata.json"
    image_dir: str = "/data/images"

    # Elasticsearch
    es_url: str = "http://elasticsearch:9200"
    es_index: str = "memes"
    es_dims: int = 512

    # Retrieval (matches douban best config, text-side PRF only for online serving)
    projection_mix: float = 0.9
    sim_fusion_beta: float = 0.5
    expansion_alpha: float = 0.03
    expansion_k_schedule: str = "3,2,2"
    expansion_iters: int = 3
    candidate_pool: int = 1000

    # Device
    device: str = "cpu"

    class Config:
        env_prefix = "CARE_"


settings = Settings()

# Resolve relative paths when running outside Docker
_ROOT = Path(__file__).resolve().parents[2]
for attr, default_rel in [
    ("clip_checkpoint", "models/clip_douban.pt"),
    ("image_projection_path", "models/image_proj_ridge_l1.pt"),
    ("feature_cache_path", "models/test_cache.pt"),
    ("metadata_path", "data/metadata.json"),
    ("image_dir", "data/images"),
]:
    val = getattr(settings, attr)
    if not Path(val).exists():
        alt = _ROOT / default_rel
        if alt.exists():
            setattr(settings, attr, str(alt))
