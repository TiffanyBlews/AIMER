"""Index meme image vectors into Elasticsearch.

Prefer precomputed CLIP features from test_cache.pt (fast); otherwise encode images.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Allow `python -m scripts.index_memes` from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.encoder import apply_image_projection, get_encoder  # noqa: E402
from app import es as es_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("index_memes")


def load_metadata() -> list:
    with open(settings.metadata_path, encoding="utf-8") as f:
        return json.load(f)


def load_features_from_cache() -> tuple | None:
    cache_path = Path(settings.feature_cache_path)
    if not cache_path.is_file():
        return None
    logger.info("Loading feature cache: %s", cache_path)
    obj = torch.load(cache_path, map_location="cpu", weights_only=False)
    images = obj["image"].cpu().numpy().astype(np.float32)
    vids = obj.get("video_ids")
    if vids is None:
        return None
    # L2-normalize originals (cache should already be normalized)
    norms = np.linalg.norm(images, axis=1, keepdims=True)
    images = images / (norms + 1e-8)
    return list(vids), images


def encode_all_images(meta: list, batch_size: int = 64) -> tuple:
    from PIL import Image

    encoder = get_encoder()
    image_dir = Path(settings.image_dir)
    feats = []
    vids = []
    batch_imgs = []
    batch_ids = []

    def flush():
        nonlocal batch_imgs, batch_ids
        if not batch_imgs:
            return
        import torch as th

        tensor = th.stack(batch_imgs, dim=0)
        arr = encoder.encode_image_batch(tensor)
        feats.append(arr)
        vids.extend(batch_ids)
        batch_imgs, batch_ids = [], []

    for item in tqdm(meta, desc="encode"):
        path = image_dir / item["image"]
        if not path.is_file():
            logger.warning("Missing image: %s", path)
            continue
        img = Image.open(path).convert("RGB")
        batch_imgs.append(encoder.preprocess(img))
        batch_ids.append(item["meme_id"])
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return vids, np.concatenate(feats, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate ES index")
    parser.add_argument("--from-images", action="store_true", help="Force encode from images")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    meta = load_metadata()
    meta_by_id = {m["meme_id"]: m for m in meta}
    logger.info("Metadata entries: %d", len(meta))

    es = es_mod.wait_for_es()
    es_mod.ensure_index(es, recreate=args.recreate)

    cached = None if args.from_images else load_features_from_cache()
    if cached is not None:
        vids, image_matrix = cached
        logger.info("Cache features: %d x %d", *image_matrix.shape)
        # Index-from-cache only needs projection matrix W (skip loading CLIP)
        proj = torch.load(settings.image_projection_path, map_location="cpu", weights_only=False)
        W = proj["W"] if isinstance(proj, dict) and "W" in proj else proj
        if isinstance(W, torch.Tensor):
            W = W.cpu().numpy()
        vec_proj, vec_orig = apply_image_projection(
            image_matrix, np.asarray(W, dtype=np.float32), settings.projection_mix
        )
    else:
        logger.info("Encoding images from disk ...")
        encoder = get_encoder()
        vids, image_matrix = encode_all_images(meta, batch_size=args.batch_size)
        vec_proj, vec_orig = encoder.project_images(image_matrix)

    docs = []
    skipped = 0
    for i, vid in enumerate(vids):
        m = meta_by_id.get(vid)
        if m is None:
            skipped += 1
            continue
        docs.append(
            {
                "meme_id": vid,
                "gt": m.get("gt", ""),
                "caption": m.get("caption", ""),
                "image": m.get("image", f"{vid}.jpg"),
                "vector_proj": vec_proj[i].astype(np.float32).tolist(),
                "vector_orig": vec_orig[i].astype(np.float32).tolist(),
            }
        )

    logger.info("Indexing %d docs (skipped %d without metadata) ...", len(docs), skipped)
    n = es_mod.bulk_index(docs, es=es)
    logger.info("Done. bulk success≈%d, index count=%d", n, es_mod.count_docs(es))


if __name__ == "__main__":
    main()
