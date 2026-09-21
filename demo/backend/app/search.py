"""Online meme retrieval: CLIP query encode → ES kNN → sim fusion → text-side PRF → paginate."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .config import settings
from .encoder import get_encoder
from . import es as es_mod


def _parse_schedule(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


def _fuse_score(q: np.ndarray, vec_proj: np.ndarray, vec_orig: np.ndarray, beta: float) -> float:
    # Cosine since vectors are L2-normalized
    sim_proj = float(np.dot(q, vec_proj))
    sim_orig = float(np.dot(q, vec_orig))
    return (1.0 - beta) * sim_proj + beta * sim_orig


def _expand_query(
    q: np.ndarray,
    candidates: List[Dict[str, Any]],
    scores: np.ndarray,
    alpha: float,
    k: int,
) -> np.ndarray:
    order = np.argsort(-scores)[:k]
    weights = np.maximum(scores[order], 0.0)
    weights = weights / (weights.sum() + 1e-8)
    mats = np.stack([np.asarray(candidates[i]["vector_proj"], dtype=np.float32) for i in order], axis=0)
    weighted = (mats * weights[:, None]).sum(axis=0)
    expanded = (1.0 - alpha) * q + alpha * weighted
    return expanded / (np.linalg.norm(expanded) + 1e-8)


def retrieve(query: str, page: int = 1, page_size: int = 12) -> Dict[str, Any]:
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 50))

    encoder = get_encoder()
    q = encoder.encode_text(query)
    beta = float(settings.sim_fusion_beta)
    n_docs = es_mod.count_docs()
    pool = min(int(settings.candidate_pool), n_docs) if n_docs else int(settings.candidate_pool)

    # Stage-1: ES script_score with fused projected/original cosine (exact over gallery slice)
    hits = es_mod.fused_script_search(q.tolist(), beta=beta, size=pool)
    if not hits:
        return {"query": query, "total": 0, "page": page, "page_size": page_size, "total_pages": 0, "results": []}

    def rescore(qv: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        sc = np.array(
            [
                _fuse_score(
                    qv,
                    np.asarray(h["vector_proj"], dtype=np.float32),
                    np.asarray(h["vector_orig"], dtype=np.float32),
                    beta,
                )
                for h in hits
            ],
            dtype=np.float32,
        )
        return hits, sc

    hits, scores = rescore(q)

    # Text-side pseudo-relevance feedback (online-safe; no image-side expansion)
    schedule = _parse_schedule(settings.expansion_k_schedule)
    alpha = float(settings.expansion_alpha)
    for it in range(max(1, int(settings.expansion_iters))):
        k_exp = schedule[min(it, len(schedule) - 1)] if schedule else 2
        k_exp = min(k_exp, len(hits))
        q = _expand_query(q, hits, scores, alpha, k_exp)
        hits, scores = rescore(q)

    order = np.argsort(-scores)
    ranked = []
    for rank, idx in enumerate(order, start=1):
        h = hits[idx]
        ranked.append(
            {
                "rank": rank,
                "meme_id": h["meme_id"],
                "gt": h.get("gt", ""),
                "caption": h.get("caption", ""),
                "score": float(scores[idx]),
                "image_url": f"/api/images/{h['image']}",
            }
        )

    total = len(ranked)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "query": query,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if page_size else 0,
        "results": ranked[start:end],
    }
