"""Elasticsearch client helpers for meme dense-vector index."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch, helpers

from .config import settings

logger = logging.getLogger(__name__)


def get_es() -> Elasticsearch:
    return Elasticsearch(settings.es_url, request_timeout=60)


def wait_for_es(timeout: int = 120) -> Elasticsearch:
    es = get_es()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if es.ping():
                logger.info("Elasticsearch is ready at %s", settings.es_url)
                return es
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Elasticsearch not ready after {timeout}s: {settings.es_url}")


def ensure_index(es: Optional[Elasticsearch] = None, recreate: bool = False) -> None:
    es = es or get_es()
    if recreate and es.indices.exists(index=settings.es_index):
        es.indices.delete(index=settings.es_index)
        logger.info("Deleted existing index %s", settings.es_index)

    if es.indices.exists(index=settings.es_index):
        return

    body = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "meme_id": {"type": "keyword"},
                "gt": {"type": "text"},
                "caption": {"type": "text"},
                "image": {"type": "keyword"},
                "vector_proj": {
                    "type": "dense_vector",
                    "dims": settings.es_dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "vector_orig": {
                    "type": "dense_vector",
                    "dims": settings.es_dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }
    es.indices.create(index=settings.es_index, body=body)
    logger.info("Created index %s", settings.es_index)


def bulk_index(docs: List[Dict[str, Any]], es: Optional[Elasticsearch] = None) -> int:
    es = es or get_es()
    actions = (
        {
            "_index": settings.es_index,
            "_id": d["meme_id"],
            "_source": d,
        }
        for d in docs
    )
    success, errors = helpers.bulk(es, actions, raise_on_error=False, request_timeout=120)
    if errors:
        logger.warning("Bulk index reported %d errors (showing first): %s", len(errors), errors[:1])
    es.indices.refresh(index=settings.es_index)
    return success


def fused_script_search(
    query_vector: List[float],
    beta: float = 0.5,
    size: int = 500,
    es: Optional[Elasticsearch] = None,
) -> List[Dict[str, Any]]:
    """Exact (bruteforce) fused cosine ranking over the gallery via script_score."""
    es = es or get_es()
    # cosineSimilarity can be negative; shift by +1.0 so script_score stays positive.
    # Final app score is recomputed in Python without the shift.
    source = """
    double sp = cosineSimilarity(params.q, 'vector_proj');
    double so = cosineSimilarity(params.q, 'vector_orig');
    return (1.0 - params.beta) * sp + params.beta * so + 1.0;
    """
    resp = es.search(
        index=settings.es_index,
        size=size,
        query={
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": source,
                    "params": {"q": query_vector, "beta": float(beta)},
                },
            }
        },
        _source=True,
    )
    hits = []
    for h in resp["hits"]["hits"]:
        src = h["_source"]
        src["_score"] = h["_score"]
        hits.append(src)
    return hits


def count_docs(es: Optional[Elasticsearch] = None) -> int:
    es = es or get_es()
    try:
        if not es.indices.exists(index=settings.es_index):
            return 0
        return int(es.count(index=settings.es_index)["count"])
    except Exception as e:
        logger.warning("count_docs failed: %s", e)
        return 0
