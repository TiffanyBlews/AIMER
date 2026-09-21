"""CARE meme retrieval API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import settings
from . import es as es_mod
from .search import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("care")

app = FastAPI(title="CARE", description="CLIP + Elasticsearch meme retrieval", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    page: int = Field(1, ge=1)
    page_size: int = Field(12, ge=1, le=50)


@app.on_event("startup")
def startup() -> None:
    try:
        es_mod.wait_for_es(timeout=180)
        n = es_mod.count_docs()
        logger.info("ES index '%s' has %d documents", settings.es_index, n)
    except Exception as e:
        logger.warning("ES not ready at startup: %s", e)


@app.get("/api/health")
def health():
    try:
        n = es_mod.count_docs()
        es_ok = True
    except Exception:
        n = 0
        es_ok = False
    return {"status": "ok" if es_ok else "degraded", "es": es_ok, "indexed": n}


@app.get("/api/search")
def search_get(
    q: str = Query(..., min_length=1, max_length=500),
    page: int = Query(1, ge=1),
    page_size: int = Query(12, ge=1, le=50),
):
    if es_mod.count_docs() == 0:
        raise HTTPException(503, "Index is empty. Run: python -m scripts.index_memes")
    return retrieve(q, page=page, page_size=page_size)


@app.post("/api/search")
def search_post(body: SearchRequest):
    if es_mod.count_docs() == 0:
        raise HTTPException(503, "Index is empty. Run: python -m scripts.index_memes")
    return retrieve(body.query, page=body.page, page_size=body.page_size)


@app.get("/api/images/{filename}")
def get_image(filename: str):
    # Prevent path traversal
    name = Path(filename).name
    if not name.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
        raise HTTPException(400, "Unsupported image type")
    path = Path(settings.image_dir) / name
    if not path.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@app.get("/api/stats")
def stats():
    return {
        "indexed": es_mod.count_docs(),
        "index": settings.es_index,
        "projection_mix": settings.projection_mix,
        "sim_fusion_beta": settings.sim_fusion_beta,
    }
