# CARE — CLIP + Elasticsearch Meme Retrieval

前后端分离的豆瓣梗图检索服务。检索管线对齐 `scripts/infer_douban.sh` 最优配置：

- CLIP ViT-B/32（`clip_douban.pt`）
- Ridge 图像投影（`image_proj_ridge_l1.pt`，`projection_mix=0.9`）
- 投影/原始相似度融合（`β=0.5`）
- 文本侧 Pseudo-Relevance Feedback（`α=0.03`，k=`3,2,2`，3 轮）

向量存于 Elasticsearch `dense_vector`；在线只编码 query 文本。

## 目录结构

```
demo/
├── docker-compose.yml
├── backend/           # FastAPI + CLIP + ES
│   ├── app/
│   │   ├── main.py        # API
│   │   ├── encoder.py     # CLIP + projection
│   │   ├── search.py      # 检索 + 翻页
│   │   ├── es.py          # ES 索引 / kNN
│   │   └── config.py
│   └── scripts/index_memes.py
├── frontend/          # 静态页 + nginx 反代
├── models/            # 模型软链接
│   ├── clip_douban.pt
│   ├── image_proj_ridge_l1.pt
│   └── test_cache.pt      # 预计算图像特征（加速建库）
└── data/
    ├── metadata.json      # meme 元数据
    └── images/            # → douban_data/image
```

## 快速启动

```bash
cd demo
docker compose up --build -d
```

首次启动后端会自动把约 4108 条 meme 向量写入 ES（优先用 `test_cache.pt`，约 1–2 分钟）。

| 服务 | 地址 |
|------|------|
| 前端 | http://localhost:8080 |
| API  | http://localhost:8010 |
| ES   | http://localhost:9200 |

### API

```bash
# 检索（翻页）
curl 'http://localhost:8010/api/search?q=吐槽过线普通同学&page=1&page_size=12'

# 健康检查
curl http://localhost:8010/api/health
```

响应字段：`total` / `page` / `page_size` / `total_pages` / `results[{rank,meme_id,gt,score,image_url}]`。

### 重建索引

```bash
docker compose exec backend python -m scripts.index_memes --recreate
```

从原图重新编码（不用 cache）：

```bash
docker compose exec backend python -m scripts.index_memes --recreate --from-images
```

## 本地开发（不经过 Docker 跑 API）

需本机已有 Elasticsearch（或只起 ES）：

```bash
docker compose up -d elasticsearch
cd backend
pip install -r requirements.txt
export CARE_ES_URL=http://localhost:9200
export CARE_CLIP_CHECKPOINT=../models/clip_douban.pt
export CARE_IMAGE_PROJECTION_PATH=../models/image_proj_ridge_l1.pt
export CARE_FEATURE_CACHE_PATH=../models/test_cache.pt
export CARE_METADATA_PATH=../data/metadata.json
export CARE_IMAGE_DIR=../data/images
python -m scripts.index_memes
uvicorn app.main:app --reload --port 8000
```

前端可直接用任意静态服务器，或 `docker compose up frontend`。

## 说明

- 在线服务只做 **文本侧** query expansion；评估脚本里的全库 image-side expansion 不适合单 query API。
- 检索用 Elasticsearch `script_score` 做投影/原始特征余弦融合（默认候选池 1000），再在池内做 PRF 与翻页。
- 图片与权重通过挂载引用仓库内 `douban_data` / `archive` / `autoresearch`，不复制 1.5G+ 数据。
- 本机磁盘较满时，compose 已关闭 ES disk watermark；否则可能出现分片无法分配（cluster red）。
- 首次查询会下载 OpenAI CLIP ViT-B/32 底座权重（约 338MB），之后缓存在 `torch_cache` volume。
- API 默认映射到 **8010**（避免与本机已占用的 8000/8001 冲突）。

若软链接断裂，请重新执行：

```bash
ln -sfn "$(pwd)/../archive/clip_douban/clip_douban.pt" models/clip_douban.pt
ln -sfn "$(pwd)/../autoresearch/autoresearch_cache_douban/image_proj_ridge_l1.pt" models/image_proj_ridge_l1.pt
ln -sfn "$(pwd)/../autoresearch/autoresearch_cache_douban/test_cache.pt" models/test_cache.pt
ln -sfn "$(pwd)/../douban_data/image" data/images
```
