# AIMER

**AIMER** (**A**ffective-**I**nfused **Me**me **R**etrieval) is a text-to-meme image retrieval system. This repository contains the code for retrieval experiments on two meme datasets:

- **imgflip** — English memes
- **douban** — Chinese memes

The implemented pipeline consists of CLIP ViT-B/32 first-stage retrieval, optional image-emotion fusion (IEF), and reranking with either an MLP or a Jina-reranker-M0 score head.

## Installation

Install the Python dependencies in an environment with a CUDA-enabled PyTorch installation:

```bash
pip install -r requirements.txt
```

The shell scripts under `scripts/` are convenience wrappers for the development Docker container. They use these defaults:

| Variable | Default | Description |
|---|---|---|
| `DOCKER_NAME` | `cuda_1111` | Docker container name |
| `CONDA_ENV` | `neo_meme` | Conda environment inside the container |
| `CUDA_VISIBLE_DEVICES` | `0` | Visible GPU |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Hugging Face endpoint |

To run without Docker, use the Python commands printed by `DRY_RUN=1 bash scripts/<script>.sh`.

## Datasets

Datasets and images are not included in Git because of their size and distribution terms. Prepare them in the MSRVTT-compatible layout expected by `dataset.py`:

```text
imgflip_data/
├── msrvtt/
│   ├── train_data.json
│   ├── train_emotion.json
│   ├── test_data.json
│   ├── test_emotion.json
│   └── train_ids.csv
└── images/

douban_data/
├── input_file/
│   ├── train_data.json
│   ├── train_emotion.json
│   ├── test_data.json
│   ├── test_emotion.json
│   └── train_ids.csv
└── image/
```

The original preprocessing scripts have intentionally been left in their existing dataset directories for this cleanup pass.

## Training and evaluation

```bash
# imgflip CLIP fine-tuning and evaluation
bash scripts/train_imgflip.sh
bash scripts/infer_imgflip.sh

# douban CLIP fine-tuning and evaluation
bash scripts/train_douban.sh
bash scripts/infer_douban.sh

# douban Jina-reranker-M0 score head
bash scripts/train_jina_douban.sh
bash scripts/eval_jina_douban.sh
```

Common overrides:

| Variable | Default | Description |
|---|---|---|
| `NUM_SAMPLES` | `0` | Number of evaluation queries; `0` means all queries |
| `SAMPLE_MODE` | `first` | `first` or `random` |
| `BATCH_SIZE` | model-specific | Training/inference batch size |
| `EPOCHS` / `LR` | model-specific | Training hyperparameters |
| `SAVE_DIR` | model-specific | Output/cache directory |
| `FORCE_TRAIN=1` | — | Retrain even if a checkpoint exists |
| `DRY_RUN=1` | — | Print the Docker command without running it |
| `JINA_API_KEY` | empty | Only needed when explicitly using the Jina API path |

## Trained artifacts

Large binaries are not committed. The default local paths are:

| Artifact | Path |
|---|---|
| Fine-tuned imgflip CLIP | `./clip_imgflip/clip_imgflip.pt` |
| Fine-tuned douban CLIP | `./clip_douban/clip_douban.pt` |
| Douban Jina score head | `./checkpoints_jina_douban_topk25_768/jina_m0_lora_best_douban.pt` |
| Jina training cache | `./checkpoints_jina_douban_topk25_768/jina_features_cache.pt` |
| Jina evaluation cache | `./checkpoints_jina_douban_topk25_768/jina_features_eval_cache.pt` |
| Ridge image projection | `./autoresearch_cache_douban/image_proj_ridge_l1.pt` |

The default hyperparameters in the code and the launch scripts are the final tuned values for each dataset (e.g., the douban best configuration: projection mix 0.9 and similarity fusion 0.5).

## Demo

The `demo/` directory contains an optional FastAPI + Elasticsearch + static frontend demo of the Douban retrieval system.

```bash
cd demo
docker compose up --build -d
```

The demo mounts local datasets and checkpoints; these files are not bundled with the source release. See `demo/README.md` for details.

## Repository layout

```text
.
├── dataset.py                 # Unified MSRVTT-style dataset
├── models/                    # CLIP, IEF, rerankers, and baselines
├── scripts/                   # Training and evaluation launchers
├── demo/                      # Optional retrieval demo
├── requirements.txt
├── LICENSE
└── README.md
```

## License

This project is released under the [MIT License](LICENSE).
