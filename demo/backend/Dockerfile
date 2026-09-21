# Use a locally cached PyTorch image to avoid Docker Hub rate limits.
FROM cnstark/pytorch:2.0.1-py3.9.17-cuda11.8.0-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch/torchvision already in base image
RUN pip install --no-cache-dir \
        fastapi==0.115.6 \
        "uvicorn[standard]==0.34.0" \
        elasticsearch==8.17.0 \
        ftfy==6.3.1 \
        regex==2024.11.6 \
        pydantic-settings==2.7.1 \
        httpx==0.28.1 \
        tqdm==4.67.1 \
        "git+https://github.com/openai/CLIP.git"

COPY app ./app
COPY scripts ./scripts
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    CARE_DEVICE=cpu \
    CARE_ES_URL=http://elasticsearch:9200

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
