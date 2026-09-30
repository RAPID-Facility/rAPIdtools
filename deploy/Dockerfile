# rapidtools GUI server image.
#
# GPU (default): a PyTorch CUDA runtime base, so SAM 3 and the local models
# (Gemma 4, Qwen, Llama) run on the card the host passes through.
#   docker build -t rapidtools-gui -f deploy/Dockerfile .
# CPU only (cloud model APIs, slow SAM 3):
#   docker build -t rapidtools-gui -f deploy/Dockerfile \
#       --build-arg BASE_IMAGE=python:3.11-slim --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}
ARG TORCH_INDEX=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    RAPIDTOOLS_GUI_NO_BROWSER=1

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# CPU builds install a CPU-only torch first so the rapidtools install does not pull CUDA wheels.
RUN if [ -n "$TORCH_INDEX" ]; then pip install torch torchvision --index-url "$TORCH_INDEX"; fi

# Install rapidtools from the repository checkout that contains this file.
WORKDIR /src
COPY . /src
RUN pip install .

# Mount points: imagery colleagues may load, results, and caches (route database, model weights).
RUN mkdir -p /data /output /cache/rapidtools /cache/huggingface
ENV HOME=/cache
VOLUME ["/data", "/output", "/cache"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fsS http://127.0.0.1:8765/ >/dev/null || exit 1

# The token, SMTP settings and public URL come from the environment (see .env.example).
CMD ["rapidtools-gui", "--host", "0.0.0.0", "--port", "8765", "--no-browser", "--data-root", "/data", "--output-dir", "/output"]
