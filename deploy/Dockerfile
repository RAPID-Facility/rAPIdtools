# rapidtools GUI server image.
#
# Ready-built images are published for every release by
# .github/workflows/container.yml as ghcr.io/rapid-facility/rapidtools-gui
# (CUDA: <version>, latest; CPU: <version>-cpu, latest-cpu). To build locally:
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
# The probe presents the token (the app answers 401 without it) and marks
# itself so it does not count as user activity for idle-shutdown.sh.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD \
    curl -fsS -H "X-Rapidtools-Token: ${RAPIDTOOLS_GUI_TOKEN:-}" -H "X-Rapidtools-Probe: 1" \
        http://127.0.0.1:8765/api/state >/dev/null || exit 1

# The token, SMTP settings and public URL come from the environment (see .env.example).
CMD ["rapidtools-gui", "--host", "0.0.0.0", "--port", "8765", "--no-browser", "--data-root", "/data", "--output-dir", "/output"]
