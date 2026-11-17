# syntax=docker/dockerfile:1.7
FROM scratch AS sdk
ADD https://github.com/mii-llm/palingenesis.git#9ceb075cd6771b50821c38e872c127e32cdbae5d /sdk

FROM ghcr.io/astral-sh/uv:0.11.6 AS uv
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/mii-llm/palingenesis-server" \
      org.opencontainers.image.version="0.1.0"

# Triton and Inductor compile launchers on the GPU host at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 trainer \
    && useradd --create-home --uid 10001 --gid 10001 trainer \
    && install -d -o trainer -g trainer /workspace /state /cache

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PGS_SERVER_WORKDIR=/workspace PGS_SERVER_STATE=/state \
    XDG_CACHE_HOME=/cache HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton

RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=bind,from=sdk,source=/sdk,target=/sdk \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never \
    uv sync --project /sdk --locked --no-dev --extra train --no-editable --no-cache

# Build from a writable copy so setuptools can generate package metadata.
COPY pyproject.toml README.md /build/server/
COPY palingenesis_server/ /build/server/palingenesis_server/
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    uv pip install --no-cache /build/server && rm -rf /build/server \
    && python -c "import torch, palingenesis_server.api; assert torch.version.cuda == '12.9'"

WORKDIR /workspace
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD ["python", "-c", "import os, urllib.request as u; u.urlopen(u.Request('http://127.0.0.1:8000/health', headers={'Authorization': 'Bearer ' + os.environ['PGS_SERVER_TOKEN']}), timeout=3).close()"]
ENTRYPOINT ["palingenesis-server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
