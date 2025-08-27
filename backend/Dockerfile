FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim
# https://docs.astral.sh/uv/guides/integration/docker

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/

RUN uv sync

COPY . /app/

ENV PYTHONUNBUFFERED='1'
ENV PYTHONPATH="/app"

ENTRYPOINT ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1" ]
