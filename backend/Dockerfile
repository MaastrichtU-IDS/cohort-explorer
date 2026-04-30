FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim
# https://docs.astral.sh/uv/guides/integration/docker

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen

# spaCy English model (used by updated_CohortVarLinker.fuzz_match).
# Installed as a pip-distributed package so spacy.load("en_core_web_sm") resolves at runtime.
RUN uv run python -m spacy download en_core_web_sm

COPY . /app/

ENV PYTHONUNBUFFERED='1'
ENV PYTHONPATH="/app"

ENTRYPOINT ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "6" ]
