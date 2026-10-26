FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
# https://docs.astral.sh/uv/guides/integration/docker

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen

# spaCy English model (used by CohortVarLinker.src.fuzz_match), installed into the project venv
# so spacy.load("en_core_web_sm") resolves at runtime. `python -m spacy download` shells out to
# pip, which the uv-managed venv does not contain, so the wheel is installed with uv directly.
# The model version must match the locked spaCy minor (3.8.x).
RUN uv pip install --python /app/.venv/bin/python \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

COPY . /app/

ENV PYTHONUNBUFFERED='1'
ENV PYTHONPATH="/app"

ENTRYPOINT ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4" ]
