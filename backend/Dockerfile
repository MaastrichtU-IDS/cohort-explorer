FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim
# FROM ghcr.io/astral-sh/uv:python3.10-bookworm
# https://docs.astral.sh/uv/guides/integration/docker/#available-images

# RUN pip install --upgrade pip

WORKDIR /app

COPY pyproject.toml /app/

RUN uv sync

# COPY requirements.txt ./
# RUN pip install -r requirements.txt

COPY . /app/

ENV PYTHONUNBUFFERED='1'

# ENV PYTHONUNBUFFERED='1' \
#     PYTHONPATH=/app \
#     APP_MODULE=src.main:app

# RUN pip install -e .

ENTRYPOINT ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "6" ]
