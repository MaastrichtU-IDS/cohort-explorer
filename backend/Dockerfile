FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
# FROM python:3.10

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /app/

ENV PYTHONUNBUFFERED='1' \
    PYTHONPATH=/app \
    APP_MODULE=src.main:app

RUN pip install -e .

# ENTRYPOINT [ "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80" ]
