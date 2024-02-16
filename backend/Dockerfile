FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app/

ENV PYTHONUNBUFFERED='1'

RUN pip install --upgrade pip

# COPY requirements.txt ./
# RUN pip install -r requirements.txt

COPY . /app/

ENV PYTHONPATH=/app

RUN pip install .
