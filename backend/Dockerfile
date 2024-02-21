FROM python:3.10
# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

ENV PYTHONUNBUFFERED='1'
    # PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN apt-get update && \
    apt-get install -y build-essential wget python3-dev && \
    # zlib1g-dev make https://stackoverflow.com/questions/49142409/i-cant-install-specific-version-1-0-2g-of-openssl-in-docker
    # apt-get install -y openssl=3.0.2 libssl-dev python3-dev python3-openssl libsasl2-dev libldap2-dev && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Install openssl from source to get specific version
ARG OPENSSL_VERSION=3.0.2
WORKDIR /tmp
RUN wget https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz -O - | tar -xz
WORKDIR /tmp/openssl-${OPENSSL_VERSION}
RUN ./config --prefix=/usr/local/openssl --openssldir=/usr/local/openssl && make && make install
RUN cat <<'EOT' | tee /etc/ld.so.conf.d/openssl-${OPENSSL_VERSION}.conf
/usr/local/ssl/lib64
EOT

WORKDIR /app

# COPY requirements.txt ./
# RUN pip install -r requirements.txt

COPY . /app/

ENV PYTHONPATH=/app \
    APP_MODULE=src.main:app

RUN pip install -e .

ENTRYPOINT [ "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80" ]
