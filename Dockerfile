ARG PG_MAJOR=16
FROM postgres:${PG_MAJOR}

ARG Q3C_VERSION=2.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    postgresql-server-dev-${PG_MAJOR} \
    zlib1g-dev \
    liblz4-dev \
    libzstd-dev \
    libkrb5-dev \
    libreadline-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/segasai/q3c.git /tmp/q3c \
    && cd /tmp/q3c \
    && make \
    && make install \
    && rm -rf /tmp/q3c
