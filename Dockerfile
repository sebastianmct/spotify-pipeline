FROM python:3.11-slim

LABEL maintainer="Equipo DataOps Spotify"
LABEL description="Pipeline de ML para clasificación de popularidad de canciones Spotify"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY scripts/ ./scripts/
COPY sql/ ./sql/
COPY data/ ./data/
COPY .env ./.env

RUN mkdir -p logs data/raw data/processed data/validated data/reports data/dashboard data/model data/external

RUN useradd -m -u 1000 pipeline && \
    chown -R pipeline:pipeline /app

USER pipeline

CMD ["python", "--version"]
