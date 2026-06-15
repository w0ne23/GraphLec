# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY app/backend/recommender/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY app/backend/recommender/ ./recommender/

ENV PYTHONPATH=/app

CMD ["uvicorn", "recommender.recommender_web:app", "--host", "0.0.0.0", "--port", "8002"]
