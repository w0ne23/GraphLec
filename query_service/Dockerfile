FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

ENV PYTHONPATH=/app

COPY query_service/ ./query_service/
COPY app/backend/pipeline/lance_ingest.py ./pipeline/lance_ingest.py
COPY app/backend/pipeline/embedding_utils.py ./pipeline/embedding_utils.py
RUN touch ./pipeline/__init__.py
COPY app/backend/pipeline/utils.py ./pipeline/utils.py

CMD ["uvicorn", "query_service.main:app", "--host", "0.0.0.0", "--port", "8001"]