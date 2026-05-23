FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv sync --frozen --no-dev --no-group ui

COPY api/ ./api/
COPY config.py .
COPY config.toml .

ENV PYTHONPATH=/app
ENV RUN_MODE=production

EXPOSE 8000
