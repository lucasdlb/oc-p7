FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv sync --frozen --no-dev --no-group ui

COPY api/ ./api/
COPY scripts/ ./scripts/
COPY config.py .
COPY config.toml .
COPY logging_config.py .

ENV PYTHONPATH=/app
ENV RUN_MODE=production

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
