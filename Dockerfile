FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install uv && uv pip install --system -r pyproject.toml

COPY api/ ./api/
COPY config.py .
COPY config.toml .

ENV PYTHONPATH=/app
ENV RUN_MODE=production

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
