FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev --only-group ui

COPY ui/ ./ui/

EXPOSE 7860

CMD ["/app/.venv/bin/python", "ui/app.py"]
