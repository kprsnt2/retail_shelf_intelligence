# On-Prem Retail Intelligence Container
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for high-speed package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project specification
COPY pyproject.toml .

# Install dependencies
RUN uv pip install --system -r pyproject.toml

# Copy application source code and benchmark data
COPY src/ /app/src/
COPY data/ /app/data/
COPY BENCHMARK_REPORT.md /app/

EXPOSE 8000

# Start on-prem inference server
CMD ["uvicorn", "retail_shelf.main:app", "--host", "0.0.0.0", "--port", "8000"]
