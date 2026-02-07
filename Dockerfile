# JD Parser Service - Multi-stage Docker Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY *.py ./
COPY models.py ./
COPY config.py ./

# Create necessary directories
RUN mkdir -p faiss_index logs

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# Don't switch user for now - causes permission issues
# USER appuser

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Run FastAPI app
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
