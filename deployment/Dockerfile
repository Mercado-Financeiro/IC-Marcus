# Multi-stage build for ML Trading Pipeline
FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONHASHSEED=42 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash trader && \
    mkdir -p /app/data /app/artifacts /app/logs && \
    chown -R trader:trader /app

WORKDIR /app

# Copy requirements first for better caching
COPY --chown=trader:trader ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=trader:trader . .

# Switch to non-root user
USER trader

# Default command (can be overridden)
CMD ["python", "-m", "src.dashboard.app_enhanced"]

# Training stage
FROM base AS training

# Additional dependencies for training
RUN pip install --no-cache-dir --user \
    optuna-dashboard \
    tensorboard

CMD ["python", "run_optimization.py"]

# API server stage
FROM base AS api

# Install API dependencies
RUN pip install --no-cache-dir --user \
    fastapi \
    uvicorn[standard] \
    python-multipart

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Dashboard stage
FROM base AS dashboard

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/dashboard/app_enhanced.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

# Paper trading stage
FROM base AS trader

CMD ["python", "-m", "src.trading.paper_trader"]

# Development stage
FROM base AS development

# Install development dependencies
RUN pip install --no-cache-dir --user \
    jupyter \
    jupytext \
    ipywidgets \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy \
    pre-commit

EXPOSE 8888 8501

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]