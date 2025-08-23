# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Build arguments
ARG MODEL_VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

# Labels
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.version=$MODEL_VERSION \
      org.label-schema.name="crypto-ml" \
      org.label-schema.description="Cryptocurrency ML Trading System"

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    mkdir -p /app/artifacts /app/logs /app/data && \
    chown -R mluser:mluser /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/mluser/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=mluser:mluser src/ ./src/
COPY --chown=mluser:mluser configs/ ./configs/
COPY --chown=mluser:mluser scripts/ ./scripts/

# Switch to non-root user
USER mluser

# Add local bin to PATH
ENV PATH=/home/mluser/.local/bin:$PATH

# Environment variables for reproducibility
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=/app/artifacts/mlruns \
    TZ=UTC

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "src.api.server"]