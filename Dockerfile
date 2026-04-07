# Clinical Trial Matcher Environment - Hugging Face Spaces Dockerfile
# This Dockerfile is optimized for deployment on Hugging Face Spaces

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/server/requirements.txt

# Copy application code
COPY models.py /app/models.py
COPY client.py /app/client.py
COPY baseline.py /app/baseline.py
COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml
COPY server/ /app/server/

# Create __init__.py files
RUN touch /app/__init__.py /app/server/__init__.py

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the FastAPI application
# Note: HF Spaces expects the app to run on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
