# Clinical Trial Matcher Environment - Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Copy application code
COPY models.py /app/models.py
COPY server/ /app/server/
RUN mkdir -p /app/server/static
COPY server/static/ /app/server/static/

# Create __init__.py for server module
RUN touch /app/server/__init__.py

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
