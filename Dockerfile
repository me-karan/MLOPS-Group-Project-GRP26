# Use lightweight Python base image
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create application directory
WORKDIR /app


# Copy dependency file first for better layer caching
COPY requirements-api.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Allow the model source to be configurable at build time
ARG HF_MODEL_NAME=g25Ait2048/distilbert-base-uncased
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# Create non-root user before copying application files
RUN useradd --create-home --shell /bin/bash --uid 10001 appuser

# Copy source code (match repository folder name)
COPY Src/ ./src

# Set ownership of application files
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Run application
CMD ["python", "src/inference.py"]