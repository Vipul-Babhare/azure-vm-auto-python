# Use Python 3.12 slim image as base
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Environment variables (can be overridden at runtime)
ENV JWT_SECRET_KEY=your-secret-key-change-this
ENV MODEL_SERVING_URL=http://localhost:8083

# Expose port
EXPOSE 8083

# Healthcheck inside container
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8083/health || exit 1

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8083"]
