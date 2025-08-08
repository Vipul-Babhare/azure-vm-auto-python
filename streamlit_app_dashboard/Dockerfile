# Use Python 3.9 slim image as base
FROM python:3.9-slim
 
# Set working directory in the container
WORKDIR /app
 
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0
 
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
&& rm -rf /var/lib/apt/lists/*
 
# Copy requirements first for better caching
COPY requirements.txt .
 
# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
 
# Copy application code
COPY app.py .
 
# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser
 
# Expose the Streamlit port
EXPOSE 8502
 
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://172.166.194.34:8502/_stcore/health || exit 1
 
# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8502"]