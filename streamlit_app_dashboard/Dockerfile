# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit-specific config: listen on all interfaces at port 8502
EXPOSE 8502

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
