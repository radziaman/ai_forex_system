FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config.yaml .
COPY dashboard.html .

# Create data directories
RUN mkdir -p data/logs data/trades data/historical models

# Expose dashboard port
EXPOSE 8000

# Run the system
CMD ["python", "-m", "src.main"]
