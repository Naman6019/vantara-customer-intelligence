# Base image
FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY config/ config/
COPY src/ src/
COPY api/ api/
COPY frontend/ frontend/
COPY models_artifacts/ models_artifacts/
COPY data/ data/
COPY tests/ tests/
COPY pytest.ini .
COPY README.md .

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
