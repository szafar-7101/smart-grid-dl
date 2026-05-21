# Smart Grid DL — multi-stage Dockerfile
# Exposes port 8000 (FastAPI) and 8501 (Streamlit).
# Both services are started by start.sh.

FROM python:3.11-slim

# Suppress Python bytecode and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies required by TensorFlow and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer is cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Install the project package in editable mode so `src.*` imports resolve
RUN pip install --no-cache-dir -e .

# Make the startup script executable
RUN chmod +x start.sh

EXPOSE 8000 8501

CMD ["./start.sh"]
