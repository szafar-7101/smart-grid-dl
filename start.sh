#!/bin/bash
# Starts the FastAPI backend and Streamlit dashboard simultaneously.
# FastAPI runs in the background; the shell is replaced by Streamlit (exec)
# so container signals are forwarded correctly.

set -e

echo "Starting FastAPI backend on port 8000 …"
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit dashboard on port 8501 …"
exec streamlit run app/dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false
