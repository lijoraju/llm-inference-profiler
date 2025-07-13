#!/bin/bash

# ----------------------------------------------------
# Start script for EduRAG FastAPI backend (Docker)
# Author: Lijo Raju
# ----------------------------------------------------

echo "🚀 Starting EduRAG FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1