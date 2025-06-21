# ----------------------------------------------------
# EduRAG Dockerfile - FastAPI + Llama-cpp + FAISS
# Author: Lijo Raju
# ----------------------------------------------------

FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Build llama-cpp-python with CMAKE args
ENV LLAMA_CMAKE_ARGS='-DLLAMA_CUBLAS=off'
RUN pip install llama-cpp-python --no-cache-dir

# Make sure scripts are executable
RUN chmod +x /app/start.sh

# Expose FastAPI port
EXPOSE 8000

# Start server
CMD ["/app/start.sh"]