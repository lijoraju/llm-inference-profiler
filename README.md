# Charitra: Retrieval-Based Q&A for High School Social Science Studies
A lightweight, LLM-powered question answering system fine-tuned on NCERT Social Science texbooks using QLoRA and deployed via FastAPI + Docker.

🛰️ Live Demo: [Charitra API on Hugging Face Spaces](https://lijoraju-charitra-backend.hf.space/docs)

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [How to Run Locally](#how-to-run-locally)
- [Training & Fine-Tuning](#training--fine-tuning)
- [Project Structure](#project-structure)
- [Sample Queries](#sample-queries)
- [Future Work](#future-work)
- [License](#license)

## 🧠 Overview

**Charitra** is an open-source Retrieval-Augmented Generation (RAG) system designed to help high school students explore answers from *NCERT Class 10 Social Science* material. It uses a fine-tuned version of TinyLlama with QLoRA for domain adaptation and integrates with a FAISS-based semantic retriever.

This project demonstrates how to build and deploy a low-latency, memory-efficient RAG system using open tools and modest compute.

## ✨ Features

- 🔍 Top-k semantic retrieval using FAISS
- 📖 Fine-tuned LLM using QLoRA on curated Q&A pairs
- ⚡ Quantized inference with GGUF and llama.cpp
- 🚀 FastAPI backend with clean REST interface
- 🐳 Dockerized for deployment on Hugging Face Spaces
- 📁 Organized for extensibility and testing

## 🏗️ Architecture

1. PDF NCERT textbook → Chunked & embedded
2. FAISS vectorstore → stores semantic chunks
3. User query → semantic search → top-k chunks
4. Chunks + query → Prompt to TinyLlama (QLoRA fine-tuned)
5. Final answer → served via FastAPI

## 🛠️ Tech Stack

- 🧠 LLM: [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- 🔧 Fine-tuning: QLoRA + PEFT + bitsandbytes
- 📚 Retriever: SentenceTransformers + FAISS
- 🐍 Backend: Python + FastAPI
- 🐳 Deployment: Docker + Hugging Face Spaces
- 📦 Vectorstore: FAISS

## 🧪 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/lijoraju/charitra.git && cd charitra

# Build the docker image
docker build -t charitra-app .

# Run the container
docker run -p 7860:7860 charitra-app
```
Once running, visit http://localhost:7860/docs or send a POST request to /query
```
curl -X POST http://localhost:7860/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the importance of the Indian National Congress in India’s freedom struggle?"}'
```


---

## 📌 Training & Fine-Tuning Summary

- Dataset: 300+ QA pairs auto-generated from NCERT Class X chapters
- Format: OpenChat-style JSON
- Fine-tuning: 4-bit QLoRA on TinyLlama using `transformers` + PEFT
- Output: GGUF quantized model for CPU-based inference

## 🗂️ Project Structure

```
charitra/
├── app/ # FastAPI app
├── pipeline/ # RAG pipeline (embedding, retrieval, model)
├── scripts/ # CLI scripts to test and train
├── data/ # Sample data (docs, QA pairs)
├── tests/ # Unit tests
├── Dockerfile
├── start.sh
└── requirements.txt
```

## 📎 Sample Queries

```bash
Q: What were the causes of the Non-Cooperation Movement?
A: The Non-Cooperation Movement was launched by Gandhi to resist British rule through peaceful means like boycotting schools, goods, and titles...

Q: What is the role of the Constitution in a democracy?
A: The Constitution lays down the framework of governance, ensures rights and justice, and acts as the guardian of democracy...
```


---

## 📌 Future Work
- Add PDF ingestion interface
- Extend to other subjects (Science, English, etc.)
- Integrate with a frontend chatbot (Gradio or Telegram)
- Enable multilingual answers using translation APIs
