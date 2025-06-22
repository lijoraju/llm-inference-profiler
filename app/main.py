"""
main.py

Author: Lijo Raju
Purpose: FastAPI app entrypoint for EduRAG.
"""
import logging
from fastapi import FastAPI

from app.api import router as api_router 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EduRAG - Retrieval-Augmented Generation with TinyLlama",
    description="FastAPI backend to serve RAG based educational QA",
    version="1.0.0"
)

app.include_router(api_router)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "EduRAG backend is running..."}