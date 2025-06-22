"""
main.py

Author: Lijo Raju
Purpose: FastAPI app entrypoint for EduRAG.
"""
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import router as api_router 
from app.rag_engine import get_rag_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context Manager for application lifespan events (startup and shutdown).
    Loads the RAG engine on startup and handles its cleanup on shutdown.
    """
    logger.info("Preloading RAG engine at startup...")
    app.state.rag_engine = get_rag_engine()
    logger.info("RAG engine preloaded successfully...")
    yield
    logger.info("Cleaning up RAG engine on application shutdown...")
    if hasattr(app.state.rag_engine, "close"):
        try:
            app.state.rag_engine.close()
            logger.info("RAG engine cleanup successful...")
        except Exception as e:
            logger.error(f"Error during RAG engine cleanup: {e}", exc_info=True)
    else:
        logger.info("No specific cleanup method found for RAG engine.")


app = FastAPI(
    title="EduRAG - Retrieval-Augmented Generation with TinyLlama",
    description="FastAPI backend to serve RAG based educational QA",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "EduRAG backend is running..."}