"""
rag_engine.py

Author: Lijo Raju
Purpose: Singleton loader for RAGEngine in FastAPI app.
"""
import logging
from typing import Optional

from pipeline.rag_engine import RAGEngine
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_rag_engine_instance: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """
    Load and return singleton RAGEngine instance.

    Returns:
        RAGEngine: The initialized RAG engine.
    """
    global _rag_engine_instance

    if _rag_engine_instance is None:
        try:
            logger.info("Initializing RAG engine...")
            _rag_engine_instance = RAGEngine(
                model_path=settings.GGUF_MODEL_PATH,
                k=settings.TOP_K
            )
            logger.info("RAG engine loaded and is ready...")
        except Exception as e:
            logger.exception(f"Failed to initialize RAG engine: {e}")
            raise
    
    return _rag_engine_instance