"""
config.py

Author: Lijo Raju
Purpose: Configuration file for FastAPI RAG backend.
         Stores static paths and shared constants.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuartion settings."""

    GGUF_MODEL_PATH: str = "tinyllama-merged.gguf"
    FAISS_INDEX_PATH: str = "data/vectorstore/faiss_index"
    TOP_K: int = 3

    class Config:
        env_file = ".env"


settings = Settings()