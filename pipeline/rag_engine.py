"""
rag_engine.py

Author: Lijo Raju
Purpose: End-to-End RAG engine for EduRAG.
    Combines: 
        - FAISS retriever
        - Prompt template formatter
        - GGUF model inference
    Used by FastAPI backend for answering user queries.
"""
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.retrieve import load_retriever, retrieve
from pipeline.prompt_template import format_prompt
from pipeline.llm_gguf import GGUFModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieval-Augmented Generation engine for inference."""

    def __init__(self, model_path: str, k: int):
        """
        Initialize RAG engine components.

        Args:
            model_path (str): Path to GGUF model.
            k (int): Number of top documents to retrieve.
        """
        try:
            logger.info("Initializing GGUF model...")
            self.llm = GGUFModel(model_path)

            logger.info("Loading FAISS retriever...")
            self.retriever = load_retriever(top_k=k)
        except Exception as e:
            logger.exception(f"Failed to initialize RAG engine: {e}")
            raise RuntimeError("RAG engine initialization failed.")
    
    def run(self, query: str) -> str:
        """
        Run full RAG pipeline: Retrieve -> Prompt -> Generate.

        Args:
            query (str): User question.
        
        Returns:
            str: Generated response from LLM.
        """
        try:
            logger.info("Running retrieval for query: %s", query)
            retrieved_docs = retrieve(self.retriever, query)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            prompt = format_prompt(query, context)

            logger.info(f"Generating response from model...")
            response = self.llm.generate(prompt)
            return response
        except Exception as e:
            logger.exception("RAG pipeline failed: %s", e)
            raise RuntimeError("Failed to generate response.")
        
