"""
retrieve.py

Author: Lijo Raju
Purpose: Load FAISS index and perform top-k similarity-based retrieval
         for input queries using LangChain retriever.

Usage:
    from pipeline.retrieve import load_retriever, retrieve_top_k
"""

import logging
import os 
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
VECTORSTORE_DIR = "data/vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_DIR, "faiss_index")
METADATA_FILE = os.path.join(VECTORSTORE_DIR, "index.pkl")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_retriever(top_k: int = 3) -> FAISS:
    """
    Load FAISS index and return a retriever.

    Args:
        top_k (int): Number of top documents to retrieve.
    
    Returns:
        FAISS: A retriever object.
    """
    try:
        logging.info("Loading FAISS vectorstore...")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        vectorstore = FAISS.load_local(
            folder_path=INDEX_FILE,
            embeddings=embedding_model, 
            allow_dangerous_deserialization=True
            )
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        logger.info("Retriever loaded successfully.")
        return retriever
    except Exception as e:
        logging.error("Failed to load retriever: {e}")
        raise 


def retrieve_top_k(query: str, top_k: int = 3) -> List[Document]:
    """
    Retrieve top-k relevant documents for the given query.

    Args:
        query (str): Input query string.
        top_k (int): Number of results to return.

    Return:
        List[Document]: Top-k retrieved documents with metadata.
    """
    retriever = load_retriever(top_k=top_k)
    try:
        logger.info(f"Retrieving top {top_k} documents for query: '{query}'")
        docs = retriever.get_relevant_documents(query)
        return docs
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise