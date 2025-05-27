"""
embed.py

Author: Lijo Raju
Purpose: Embeds chunked documents using MiniLM and stores them in FAISS.
"""

import os
import pickle
import logging
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

VECTORSTORE_DIR = "data/vectorstore/"
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
FAISS_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")
INDEX_PKL_PATH = os.path.join(VECTORSTORE_DIR, "index.pkl")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(pickle_path: str) -> List[Document]:
    """
    Load chunked documents from a pickle file.

    Args:
        pickle_path (str): Path to the pickle file.

    Returns:
        List[Document]: List of LangChain Document objects.
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Missing file: {pickle_path}")

    try:
        with open(pickle_path, "rb") as f:
            chunks = pickle.load(f)
        logging.info(f"Loaded {len(chunks)} chunks from {pickle_path}")
        return chunks
    except Exception as e:
        logging.error(f"Error loading pickle file.")
        raise e 
    

def embed_chunks(chunks: List[Document]) -> FAISS:
    """
    Embed documents using HuggingFaceEmbeddings and store in FAISS.

    Args:
        chunks (List[Document]): List of chunked documents.
    
    Returns:
        FAISS: FAISS vector store.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        faiss_index = FAISS.from_documents(chunks, embedding=embeddings)
        logging.info(f"Successfully created FAISS index.")
        return faiss_index
    except Exception as e:
        logging.error("Embedding Failed.")
        raise e



def save_faiss_index(faiss_index: FAISS, index_dir: str, index_pkl_path: str) -> None:
    """
    Save the FAISS index and metadata.

    Args:
        faiss_index (FAISS): FAISS index object.
        index_dir (str): Directory to save FAISS index.
        index_pkl_path (str): Path to save metadata. 
    """
    os.makedirs(index_dir, exist_ok=True)
    try:
        faiss_index.save_local(folder_path=index_dir)
        with open(index_pkl_path, "wb") as f:
            pickle.dump(faiss_index.index_to_docstore_id, f)
        logging.info(f"Saved FAISS index to {index_dir}")
        logging.info(f"Saved metadata to {index_pkl_path}")
    except Exception as e:
        logging.error("Error saving FAISS index or metadata.")
        raise e


def run_embedding_pipeline():
    """
    Full embedding pipeline runner.
    """
    logging.info("Starting embedding pipeline...")
    chunks = load_chunks(CHUNKS_PATH)
    faiss_index = embed_chunks(chunks)
    save_faiss_index(faiss_index, FAISS_PATH, INDEX_PKL_PATH)
    logging.info("Embedding pipeline completed.")