"""
ingest.py

Author: Lijo Raju
Purpose: Extract text from PDF files and split into token-limited chunks for embedding.

Usage:
- Called via scripts/run_refresh.py to process PDF into chunked documents.
"""

import logging
import pickle
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Setup logging
logging.basicConfig (
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Constants
OUTPUT_PATH = Path("data/vectorstore/chunks.pkl")
CHUNK_SIZE=1000
CHUNK_OVERLAP=200


def load_pdfs(directory: Path) -> List[Document]:
    """
    Loads and extracts text from PDF files in the specified directory.

    Args:
        directory (Path): Path to the directory containing PDF files.
    
    Returns:
        List[str]: List of text documents extracted from each PDF.
    """
    if not directory.exists():
        logging.error(f"Directory not found: {directory}")
        return []
    
    documents = []
    for file in directory.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(file))
            pages = loader.load()
            documents.extend(pages)
            logging.info(f"Loaded {len(pages)} pages from {file.name}")
        except Exception as e:
            logging.error(f"Failed to load {file.name}: {e}")
    
    return documents
    

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Args:
        documents (List[str]): List of raw text documents.

    Returns:
        List[str]: List of token limited chunks. 
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    try:
        chunks = splitter.split_documents(documents)
        logging.info(f"Split into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logging.error(f"Failed to split documents: {e}")
        return []


def save_chunks(chunks: List[Document], output_path: Path) -> None:
    """
    Save the list of chunks to disk using pickle.

    Args:
        chunks (List[str]): List of text chunks.
        output_path (Path): Destination file path for pickle file.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(chunks, f)
        logging.info(f"Saved {len(chunks)} chunks to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save chunks: {e}")


def run_ingestion(path: str) -> None:
    """
    End-to-end ingestion pipeline to extract, split, and save chunks from PDFs.

    Args:
        path (str): Path to PDF documents directory.
    """
    logging.info(f"Starting injection pipeline...")
    dir_path = Path(path)
    docs = load_pdfs(dir_path)
    if not docs:
        logging.warning(f"No documents to process.")
        return

    chunks = chunk_documents(docs)
    if not chunks:
        logging.warning(f"No chunks to save.")
        return
    
    save_chunks(chunks, OUTPUT_PATH)
    logging.info(f"Ingestion pipeline completed.")