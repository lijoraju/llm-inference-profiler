"""
Document ingestion module for extracting and chunking text from PDFs

Author: Lijo Raju
"""

import os 
import fitz
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from all pages.
    """
    try:
        with fitz.open(file_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")
        return ""
    
    
def chunk_text(text: str,
                chunk_size: int = 500,
                chunk_overlap: int = 100) -> List[str]:
    """
    Split raw text into overlapping chunks by LangChain splitter.

    Args:
        text (str): The raw input text.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[str]: List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", ".", "!", "?", " ", ""]
                                              )
    return splitter.split_text(text)


def process_document(file_path: str) -> List[str]:
    """
    Complete pipeline: extract and chunk a single document.

    Args:
        file_path(str): Path to the document.
    
    Returns:
        List[str]: List of text chunks.
    """
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text.strip():
        print(f"⚠️ No contents extracted from: {file_path}")
        return []
    
    chunks = chunk_text(raw_text)
    print(f"✅ {file_path}: Extracted {len(chunks)} chunks.")
    return chunks