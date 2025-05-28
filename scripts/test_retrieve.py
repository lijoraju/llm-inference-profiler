"""
test_retrieve.py

Author: Lijo Raju
Purpose: CLI to test document retrieval from FAISS vectorstore.

Usage:
    python scripts/test_retrieve.py --query "What is photosynthesis?"
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from pipeline.retrieve import retrieve_top_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test document retrieval.")
    parser.add_argument("--query", type=str, required=True, help="Input query string")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top documents to retrieve")
    args = parser.parse_args()

    try:
        results = retrieve_top_k(query=args.query, top_k=args.top_k)
        print(f"\n🔍 Top {args.top_k} results for query: \"{args.query}\"\n")
        for i, doc in enumerate(results):
            print(f"--- Result {i + 1} ---")
            print(f"Page #: {doc.metadata.get('page')}")
            print(f"Content:\n{doc.page_content.strip()}\n")
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {e}")
    

if __name__ == "__main__":
    main()