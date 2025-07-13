"""
run_refresh.py

Author: Lijo Raju
Purpose: CLI utility to run document ingestion pipeline.

Usage:
    python scripts/run_refresh.py 
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse 
import logging
from pipeline.ingest import run_ingestion

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Document ingestion")
    parser.add_argument("--document_path", type=str, default="data/sample_docs", help="Path to PDF document directory")
    args = parser.parse_args()

    try:
        run_ingestion(path=args.document_path)
    except Exception as e:
        logging.error(f"Document ingestion failed with error: {e}")


if __name__ == "__main__":
    main()