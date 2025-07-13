"""
embed_chunks.py

Author: Lijo Raju
Purpose: CLI script to run embedding pipeline.

Usage:
    python scripts/embed_chunks.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.embed import run_embedding_pipeline
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

if __name__ == "__main__":
    run_embedding_pipeline()