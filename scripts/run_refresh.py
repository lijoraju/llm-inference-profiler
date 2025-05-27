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

from pipeline.ingest import run_ingestion


if __name__ == "__main__":
    run_ingestion()