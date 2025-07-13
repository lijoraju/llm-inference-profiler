"""
run_generate_qa.py

Author: Lijo Raju
Purpose: CLI to QA generation.

Usage:
    python scripts/run_generate_qa.py 
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from pipeline.generate_qa import generate_dataset

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from document chunks")
    parser.add_argument("--chunks_path", type=str, default="data/vectorstore/chunks.pkl", help="Path to chunks.pkl")
    parser.add_argument("--save_path", type=str, default="data/qa/train.json", help="Path to save generated QA pairs")
    parser.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta", help="HuggingFace model name")
    args = parser.parse_args()
    
    try:
        generate_dataset(chunks_path=args.chunks_path, output_path=args.save_path)
    except Exception as e:
        logging.error(f"Failed to run QA generation: {e}")


if __name__ == "__main__":
    main()