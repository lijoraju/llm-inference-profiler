"""
run_prepare_dataset.py

Author: Lijo Raju
Purpose: CLI runner to prepare the dataset for QLoRA fine-tuning.

Usage:
    python scripts/run_prepare_dataset.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from pipeline.prepare_dataset import prepare_dataset

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for QLoRA fine-tuning")
    parser.add_argument("--qa_path", type=str, default="data/qa/train.json", help="Path to raw QA JSON file")
    parser.add_argument("--tokenizer_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Tokenizer name or path")
    parser.add_argument("--output_path", type=str, default="data/finetune/processed", help="Directory to save toneized dataset")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length")

    args = parser.parse_args()

    try:
        prepare_dataset(
            qa_path=args.qa_path,
            tokenizer_name=args.tokenizer_name,
            output_path=args.output_path,
            max_length=args.max_length
        )
    except Exception as e:
        logging.error(f"Dataset preparation failed: {e}")


if __name__ == "__main__":
    main()