"""
run_validate_qa_dataset.py

Author: Lijo Raju
Purpose: CLI to run QA dataset schema validation

Usage:
    python scripts/run_validate_qa_dataset.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from evaluation.schema.validate_qa_dataset import run_schema_validation

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Validating schema of QA dataset")
    parser.add_argument("--qa_dataset_path", type=str, default="data/qa/train.json", help="Path to the QA dataset")
    args = parser.parse_args()

    try:
        run_schema_validation(path=args.qa_dataset_path)
    except Exception as e:
        logging.error(f"Failed to run validation: {e}")


if __name__ == "__main__":
    main()