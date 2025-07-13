"""
run_heuristic_eval.py

Author: Lijo Raju
Purpose: CLI to run QA dataset quality evaluation via heuristics

Usage:
    python scripts/run_heuristic_eval.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from evaluation.qa_quality.heuristic_eval import run_heuristic_based_qa_dataset_evaluation

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Heuristic based QA dataset quality evaluation")
    parser.add_argument("--qa_dataset_path", type=str, default="data/qa/train.json", help="Path to the QA dataset")
    args = parser.parse_args()

    try:
        run_heuristic_based_qa_dataset_evaluation(path=args.qa_dataset_path)
    except Exception as e:
        logging.error(f"Failed to run evaluation: {e}")


if __name__ == "__main__":
    main()