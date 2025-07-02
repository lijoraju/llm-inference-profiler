"""
run_llm_eval.py

Author: Lijo Raju
Purpose: CLI to run QA dataset quality evaluation via LLM

Usage:
    python scripts/run_llm_eval.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from evaluation.qa_quality.llm_eval import run_llm_evalaution
from evaluation.qa_quality.llm_eval_ollama import run_llm_evaluation as run_llm_evaluation_ollama

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="LLM based QA dataset quality evaluation")
    parser.add_argument("--qa_dataset_path", type=str, default="data/qa/train.json", help="Path to the QA dataset")
    parser.add_argument("--model", type=str, default="llama3.1:70b", help="LLM to use for evaluation")
    parser.add_argument("--ollama_flag", type=bool, default=True, help="Set True for Ollama based evaluation")
    args = parser.parse_args()

    try:
        ollama_flag = args.ollama_flag
        if ollama_flag:
            run_llm_evaluation_ollama(path=args.qa_dataset_path, model=args.model)
        else:
            run_llm_evalaution(path=args.qa_dataset_path, model=args.model)
    except Exception as e:
        logging.error(f"Failed to run evaluation: {e}")


if __name__ == "__main__":
    main()