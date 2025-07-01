"""
llm_eval.py

Author: Lijo Raju
Purpose: LLM based evaluation for QA dataset quality.
"""

import json
import os
from typing import List, Dict, Tuple
import ollama
import logging
from tqdm import tqdm

from evaluation.utils import load_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_with_llm(qa_pairs: List[Dict[str, str]], model: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Evaluate QA pairs for factual consistency using model via ollama.

    Args:
        qa_pairs (List[Dict[str, str]]): List of dictionaries containing context, question and answers.
        model (str): LLM model to use for evaluation

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: Valid and Invalid QA pairs after LLM evaluation.
    """
    valid, invalid = [], []

    logger.info(f"Begining LLM evaluation using model: {model}")

    for pair in tqdm(qa_pairs, desc="LLM evaluation in progress"):
        context = pair.get("context", "").strip()
        question = pair.get("question", "").strip()
        answer = pair.get("answer", "").strip()

        if not context or not question or not answer:
            continue

        prompt = (
            f"You are a helpful assistant evaluating the quality of a QA pair.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Does the answer correctly and fully respond to the question, based only on the context?"
            f" Reply with 'Yes' or 'No'."
        )

        try:
            judgement = query_model(prompt, model)
        except Exception as e:
            logger.error(f"LLM error on QA pair: {pair}\nError: {e}")
            continue

        if judgement.startswith("yes"):
            valid.append(pair)
        else:
            invalid.append(pair)

    logger.info("Completed LLM evaluation")
    return valid, invalid


def save_results(valid: List[Dict[str, str]], 
                 invalid: List[Dict[str, str]], 
                 output_dir: str = "data/qa/llm_eval"
                ) -> None:
    """
    Save evaluation results to output files.

    Args:
        valid (List[Dict[str:str]]): Valid QA pairs
        invalid (List[Dict[str:str]]): Invalid QA pairs
        output_dir (str): Path to the directory where results will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "valid_qa.json"), "w") as f:
        json.dump(valid, f, indent=2)
    with open(os.path.join(output_dir, "invalid_qa.json"), "w") as f:
        json.dump(invalid, f, indent=2)


def query_model(prompt: str, model: str) -> str:
    """
    Prompt the specified model to generate.

    Args:
        prompt (str): Query to the model
        model (str): Specific model to be queried

    Returns:
        str: Generated response from the model
    """
    return ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])["message"]["content"].strip().lower()


def run_llm_evalaution(path: str, model: str):
    """
    Runs LLM based QA dataset evaluation.

    Args:
        path (str): Path the QA JSON file 
        model (str): LLM model to use for evaluation
    """
    data = load_json(path)

    if data is None:
        logger.error(f"🛑 Aborting validation due to load failure.")
        return 
    
    valid_entries, invalid_entries = evaluate_with_llm(data, model)
    
    logger.info(f"""
    💯total: {len(data)} ❗️invalid: {len(invalid_entries)} ✅passed: {len(valid_entries)}
    """)

    save_results(valid_entries, invalid_entries)