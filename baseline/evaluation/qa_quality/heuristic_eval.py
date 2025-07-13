"""
heuristic_eval.py

Author: Lijo Raju
Purpose: Rule based QA dataset quality evaluation.
"""

import logging
import re
from typing import Dict, List, Tuple
from evaluation.utils import load_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_question_about_figure(question: str) -> bool:
    """
    Checks if question refers to a figure (eg: "Fig 1." or "figure above")

    Args:
        question (str): The question text.
    
    Returns:
        bool: True if figure is referenced, else False
    """
    figure_pattern = [
        r"\bfig(ure)?\.?\s*\d+\b",
        r"\bthe figure\b",
        r"\bshown below\b",
        r"\bimage\b"
    ]
    return any(re.search(pattern, question.lower()) for pattern in figure_pattern)


def is_answer_na(answer: str) -> bool:
    """
    Checks if the answer is a non-informative placeholder like N/A

    Args:
        answer (str): The answer text.

    Returns:
        bool: True if the answer is invalid.
    """
    return answer.strip().lower() in {"n/a", "not available", "not given", "no answer", "not applicable"}


def is_generic_answer(answer: str) -> bool:
    """
    Checks if the answer contains generic, non specific phrases.

    Args:
        answer (str): The answer text.

    Returns:
        bool: True if answer is generic.
    """
    generic_phrases = [
        "i'm not sure", "i cannot answer", "there is no specific answer",
        "it depends", "i don't know", "not applicable"
    ]
    return any(phrase in answer.lower() for phrase in generic_phrases)


def is_invalid_length(question: str, answer: str, min_len: int = 10, max_len: int = 1000) -> bool:
    """
    Check if question or answer is too short or too long.

    Args:
        question (str): The question text.
        answer (str): The answer text.
        min_len (int): Min allowed characters.
        max_len (int): Max allowed characters.
    
    Returns:
        bool: True if ether is invalid length.
    """
    return (
        len(question.strip()) < min_len
        or len(answer.strip())  < min_len
        or len(answer.strip()) > max_len
    )


def evaluate_qa_dataset(
        qa_pairs: List[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Evaluate QA dataset on heuristic rules and separate valid from invalid ones.

    Args:
        qa_pairs (List[Dist[str, str]]): List of QA dictionaries with keys: 'question', 'answer'

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: (valid entries, invalid entries)
    """
    valid_entries = []
    invalid_entries = []

    for pair in qa_pairs:
        question = pair.get('question', "").strip()
        answer = pair.get('answer', "").strip()

        if (
            is_question_about_figure(question)
            or is_answer_na(answer)
            or is_generic_answer(answer)
            or is_invalid_length(question, answer)
        ):
            invalid_entries.append(pair)
        else:
            valid_entries.append(pair)

    return valid_entries, invalid_entries


def run_heuristic_based_qa_dataset_evaluation(path: str) -> None:
    """
    Runs heuristic based QA dataset evaluation.

    Args:
        path (str): Path the QA JSON file 
    """
    data = load_json(path)

    if data is None:
        logger.error(f"🛑 Aborting validation due to load failure.")
        return 

    valid_entries, invalid_entries = evaluate_qa_dataset(data)

    logger.info(f"""
    💯total: {len(data)} ❗️invalid: {len(invalid_entries)} ✅passed: {len(valid_entries)}
    """)
    
    for pair in invalid_entries:
        logger.info(f"""
        Question: {pair['question']}
        Answer: {pair['answer']}
        """)