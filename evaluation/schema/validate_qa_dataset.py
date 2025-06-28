"""
validate_qa_dataset.py

Author: Lijo Raju
Purpose: Schema validation for QA datasets used for this project.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Loads a JSON file and returns its content.

    Args:
        path (str): Path to the JSON file.
    
    Returns:
        Optional[List[Dict[str, Any]]]: Parsed JSON data as list of dictionaries
        or None if loading fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to JSON from {path}: {e}")
        return None


def validate_qa_schema(
        data: List[Dict[str,Any]],
        required_fields: List[str] = ["context", "question", "answer"],
) -> List[int]:
    """
    Validate schema of QA dataset entries.

    Args:
        data (List[Dist[str, Any]]): List of QA examples.
        required_fields (List[str]): Fields that must be present. Defaults to ["context", "question", "answer"].
    
    Returns:
        List[int]: List of indices where validation failed.
    """
    failed_indices = []

    for idx, example in enumerate(data):
        if not all(field in example for field in required_fields):
            logger.warning(f"⚠️ Entry {idx} is missing required fields: {example}")
            failed_indices.append(idx)
            continue

        if not all(isinstance(example[field], str) for field in required_fields):
            logger.warning(f"⚠️ Entry {idx} has non string field types: {example}")
            failed_indices.append(idx)
    
    return failed_indices


def run_schema_validation(path: str) -> None:
    """
    Run validation on the given QA dataset file.

    Args:
        path (str): Path to the dataset file.
    """
    logger.info(f"🧪 Validating schema for file {path}")
    data = load_json(path)

    if data is None:
        logger.error(f"🛑 Aborting validation due to load failure.")
        return 
    
    failures = validate_qa_schema(data)
    if not failures:
        logger.info(f"✅ All entries passed schema validation.")
    else:
        logger.error(f"❌ {len(failures)} entries failed validation.")
