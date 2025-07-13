import json 
from typing import Optional, List, Dict, Any 
import logging

logging.basicConfig(level=logging.INFO)


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
        logging.error(f"❌ Failed to JSON from {path}: {e}")
        return None