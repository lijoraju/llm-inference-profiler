"""
prepare_dataset.py

Author: Lijo Raju
Purpose: Prepare dataset for QLoRA fine-tuning.
"""
import os
import json
import logging
from typing import List, Dict

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_qa_pairs(path: str) -> List[Dict[str, str]]:
    """
    Load QA pairs from a JSON file.

    Args:
        path(str): Path to the JSON file containing QA pairs.
    
    Returns: 
        List[Dict[str, str]]: A list of QA records with question and answer.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load QA pairs from {path}: {e}")
        raise


def format_for_instruction_tuning(qa_pairs: List[Dict[str, str]]) -> Dataset:
    """
    Format QA pairs for instruction tuning.

    Args:
        qa_pairs (List[Dict[str, str]]): List of question-answer dictionaries.

    Returns:
        Dataset: HuggingFace Dataset in instruction format.
    """
    formatted = []
    for qa in qa_pairs:
        formatted.append({
            "instruction": "Answer the following question based on NCERT content.",
            "input": qa["question"],
            "output": qa["answer"]
        })
    try:
        dataset = Dataset.from_list(formatted)
        return dataset
    except Exception as e:
        logger.error(f"Failed to creating HuggingFace Dataset: {e}")
        return None 


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """
    Tokenize the instruction-tuning dataset.

    Args:
        dataset (Dataset): HuggingFace Dataset with instruction/input/output.
        tokenizer: HuggingFace tokenizer.
        max_length (int): Max token length for truncation.

    Returns:
        Dataset: Tokenized dataset.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def tokenize_fn(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        full_input = prompt + " " + example['output']
        try:
            tokenized = tokenizer(
                full_input,
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        except Exception as e:
            logger.error(f"Error tokenizing example: {example.get('input', 'N/A')[:50]}... Error: {e}")
            return None

    
    tokenized_dataset = dataset.map(tokenize_fn, batched=False).filter(lambda x: x is not None)
    if len(tokenized_dataset) == 0:
        logger.warning("Tokenization resulted in an empty dataset.")
    logger.info(f"Successfully tokenized dataset. Original size: {len(dataset)}, Tokenized size: {len(tokenized_dataset)}")
    return tokenized_dataset


def split_and_tokenize_dataset(
        formatted_dataset: Dataset, 
        tokenizer,
        train_size: float = 0.9
    ) -> DatasetDict: 
    """
    Split dataset into train/validation and tokenize.

    Args:
        formatted_dataset (Dataset): Dataset to split and tokenize.
        tokenizer: HuggingFace tokenizer.
        train_size (float): Fraction of data to use for training.
    
    Returns:
        DatasetDict: Tokenized train and validation datasets.
    """
    split = formatted_dataset.train_test_split(train_size=train_size)
    tokenized_train = tokenize_dataset(split["train"], tokenizer)
    tokenized_val = tokenize_dataset(split["test"], tokenizer)
    return DatasetDict(train=tokenized_train, validation=tokenized_val)


def prepare_dataset(
    qa_path: str,
    tokenizer_name: str,
    output_path: str,
    max_length: int = 512
):
    """
    Main function to prepare dataset for QLoRA training.

    Args:
        qa_path (str): Path to the raw QA JSON file.
        tokenizer_name (str): Name or path of the tokeinzer.
        output_path (str): Directory to save the processed dataset.
        max_length (int): Max token length for model input.
    """
    logger.info(f"Loading QA pairs from {qa_path}")
    raw_pairs = load_qa_pairs(path=qa_path)

    logger.info(f"Formatting for instruction tuning...")
    formatted_dataset = format_for_instruction_tuning(qa_pairs=raw_pairs)

    logger.info(f"Loding tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info(f"Tokenizing and splitting dataset...")
    tokenized_dataset = split_and_tokenize_dataset(formatted_dataset, tokenizer)

    logger.info(f"Saving to disk at {output_path}")
    tokenized_dataset.save_to_disk(output_path)

    logger.info(f"✅ Dataset preparation complete.")