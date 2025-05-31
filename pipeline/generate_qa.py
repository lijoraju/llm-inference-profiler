import json
import logging
from typing import List, Dict
import pickle

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "HuggingFaceH4/zephyr-7b-beta"
BATCH_SIZE = 8
MAX_LENGTH = 1024


def load_chunks(path: str) -> List[str]:
    """
    Load document chunks from pickle file.

    Args:
        path (str): Path to the chunks file.

    Returns:
        List[str]: A list of document chunks as strings.
    """
    try:
        with open(path, "rb") as f:
            chunks = pickle.load(f)
        return [doc.page_content for doc in chunks]
    except Exception as e:
        logger.error("Failed to load chunks from %s: %s", path, e)
        raise


def init_qa_generator(model_name: str = DEFAULT_MODEL):
    """
    Initialize a text generation pipeline for QA pair generation.

    Args:
        model_name (str): Hugging Face model name or path.

    Returns:
        tuple: (pipeline, tokenizer) for text generation.
    """
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
            max_length=MAX_LENGTH,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        return (generator, tokenizer)
    except Exception as e:
        logger.error("Failed to initialise model %s: %s", model_name, e)
        raise


def extract_json_from_text(text: str) -> List[Dict[str, str]]:
    """
    Extract JSON list from generated text using regex to find valid JSON array.
    This version is more robust to conversational preambles.

    Args:
        text (str): Generated text containing a JSON array.

    Returns:
        List[Dict[str, str]]: Parsed QA pairs or empty list if parsing fails.
    """
    try:
        # Step 1: Find the most likely JSON string within the text
        # Look for a string that starts with [ { and ends with } ]
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if not match:
            # If a list isn't found, try to find a single JSON object and wrap it
            single_obj_match = re.search(r'\{[^}]*"question"[^}]*"answer"[^}]*\}', text, re.DOTALL | re.IGNORECASE)
            if single_obj_match:
                qa_json_str = f"[{single_obj_match.group(0)}]"
            else:
                logger.warning("No potential JSON structure found in text: %s", text[:500])
                return []
        else:
            qa_json_str = match.group(0)

        # Clean up the JSON string: remove control characters and common escape sequence issues
        qa_json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', qa_json_str) # Remove control characters
        qa_json_str = qa_json_str.replace('\\', '') # Remove backslashes that might not be valid JSON escapes
        parsed_data = json.loads(qa_json_str)
        if not isinstance(parsed_data, list):
            parsed_data = [parsed_data]
        valid_pairs = []

        for item in parsed_data:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                valid_pairs.append({
                    "question": str(item["question"]).strip(),
                    "answer": str(item["answer"]).strip()
                })
        if not valid_pairs:
            logger.warning("No valid 'question' and 'answer' keys found in parsed JSON. Text: %s", text[:500])
        return valid_pairs
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON: %s. Attempting heuristic extraction. Text: %s", e, text[:500])
        # Fallback to heuristic extraction if direct JSON parsing fails
        return _heuristic_extract_qa(text)
    except Exception as e:
        logger.warning("Error extracting JSON: %s, Text: %s", e, text[:500])
        return []


def _heuristic_extract_qa(text: str) -> List[Dict[str, str]]:
    """
    Heuristically extracts Q&A pairs from text that didn't strictly follow JSON format
    but might have "Question: ... Answer: ..." patterns.

    Args:
        text (str): Generated text containing a JSON array.

    Returns:
        List[Dict[str, str]]: Parsed QA pairs or empty list if parsing fails.
    """
    qa_pairs = []
    # Regular expression to find "Question: ... Answer: ..." patterns
    pattern = re.compile(r'(?:\d+\.|\s*-)?\s*[Qq]uestion:\s*(.*?)\s*[Aa]nswer:\s*(.*?)(?=(?:\d+\.|\s*-)?\s*[Qq]uestion:|\Z)', re.DOTALL)
    matches = pattern.finditer(text)

    for match in matches:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        if question and answer:
            qa_pairs.append({"question": question, "answer": answer})
    if not qa_pairs:
        logger.warning("Heuristic extraction also failed to find Q&A pairs. Text: %s", text[:500])

    return qa_pairs


def generate_qa_pairs(chunks: List[str], generator, tokenizer, num_pairs: int = 2) -> List[Dict[str, str]]:
    """
    Generate Question-Answer pairs for a batch of text chunks.

    Args:
        chunks (List[str]): The List of input text chunks.
        generator (pipeline): Text generation pipeline.
        tokenizer: Model tokenizer for batch processing.
        num_pairs (int): Number of QA pairs to generate per chunk.

    Returns:
        List[Dict[str, str]]: List of question-answer pairs.
    """
    qa_pairs = []

    for chunk in chunks:
        try:
            prompt = f"""<|system|>
                        You are a helpful assistant that generates question-answer pairs from text. Your output must ONLY be a valid JSON array.
                        <|user|>
                        Given the following text, generate exactly {num_pairs} question-answer pairs.

                        Text: {chunk[:800]}

                        Please return the output as a valid JSON array of objects, where each object has a 'question' key and an 'answer' key.
                        Example: [{{"question": "What is the capital of France?", "answer": "Paris"}}, {{"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"}}]
                        <|assistant|>
                        """
            response = generator(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                return_full_text=False,  
                truncation=True
            )
            if response and len(response) > 0:
                response_text = response[0]["generated_text"]
                extracted_pairs = extract_json_from_text(response_text)
                qa_pairs.extend(extracted_pairs)
        except Exception as e:
            logger.warning(f"Failed to process chunk due to generation or extraction error: {e}")
            continue

    return qa_pairs


def generate_dataset(chunks_path: str, output_path: str, model_name: str = DEFAULT_MODEL):
    """
    Generate a QA dataset from document chunks with batch processing.

    Args:
        chunks_path (str): Path to the input chunk file.
        output_path (str): Path to the output file.
        model_name (str): Hugging Face model name or path.
    """
    try:
        chunks = load_chunks(chunks_path)
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
        generator, tokenizer = init_qa_generator(model_name)
        qa_dataset = []

        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Generating QA Pairs"):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            pairs = generate_qa_pairs(batch_chunks, generator, tokenizer)
            qa_dataset.extend(pairs)
            if (i // BATCH_SIZE + 1) % 10 == 0:
                logger.info(f"Generated {len(qa_dataset)} QA pairs so far...")

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(qa_dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(qa_dataset)} QA pairs to {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        raise