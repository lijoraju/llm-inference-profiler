"""
improved_generate_qa.py

Author: Lijo Raju (Enhanced Version)
Purpose: Enhanced prompting system for generating diverse educational QA pairs.
        Implements Bloom's Taxonomy levels and various question types for social science.
"""
import json
import logging
from typing import List, Dict
import pickle
import re

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from pipeline.prompt_template import EducationalPromptGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct" # Updated model for better quality
BATCH_SIZE = 4  # Reduced for better quality
MAX_LENGTH = 1024


def clean_and_validate_chunk(chunk: str) -> str:
    """
    Clean and validate text chunks for better QA generation.
    
    Args:
        chunk (str): Raw text chunk
        
    Returns:
        str: Cleaned chunk or empty string if invalid
    """
    # Remove common metadata patterns
    chunk = re.sub(r'(Reprint \d{4}-\d{2}|SECTION [IVX]+|EVENTS AND PROCESSES)', '', chunk)
    chunk = re.sub(r'\n+', ' ', chunk)  # Replace multiple newlines with single space
    chunk = chunk.strip()
    
    # Filter out chunks that are too short or just headers
    if len(chunk.split()) < 20:  # At least 20 words
        return ""
    
    # Filter out chunks that don't contain substantial content
    if not re.search(r'[.!?]', chunk):  # Must contain sentence-ending punctuation
        return ""
        
    return chunk


def load_chunks(path: str) -> List[str]:
    """
    Load and clean document chunks from pickle file.
    """
    try:
        with open(path, "rb") as f:
            chunks = pickle.load(f)
        cleaned_chunks = []

        for doc in chunks:
            cleaned = clean_and_validate_chunk(doc.page_content)
            if cleaned:
                cleaned_chunks.append(cleaned)
                
        logger.info(f"Filtered {len(chunks)} chunks down to {len(cleaned_chunks)} valid chunks")
        return cleaned_chunks
    except Exception as e:
        logger.error("Failed to load chunks from %s: %s", path, e)
        raise



def create_diverse_educational_prompt(chunk: str, num_pairs: int = 2, prompt_style: str = 'diverse') -> str:
    """
    Main function to create enhanced prompts with different styles.
    
    Args:
        chunk: Text passage
        num_pairs: Number of QA pairs to generate
        prompt_style: 'diverse', 'bloom', 'topic_focused', or 'scenario'
    """
    generator = EducationalPromptGenerator()
    
    if prompt_style == 'diverse':
        return generator.create_diverse_prompt(chunk, num_pairs)
    elif prompt_style == 'bloom':
        return generator.create_bloom_taxonomy_prompt(chunk, num_pairs)
    elif prompt_style == 'topic_focused':
        return generator.create_topic_focused_prompt(chunk, num_pairs)
    elif prompt_style == 'scenario':
        return generator.create_scenario_based_prompt(chunk, num_pairs)
    else:
        return generator.create_diverse_prompt(chunk, num_pairs)


def validate_question_diversity(qa_pairs: List[Dict[str, str]]) -> Dict[str, any]:
    """
    Validate that generated questions show good diversity.

    Args:
        qa_pairs (List[Dict[str, str]]): List of generated QA pairs.
    
    Returns:
        Dict with diversity metrics and suggestions
    """
    if not qa_pairs:
        return {"score": 0, "issues": ["No questions to validate"]}
    
    questions = [pair["question"] for pair in qa_pairs]
    
    # Check question starters diversity
    starters = [q.split()[0].lower() for q in questions if q.split()]
    unique_starters = len(set(starters))
    
    # Check for repetitive patterns
    first_two_words = [" ".join(q.split()[:2]).lower() for q in questions if len(q.split()) >= 2]
    unique_patterns = len(set(first_two_words))
    
    # Check answer length diversity
    answer_lengths = [len(pair["answer"].split()) for pair in qa_pairs]
    avg_length = sum(answer_lengths) / len(answer_lengths)
    length_variance = sum((l - avg_length) ** 2 for l in answer_lengths) / len(answer_lengths)
    
    # Calculate diversity score (0-10)
    starter_score = min(10, (unique_starters / len(questions)) * 10)
    pattern_score = min(10, (unique_patterns / len(questions)) * 10)
    length_score = min(10, length_variance / 5)  # Normalize variance
    
    overall_score = (starter_score + pattern_score + length_score) / 3
    
    issues = []
    if unique_starters < len(questions) * 0.7:
        issues.append("Low diversity in question starters")
    if unique_patterns < len(questions) * 0.8:
        issues.append("Too many similar question patterns")
    if length_variance < 10:
        issues.append("Answer lengths too similar")
    
    return {
        "score": overall_score,
        "unique_starters": unique_starters,
        "unique_patterns": unique_patterns,
        "avg_answer_length": avg_length,
        "issues": issues
    }


def extract_json_from_text(text: str, chunk: str) -> List[Dict[str, str]]:
    """
    Extract JSON list from generated text using regex to find valid JSON array.
    This version is more robust to conversational preambles.

    Args:
        text (str): Generated text containing a JSON array.
        chunk (str): Context for which the text is generated.

    Returns:
        List[Dict[str, str]]: Parsed QA pairs with context or empty list if parsing fails.
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
                    "context": chunk,
                    "question": str(item["question"]).strip(),
                    "answer": str(item["answer"]).strip()
                })
        if not valid_pairs:
            logger.warning("No valid 'question' and 'answer' keys found in parsed JSON. Text: %s", text[:500])
        return valid_pairs
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON: %s. Attempting heuristic extraction. Text: %s", e, text[:500])
        # Fallback to heuristic extraction if direct JSON parsing fails
        return _heuristic_extract_qa(text, chunk)
    except Exception as e:
        logger.warning("Error extracting JSON: %s, Text: %s", e, text[:500])
        return []


def _heuristic_extract_qa(text: str, chunk: str) -> List[Dict[str, str]]:
    """
    Heuristically extracts Q&A pairs from text that didn't strictly follow JSON format
    but might have "Question: ... Answer: ..." patterns.

    Args:
        text (str): Generated text containing a JSON array.
        chunk (str): Context for which the text is generated.

    Returns:
        List[Dict[str, str]]: Parsed QA pairs with context or empty list if parsing fails.
    """
    qa_pairs = []
    # Regular expression to find "Question: ... Answer: ..." patterns
    pattern = re.compile(r'(?:\d+\.|\s*-)?\s*[Qq]uestion:\s*(.*?)\s*[Aa]nswer:\s*(.*?)(?=(?:\d+\.|\s*-)?\s*[Qq]uestion:|\Z)', re.DOTALL)
    matches = pattern.finditer(text)

    for match in matches:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        if question and answer:
            qa_pairs.append({
                "context": chunk,
                "question": question, 
                "answer": answer
            })
    if not qa_pairs:
        logger.warning("Heuristic extraction also failed to find Q&A pairs. Text: %s", text[:500])

    return qa_pairs


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
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id
        )
        return (generator, tokenizer)
    except Exception as e:
        logger.error("Failed to initialise model %s: %s", model_name, e)
        raise


def generate_qa_pairs_enhanced(chunks: List[str], generator, tokenizer, num_pairs: int = 2) -> List[Dict[str, str]]:
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
    prompt_styles = ['diverse', 'bloom', 'topic_focused', 'scenario']
    
    for i, chunk in enumerate(chunks):
        try:
            # Rotate through different prompt styles for variety
            style = prompt_styles[i % len(prompt_styles)]
            prompt = create_diverse_educational_prompt(chunk, num_pairs, style)
            
            response = generator(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7, 
                top_p=0.9,
                repetition_penalty=1.15, 
                return_full_text=False,
                truncation=True
            )
            
            if response and len(response) > 0:
                response_text = response[0]["generated_text"]
                extracted_pairs = extract_json_from_text(response_text, chunk)

                if extracted_pairs:
                    diversity_check = validate_question_diversity(extracted_pairs)
                    if diversity_check["score"] < 5:
                        logger.warning(f"Low diversity score ({diversity_check['score']:.1f}) for chunk {i}")
                        logger.warning(f"Issues: {diversity_check['issues']}")
                
                qa_pairs.extend(extracted_pairs)
                
        except Exception as e:
            logger.warning(f"Failed to process chunk {i}: {e}")
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
            pairs = generate_qa_pairs_enhanced(batch_chunks, generator, tokenizer)
            qa_dataset.extend(pairs)
            if (i // BATCH_SIZE + 1) % 10 == 0:
                logger.info(f"Generated {len(qa_dataset)} QA pairs so far...")

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(qa_dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(qa_dataset)} QA pairs to {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        raise