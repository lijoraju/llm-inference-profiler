"""
merge_lora.py

Author: Lijo Raju
Purpose: Merge LoRA adapter into base model and save the merged model.

This module is used to fuse a trained LoRA adapter with its base model.
Used to prepare a fully merged model for CPU inference or GGUF conversion. 
"""
import os
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_adapter(
        base_model_name: str,
        adapter_path: str,
        output_path: str,
        trust_remote_code: bool = True
) -> None:
    """
    Merge LoRA adapter into base model and save merged model.

    Args:
        base_model_name (str): HuggingFace model ID or path of base model.
        adapter_path (str): Path to the trained LoRA adapter directory.
        output_path (str): Directory to save the merged full model.
        trust_remote_code (bool): Allow loading custom model classes.
    
    Raises:
        RuntimeError: If model merging or saving fails.
    """
    try:
        logger.info("Loading base model: %s", base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=trust_remote_code
        )

        logger.info("Loading adapter from: %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

        logger.info("Meging LoRA adapter into base model...")
        model = model.merge_and_unload()

        logger.info("Saving merged model to: %s", output_path)
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(output_path)

        logger.info(f"✅ Merged model successfully saved.")
    except Exception as e:
        logger.error("❌ Failed to merge LoRA adapter: %s", e)
        raise RuntimeError(f"Model merge failed: {e}")
