"""
train.py

Author: Lijo Raju
Purpose: Train TinyLlama with QLoRA on NCERT QA Dataset.

This script loads a 4-bit quantized model, applies LoRA adapters using PEFT,
loads tokenized dataset from disk, and trains using HuggingFace Trainer.
"""
import os
import logging
from typing import Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH_TRAIN = "data/finetune/processed/train"
DATASET_PATH_VALIDATION = "data/finetune/processed/validation"
OUTPUT_DIR = "outputs/qlora_model"


def get_bnb_config() -> BitsAndBytesConfig:
    """
    Creates bitsandbytes config for QLoRA training.

    Returns:
        BitsAndBytesConfig: Configuration object for bnb quantization.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def get_lora_config() -> LoraConfig:
    """
    Return LoRA adapter configuration.

    Returns:
        LoraConfig: Configuration for LoRA.
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )


def load_model_tokenizer(model_name: str, bnb_config: BitsAndBytesConfig):
    """
    Load model with 4-bit quantization and apply PEFT adapter.

    Args:
        model_name (str): Name of the base model.
        bnb_config (BitsAndBytesConfig): Quantization config.
    
    Returns:
        model, tokenizer: PEFT-wrapped model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    return model, tokenizer


def run_training():
    """Main function to execute QLoRA fine-tuning."""
    try:
        logger.info("Loading train dataset from %s ...", DATASET_PATH_TRAIN)
        logger.info("Loading validation dataset from %s ...", DATASET_PATH_VALIDATION)
        train_dataset = load_from_disk(DATASET_PATH_TRAIN)
        eval_dataset = load_from_disk(DATASET_PATH_VALIDATION)

        logger.info("Loading model and tokenizer...")
        bnb_config = get_bnb_config()
        model, tokenizer = load_model_tokenizer(MODEL_NAME, bnb_config)

        logger.info("Setting up trainer...")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=20,
            save_strategy="epoch",
            eval_strategy="epoch",
            do_train=True,
            do_eval=True,
            fp16=True,
            report_to="none",
            save_total_limit=1
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info("Saving model to %s", OUTPUT_DIR)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        logger.info(f"✅ Training complete.")
    
    except Exception as e:
        logger.error("Training failed: %s", e)