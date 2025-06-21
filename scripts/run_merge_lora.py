"""
run_merge_lora.py

Author: Lijo Raju
Purpose: CLI merge LoRA adapter into base model.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
from pipeline.merge_lora import merge_lora_adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default="outputs/qlora_model", 
                        help="Path to trained LoRA adapter")
    parser.add_argument("--output_path", type=str, default="merged_model",
                        help="Where to save the merged model")
    
    args = parser.parse_args()

    try:
        merge_lora_adapter(
            base_model_name=args.base_model,
            adapter_path=args.adapter_path,
            output_path=args.output_path
        )
    except Exception as e:
        logger.error("Merge failed: %s", e)


if __name__ == "__main__":
    main()
