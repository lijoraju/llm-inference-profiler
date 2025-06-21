"""
llm_gguf.py

Author: Lijo Raju
Purpose: LLM inference interface for GGUF model using llama-cpp-python.

This module wraps the Llama class from llama-cpp-python to support inference 
from quantized .gguf models in the EduRAG pipeline.
"""

import logging
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GGUFModel:
    "Wrapper for running inference on quantized GGUF model."
    
    def __init__(self, model_path: str, max_tokens: int = 256):
        """
        Initializing GGUF model for inference.

        Args:
            model_path (str): Path to the .gguf model file.
            max_tokens (int): Max number of tokens to generate.
        """
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=1024,
                n_threads=4,
                n_gpu_layers=0,
                use_mlock=True,
                verbose=False
            )
            self.max_tokens = max_tokens
            logger.info(f"GGUF model loaded from: {model_path}")
        except Exception as e:
            logger.exception("Failed to load GGUF model: %s", e)
            raise RuntimeError(f"Failed to load GGUF model: {e}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the GGUF model.

        Args:
            prompt (str): Prompt string to generate response from.
        
        Returns:
            str: Generated response text.
        """
        try:
            output = self.model(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=0.7,
                stop=["###"],
                echo=False
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.exception(f"Inference failed: {e}")
            raise RuntimeError(f"Model inference failed: {e}")
        
