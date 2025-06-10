"""
run_train.py

Author: Lijo Raju
Purpose: CLI script to start QLoRA fine-tuning on NCERT QA dataset.

Usage:
    python scripts/run_train.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.train import run_training

if __name__ == "__main__":
    run_training()