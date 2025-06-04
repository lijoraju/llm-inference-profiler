import pytest
from pathlib import Path
import json
from transformers import AutoTokenizer

@pytest.fixture
def sample_pdf_path():
    return Path("data/sample_docs")


@pytest.fixture
def chunks_pkl_path():
    return Path("data/vectorstore/chunks.pkl")


@pytest.fixture
def faiss_index_path():
    return Path("data/vectorstore/faiss_index")

@pytest.fixture
def top_k():
    return 3

@pytest.fixture
def mock_qa_path(tmp_path):
    qa_data = [
        {"question": "What is an acid?", "answer": "An acid is a substance with pH < 7."},
        {"question": "What is photosynthesis?", "answer": "Photosynthesis is how plants make food using sunlight."}
    ]
    path = tmp_path/"mock_train.json"
    with open(path, "w") as f:
        json.dump(qa_data, f)
    return path

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")