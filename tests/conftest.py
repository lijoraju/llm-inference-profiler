import pytest
from pathlib import Path

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