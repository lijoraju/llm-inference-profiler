from pipeline import embed
from langchain.docstore.document import Document


def test_load_chunks(chunks_pkl_path):
    chunks = embed.load_chunks(chunks_pkl_path)
    assert isinstance(chunks, list)
    assert isinstance(chunks[0], Document)


def test_embed_chunks(chunks_pkl_path):
    chunks = embed.load_chunks(chunks_pkl_path)
    index = embed.embed_chunks(chunks)
    assert index is not None


def test_save_faiss_index(chunks_pkl_path, faiss_index_path):
    chunks = embed.load_chunks(chunks_pkl_path)
    index = embed.embed_chunks(chunks)
    embed.save_faiss_index(index, faiss_index_path)
    assert faiss_index_path.exists()
