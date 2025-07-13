from pipeline import ingest


def test_load_pdf(sample_pdf_path):
    docs = ingest.load_pdfs(sample_pdf_path)
    assert len(docs) > 0
    assert hasattr(docs[0], "page_content")


def test_chunk_documents(sample_pdf_path):
    docs = ingest.load_pdfs(sample_pdf_path)
    chunks = ingest.chunk_documents(documents=docs)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert hasattr(chunks[0], "page_content")


def test_save_chunks(chunks_pkl_path, sample_pdf_path):
    docs = ingest.load_pdfs(sample_pdf_path)
    chunks = ingest.chunk_documents(docs)
    ingest.save_chunks(chunks, chunks_pkl_path)
    assert chunks_pkl_path.exists()