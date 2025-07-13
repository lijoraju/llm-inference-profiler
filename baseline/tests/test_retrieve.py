from pipeline import retrieve


def test_load_retriever(top_k):
    retriever = retrieve.load_retriever(top_k=top_k)
    assert retriever is not None


def test_retrieve_top_k(top_k):
    query = "What changes did Napoleon introduce to \
            make the administrative system more efficient \
            in the territories ruled by him?"
    docs = retrieve.retrieve_top_k(query, top_k)
    assert isinstance(docs, list)
    assert len(docs) == top_k
    assert hasattr(docs[0], "page_content")
