import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.ingest import process_document


if __name__ == "__main__":
    sample_pdf = "data/sample_docs/sample.pdf"

    if os.path.exists(sample_pdf):
        chunks = process_document(sample_pdf)
        print(f"First chunk:\n{chunks[0][:300]}")
    else:
        print(f"⚠️ Please add a sample PDF to data/sample_docs/sample.pdf")