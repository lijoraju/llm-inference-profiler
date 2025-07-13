"""
response_model.py

Author: Lijo Raju
Purpose: Pydantic response model for RAG query endpoint.
"""
from pydantic import BaseModel


class QueryResponse(BaseModel):
    """Schema for /query response."""
    answer: str

