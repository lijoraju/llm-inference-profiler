"""
request_model.py

Author: Lijo Raju
Purpose: Pydantic request model for RAG query endpoint.
"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema for POST /query request."""
    query: str = Field(..., examples=["When did nationalism first appear in Europe?"])
    top_k: int = Field(3, examples=[3], description="Number of top documents to retrieve")
