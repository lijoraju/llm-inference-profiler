"""
api.py

Author: Lijo Raju
Purpose: FastAPI route definition for EduRAG backend.
         Exposes POST /query endpoint that accepts question and returns generated answer.
"""
import logging
from fastapi import APIRouter, HTTPException

from app.models.request_model import QueryRequest
from app.models.response_model import QueryResponse
from app.rag_engine import get_rag_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Endpoint to get LLM answer using RAG.

    Args:
        request (QueryRequest): Input query and top_k value.
    
    Returns:
        QueryResponse:  Generated answer string.
    """
    try:
        engine = get_rag_engine()
        answer = engine.run(request.query)
        return QueryResponse(answer=answer)
    
    except Exception as e:
        logger.exception("RAG query failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate answer.")
