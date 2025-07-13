"""
api.py

Author: Lijo Raju
Purpose: FastAPI route definition for EduRAG backend.
         Exposes POST /query endpoint that accepts question and returns generated answer.
"""
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.request_model import QueryRequest
from app.models.response_model import QueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", tags=["Health"])
def root() -> JSONResponse:
    """
    Root endpoint to indicate API is running.

    Returns:
        JSONResponse: A simple message indicating service is live.
    """
    return JSONResponse(
        content={"status": "ok", "message": "EduRAG API is live."}
    )

@router.get("/health", tags=["Health"])
def health_check() -> JSONResponse:
    """
    Health check endpoint for uptime monitoring.

    Returns:
        JSONResponse: A status message indicating health.
    """
    return JSONResponse (
        content={"status": "healthy", "service": "EduRAG"}
    )

@router.post("/query", response_model=QueryResponse)
async def query_rag(request_body: QueryRequest, request: Request) -> QueryResponse:
    """
    Endpoint to get LLM answer using RAG.

    Args:
        request_body (QueryRequest): Input query and top_k value.
        reuest (Request): FastAPI request object with app state.
    
    Returns:
        QueryResponse:  Generated answer string.
    """
    try:
        engine = request.app.state.rag_engine
        answer = engine.run(request_body.query)
        return QueryResponse(answer=answer)
    
    except Exception as e:
        logger.exception("RAG query failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate answer.")
