from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from risk_analyzer.services.embedding import generate_embedding

router = APIRouter()

class EmbeddingRequest(BaseModel):
    text: str
    preprocess: bool = True

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    preprocess: bool = True

@router.post("/generate")
async def create_embedding(
    request: EmbeddingRequest = Body(...),
) -> Dict[str, Any]:
    """
    Generate an embedding vector for the provided text.
    """
    try:
        embedding = await generate_embedding(request.text, preprocess=request.preprocess)
        
        return {
            "text": request.text,
            "vector": embedding,
            "dimension": len(embedding),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")


@router.post("/batch")
async def batch_create_embeddings(
    request: BatchEmbeddingRequest = Body(...),
) -> Dict[str, Any]:
    """
    Generate embedding vectors for a batch of texts.
    """
    try:
        embeddings = await generate_embedding(request.texts, preprocess=request.preprocess)
        
        return {
            "count": len(embeddings),
            "vectors": [
                {
                    "text": text,
                    "vector": embedding,
                    "dimension": len(embedding),
                }
                for text, embedding in zip(request.texts, embeddings)
            ],
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating batch embeddings: {str(e)}") 