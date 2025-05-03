import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from django.conf import settings

logger = logging.getLogger(__name__)

# Global Qdrant client instance
_client: Optional[QdrantClient] = None

async def init_qdrant() -> QdrantClient:
    """Initialize Qdrant client and create collection if it doesn't exist."""
    global _client
    
    # Create client
    _client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )
    
    # Check if collection exists
    collections = _client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    # Create collection if it doesn't exist
    if settings.QDRANT_COLLECTION not in collection_names:
        logger.info(f"Creating Qdrant collection: {settings.QDRANT_COLLECTION}")
        _client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        
        # Create index for efficient filtering
        _client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name="metadata.project_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        _client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name="metadata.type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        _client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name="metadata.created_at",
            field_schema=models.PayloadSchemaType.DATETIME,
        )
    
    logger.info(f"Qdrant initialized with collection: {settings.QDRANT_COLLECTION}")
    return _client


def get_qdrant_client() -> QdrantClient:
    """Get the Qdrant client instance."""
    if _client is None:
        raise RuntimeError("Qdrant client not initialized. Call init_qdrant() first.")
    return _client


async def store_embeddings(
    vectors: List[List[float]],
    ids: List[str],
    metadata: List[Dict[str, Any]],
    collection_name: Optional[str] = None,
) -> bool:
    """Store embeddings in Qdrant."""
    client = get_qdrant_client()
    collection = collection_name or settings.QDRANT_COLLECTION
    
    # Create points
    points = [
        PointStruct(
            id=id,
            vector=vector,
            payload=metadata_item,
        )
        for id, vector, metadata_item in zip(ids, vectors, metadata)
    ]
    
    # Upsert points
    try:
        client.upsert(
            collection_name=collection,
            points=points,
        )
        return True
    except Exception as e:
        logger.error(f"Error storing embeddings in Qdrant: {e}")
        return False


async def search_vectors(
    query_vector: List[float],
    limit: int = 10,
    filter_conditions: Optional[Dict[str, Any]] = None,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search for vectors in Qdrant."""
    client = get_qdrant_client()
    collection = collection_name or settings.QDRANT_COLLECTION
    
    # Create filter
    filter_query = None
    if filter_conditions:
        filter_query = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
                for key, value in filter_conditions.items()
            ]
        )
    
    # Search
    try:
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_query,
            with_payload=True,
        )
        
        # Format results
        return [
            {
                "id": str(result.id),
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]
    except Exception as e:
        logger.error(f"Error searching vectors in Qdrant: {e}")
        return [] 