from typing import List, Dict, Any, Optional
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.request import Request
from django.http import HttpRequest

from apiserver.utrack.risk_analyzer.services.embedding import generate_embedding


@api_view(['POST'])
@permission_classes([IsAuthenticated])
async def create_embedding(request):
    """
    Generate an embedding vector for the provided text.
    """
    try:
        # Get request data
        text = request.data.get('text')
        preprocess = request.data.get('preprocess', True)
        
        if not text:
            return Response(
                {"error": "Text parameter is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Generate embedding
        embedding = await generate_embedding(text, preprocess=preprocess)
        
        return Response({
            "text": text,
            "vector": embedding,
            "dimension": len(embedding),
        })
    
    except Exception as e:
        return Response(
            {"error": f"Error generating embedding: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
async def batch_create_embeddings(request):
    """
    Generate embedding vectors for a batch of texts.
    """
    try:
        # Get request data
        texts = request.data.get('texts')
        preprocess = request.data.get('preprocess', True)
        
        if not texts or not isinstance(texts, list):
            return Response(
                {"error": "Texts parameter must be a non-empty list"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Generate embeddings
        embeddings = await generate_embedding(texts, preprocess=preprocess)
        
        return Response({
            "count": len(embeddings),
            "vectors": [
                {
                    "text": text,
                    "vector": embedding,
                    "dimension": len(embedding),
                }
                for text, embedding in zip(texts, embeddings)
            ],
        })
    
    except Exception as e:
        return Response(
            {"error": f"Error generating batch embeddings: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        ) 