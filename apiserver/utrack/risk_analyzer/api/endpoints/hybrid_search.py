from typing import List, Dict, Any, Optional
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from apiserver.utrack.risk_analyzer.services.embedding import generate_embedding
from apiserver.utrack.risk_analyzer.services.qdrant import search_vectors
from apiserver.utrack.risk_analyzer.services.neo4j import query_issue_graph


@api_view(['GET'])
@permission_classes([IsAuthenticated])
async def hybrid_search(request):
    """
    Perform a hybrid search across vector database (Qdrant) and graph database (Neo4j).
    
    This endpoint combines semantic similarity from vector search with
    relationship information from the graph database.
    """
    try:
        # Get query params
        query = request.query_params.get('query')
        project_id = request.query_params.get('project_id')
        limit = int(request.query_params.get('limit', 10))
        
        if not query:
            return Response(
                {"error": "Query parameter is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Generate embedding for the query
        query_embedding = await generate_embedding(query)
        
        # Create filter if project_id is provided
        filter_conditions = {}
        if project_id:
            filter_conditions["metadata.project_id"] = project_id
        
        # Search vector database
        vector_results = await search_vectors(
            query_vector=query_embedding,
            limit=limit * 2,  # Get more results to allow for filtering
            filter_conditions=filter_conditions,
        )
        
        # Enrich results with graph data
        enriched_results = []
        for result in vector_results:
            # Extract the ID from the vector result
            item_type = result["payload"].get("type")
            item_id = result["payload"].get("id")
            
            if not item_id:
                continue
            
            # Get graph data
            graph_data = {}
            if item_type == "issue":
                graph_data = query_issue_graph(item_id)
            
            # Combine vector and graph data
            combined_result = {
                "id": item_id,
                "type": item_type,
                "score": result["score"],
                "vector_data": result["payload"],
                "graph_data": graph_data,
            }
            
            enriched_results.append(combined_result)
        
        # Sort by score and limit results
        enriched_results.sort(key=lambda x: x["score"], reverse=True)
        enriched_results = enriched_results[:limit]
        
        return Response({
            "query": query,
            "project_id": project_id,
            "results": enriched_results,
        })
    
    except Exception as e:
        return Response(
            {"error": f"Error performing hybrid search: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
async def semantic_search(request):
    """
    Perform a semantic search using vector embeddings in Qdrant.
    """
    try:
        # Get query params
        query = request.query_params.get('query')
        project_id = request.query_params.get('project_id')
        limit = int(request.query_params.get('limit', 10))
        
        if not query:
            return Response(
                {"error": "Query parameter is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Generate embedding for the query
        query_embedding = await generate_embedding(query)
        
        # Create filter if project_id is provided
        filter_conditions = {}
        if project_id:
            filter_conditions["metadata.project_id"] = project_id
        
        # Search vector database
        results = await search_vectors(
            query_vector=query_embedding,
            limit=limit,
            filter_conditions=filter_conditions,
        )
        
        return Response({
            "query": query,
            "project_id": project_id,
            "results": results,
        })
    
    except Exception as e:
        return Response(
            {"error": f"Error performing semantic search: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def graph_search(request, issue_id):
    """
    Get graph data for an issue and its relationships.
    """
    try:
        graph_data = query_issue_graph(issue_id)
        
        if not graph_data:
            return Response(
                {"error": f"Issue with ID {issue_id} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        return Response(graph_data)
    
    except Exception as e:
        return Response(
            {"error": f"Error querying graph: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        ) 