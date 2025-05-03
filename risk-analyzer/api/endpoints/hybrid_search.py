from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException

from risk_analyzer.services.embedding import generate_embedding
from risk_analyzer.services.qdrant import search_vectors
from risk_analyzer.services.neo4j import query_issue_graph

router = APIRouter()

@router.get("/hybrid")
async def hybrid_search(
    query: str = Query(..., description="Search query text"),
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    limit: int = Query(10, description="Maximum number of results to return"),
) -> Dict[str, Any]:
    """
    Perform a hybrid search across vector database (Qdrant) and graph database (Neo4j).
    
    This endpoint combines semantic similarity from vector search with
    relationship information from the graph database.
    """
    try:
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
        
        return {
            "query": query,
            "project_id": project_id,
            "results": enriched_results,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing hybrid search: {str(e)}")


@router.get("/semantic")
async def semantic_search(
    query: str = Query(..., description="Search query text"),
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    limit: int = Query(10, description="Maximum number of results to return"),
) -> Dict[str, Any]:
    """
    Perform a semantic search using vector embeddings in Qdrant.
    """
    try:
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
        
        return {
            "query": query,
            "project_id": project_id,
            "results": results,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing semantic search: {str(e)}")


@router.get("/graph/{issue_id}")
async def graph_search(
    issue_id: str,
) -> Dict[str, Any]:
    """
    Get graph data for an issue and its relationships.
    """
    try:
        graph_data = query_issue_graph(issue_id)
        
        if not graph_data:
            raise HTTPException(status_code=404, detail=f"Issue with ID {issue_id} not found")
        
        return graph_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying graph: {str(e)}") 