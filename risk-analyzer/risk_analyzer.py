#!/usr/bin/env python3
"""
Risk Analyzer Main Application

This module serves as the main entry point for the Risk Analyzer service.
It initializes connections to databases, sets up the event consumer,
and exposes API endpoints for querying the RAG system.
"""

import os
import sys
import logging
import threading
import asyncio
import json
import time
import traceback
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# FastAPI imports
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Project imports - adjust these if needed
from data_extraction import extract_data_for_processing
from embedding import EmbeddingPipeline, QdrantManager
from graph_storage import GraphPipeline, Neo4jManager
from event_consumer import EventConsumer
from event_producer import check_rabbitmq_health, setup_rabbitmq
from query_engine import QueryEngine

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
embedding_pipeline = None
graph_pipeline = None
event_consumer = None
neo4j_manager = None
qdrant_manager = None
query_engine = None

# API Models
class SearchQuery(BaseModel):
    """Model for a search query."""
    query: str
    project_id: Optional[str] = None
    limit: int = 10
    use_graph: bool = True
    use_vector: bool = True

class RiskQueryResult(BaseModel):
    """Model for risk analysis results."""
    score: float
    factors: List[Dict[str, Any]]
    recommendations: List[str]

class DocumentSearchQuery(BaseModel):
    """Model for document search query."""
    query: str
    project_id: Optional[str] = None 
    document_types: Optional[List[str]] = None
    limit: int = 10

# FastAPI app
app = FastAPI(
    title="Risk Analyzer API",
    description="API for analyzing risks in projects using RAG techniques",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialization functions
def init_qdrant():
    """Initialize connection to Qdrant vector database."""
    global qdrant_manager, embedding_pipeline
    try:
        # Create Qdrant manager from embedding module
        from embedding import QdrantManager
        qdrant_manager = QdrantManager(
            host=os.getenv("QDRANT_HOST", "utrack-qdrant"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "utrack_vectors"),
        )
        
        # Initialize embedding pipeline
        embedding_pipeline = EmbeddingPipeline(qdrant_manager=qdrant_manager)
        logger.info("Qdrant connection initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant connection: {e}")
        logger.error(traceback.format_exc())
        return False

def init_neo4j():
    """Initialize connection to Neo4j graph database."""
    global neo4j_manager, graph_pipeline
    try:
        # Create Neo4j manager from graph_storage module
        neo4j_manager = Neo4jManager(
            uri=os.getenv("NEO4J_URI", "bolt://utrack-neo4j:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "utrackneo4j"),
        )
        
        # Test connection
        neo4j_manager.connect()
        
        # Initialize graph pipeline
        graph_pipeline = GraphPipeline(neo4j_manager=neo4j_manager)
        
        # Set up constraints and indexes
        graph_pipeline.neo4j_manager.setup_constraints()
        
        logger.info("Neo4j connection initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j connection: {e}")
        logger.error(traceback.format_exc())
        return False

def init_query_engine():
    """Initialize the query engine."""
    global query_engine, qdrant_manager, neo4j_manager
    try:
        # Initialize query engine with existing managers
        from embedding import TextPreprocessor, EmbeddingGenerator
        
        # Set up embedding generator with model name from env
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        embedding_generator = EmbeddingGenerator(model_name=model_name)
        
        query_engine = QueryEngine(
            qdrant_manager=qdrant_manager,
            neo4j_manager=neo4j_manager,
            embedding_generator=embedding_generator,
            text_preprocessor=TextPreprocessor()
        )
        
        logger.info("Query engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize query engine: {e}")
        logger.error(traceback.format_exc())
        return False

def start_event_consumer():
    """Start the RabbitMQ event consumer in a separate thread."""
    global event_consumer
    
    def run_consumer():
        try:
            event_consumer = EventConsumer()
            event_consumer.run()
        except Exception as e:
            logger.error(f"Event consumer failed: {e}")
            logger.error(traceback.format_exc())
    
    # Start in a separate thread
    consumer_thread = threading.Thread(target=run_consumer, daemon=True)
    consumer_thread.start()
    logger.info("Event consumer started in background thread")
    return consumer_thread

async def process_initial_data(project_id: Optional[str] = None, days: int = 30):
    """
    Process initial data for a project.
    
    Args:
        project_id: Optional project ID to process
        days: Number of days of data to process
    
    Returns:
        dict: Processing results
    """
    try:
        logger.info(f"Starting initial data processing for project_id={project_id}, days={days}")
        
        # Extract data from sources
        data = extract_data_for_processing(project_id=project_id, days=days)
        
        # Process data with graph pipeline
        graph_results = graph_pipeline.process_all_data(data, show_progress=True)
        
        # Process text data with embedding pipeline
        embedding_results = {
            "issues": 0,
            "comments": 0,
            "documents": 0,
        }
        
        # Process issues
        for issue in data.get("issues", []):
            text = f"{issue.get('title', '')} {issue.get('description', '')}"
            if text.strip():
                vector_id = f"issue_{issue['id']}"
                embedding_pipeline.process_text(
                    text=text,
                    vector_id=vector_id,
                    metadata={"id": issue["id"], "type": "issue"}
                )
                embedding_results["issues"] += 1
        
        # Process comments
        for comment in data.get("comments", []):
            text = comment.get("content", "")
            if text.strip():
                vector_id = f"comment_{comment['id']}"
                embedding_pipeline.process_text(
                    text=text,
                    vector_id=vector_id,
                    metadata={"id": comment["id"], "type": "comment"}
                )
                embedding_results["comments"] += 1
        
        # Process documents
        for doc in data.get("documents", []):
            # Extract text from document if needed
            # This would typically be handled by extract_document_text in embedding_pipeline
            vector_id = f"document_{doc['id']}"
            doc_text = embedding_pipeline.extract_document_text(
                document_name=doc["name"],
                content_type=doc.get("content_type", "")
            )
            if doc_text:
                embedding_pipeline.process_text(
                    text=doc_text,
                    vector_id=vector_id,
                    metadata={"id": doc["id"], "type": "document"}
                )
                embedding_results["documents"] += 1
        
        # Link vectors with graph nodes
        vector_mappings = {
            "Issue": {issue_id: f"issue_{issue_id}" for issue_id in graph_results["issues"]},
            "Comment": {comment_id: f"comment_{comment_id}" for comment_id in graph_results["comments"]},
            "Document": {doc_id: f"document_{doc_id}" for doc_id in graph_results["documents"]},
        }
        
        link_results = graph_pipeline.link_vectors_to_nodes(vector_mappings)
        
        results = {
            "graph_results": graph_results,
            "embedding_results": embedding_results,
            "link_results": link_results,
        }
        
        logger.info(f"Initial data processing completed successfully: {json.dumps(results)}")
        return results
    
    except Exception as e:
        logger.error(f"Error processing initial data: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# API endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Risk Analyzer API",
        "version": "1.0.0",
        "docs_url": "/docs",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_data = {
        "status": "healthy",
        "components": {
            "qdrant": "unknown",
            "neo4j": "unknown",
            "rabbitmq": "unknown",
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    # Check Qdrant health
    if qdrant_manager:
        try:
            qdrant_manager.get_client().get_collections()
            health_data["components"]["qdrant"] = "healthy"
        except Exception as e:
            health_data["components"]["qdrant"] = "unhealthy"
            health_data["status"] = "degraded"
    
    # Check Neo4j health
    if neo4j_manager:
        try:
            neo4j_manager.connect()
            health_data["components"]["neo4j"] = "healthy"
        except Exception as e:
            health_data["components"]["neo4j"] = "unhealthy"
            health_data["status"] = "degraded"
    
    # Check RabbitMQ health
    rabbitmq_healthy, _ = check_rabbitmq_health()
    health_data["components"]["rabbitmq"] = "healthy" if rabbitmq_healthy else "unhealthy"
    
    # Overall status
    if "unhealthy" in health_data["components"].values():
        health_data["status"] = "degraded"
    
    response_status = status.HTTP_200_OK if health_data["status"] != "unhealthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=health_data, status_code=response_status)

@app.post("/api/v1/search/hybrid")
async def hybrid_search(query: SearchQuery):
    """
    Hybrid search endpoint combining vector and graph databases.
    
    Args:
        query: Search query parameters
        
    Returns:
        List of search results
    """
    try:
        if not query_engine:
            raise HTTPException(
                status_code=503, 
                detail="Query engine not initialized"
            )
        
        # Use the query engine for hybrid search
        results = query_engine.hybrid_search(
            query_text=query.query,
            project_id=query.project_id,
            limit=query.limit,
            use_vector=query.use_vector,
            use_graph=query.use_graph
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/api/v1/search/documents")
async def document_search(query: DocumentSearchQuery):
    """
    Document search endpoint.
    
    Args:
        query: Document search parameters
        
    Returns:
        Document search results
    """
    try:
        if not query_engine:
            raise HTTPException(
                status_code=503, 
                detail="Query engine not initialized"
            )
        
        # Use the query engine for document search
        results = query_engine.document_search(
            query_text=query.query,
            document_types=query.document_types,
            project_id=query.project_id,
            limit=query.limit
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Document search error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Document search error: {str(e)}")

@app.get("/api/v1/search/issue/{issue_id}")
async def issue_detail(
    issue_id: str,
    query: Optional[str] = None,
    limit: int = 10
):
    """
    Issue detail search endpoint.
    
    Args:
        issue_id: ID of the issue to get details for
        query: Optional query to find similar issues
        limit: Maximum number of similar issues to return
        
    Returns:
        Issue details and related information
    """
    try:
        if not query_engine:
            raise HTTPException(
                status_code=503, 
                detail="Query engine not initialized"
            )
        
        # Use the query engine for issue detail search
        results = query_engine.issue_detail_search(
            issue_id=issue_id,
            query_text=query,
            limit=limit
        )
        
        if "error" in results:
            raise HTTPException(
                status_code=404,
                detail=results["error"]
            )
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Issue detail search error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Issue detail search error: {str(e)}")

@app.post("/api/v1/process/initial-data")
async def run_initial_processing(
    background_tasks: BackgroundTasks,
    project_id: Optional[str] = None,
    days: int = 30
):
    """
    Trigger initial data processing.
    
    Args:
        project_id: Optional project ID to process
        days: Number of days of data to process
    
    Returns:
        Processing status
    """
    # Start processing in background
    background_tasks.add_task(process_initial_data, project_id, days)
    
    return {
        "status": "processing_started",
        "message": f"Initial data processing started for project_id={project_id}, days={days}",
    }

@app.get("/api/v1/risk/project/{project_id}")
async def analyze_project_risk(project_id: str, days: int = 30):
    """
    Analyze project risk using RAG system.
    
    Args:
        project_id: Project ID to analyze
        days: Number of days of historical data to analyze
        
    Returns:
        Risk analysis results
    """
    try:
        if not query_engine:
            raise HTTPException(
                status_code=503, 
                detail="Query engine not initialized"
            )
        
        # Use the specialized risk analysis search
        risk_query = "project risks delays blockers issues"
        results = query_engine.risk_analysis_search(
            query_text=risk_query,
            project_id=project_id,
            days=days,
            limit=20
        )
        
        # Extract risk factors
        risk_factors = results.get("risk_factors", [])
        
        # Generate recommendations based on risk factors
        recommendations = []
        for factor in risk_factors:
            if factor["name"] == "Delayed Issues":
                recommendations.append("Review delayed issues and prioritize critical ones")
            elif factor["name"] == "Blocked Issues":
                recommendations.append("Address blockers and dependencies affecting progress")
            elif factor["name"] == "Resource Constraints":
                recommendations.append("Redistribute workload to balance team assignments")
            elif factor["name"] == "Technical Challenges":
                recommendations.append("Schedule technical review sessions for complex problems")
        
        # Calculate overall risk score (average of factor scores)
        risk_score = 0.0
        if risk_factors:
            risk_score = sum(factor["score"] for factor in risk_factors) / len(risk_factors)
        
        # Format response
        analysis_result = {
            "project_id": project_id,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "results": results.get("results", []),
            "timestamp": datetime.now().isoformat(),
        }
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"Risk analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Risk analysis error: {str(e)}")

def init_services():
    """Initialize all services and connections."""
    # Initialize databases
    qdrant_ok = init_qdrant()
    neo4j_ok = init_neo4j()
    
    # Initialize query engine
    query_engine_ok = False
    if qdrant_ok and neo4j_ok:
        query_engine_ok = init_query_engine()
    
    # Setup RabbitMQ if needed
    rabbitmq_ok = setup_rabbitmq()
    
    # Start event consumer
    if qdrant_ok and neo4j_ok and rabbitmq_ok:
        consumer_thread = start_event_consumer()
    
    # Log initialization results
    service_status = {
        "qdrant": "initialized" if qdrant_ok else "failed",
        "neo4j": "initialized" if neo4j_ok else "failed",
        "query_engine": "initialized" if query_engine_ok else "failed",
        "rabbitmq": "initialized" if rabbitmq_ok else "failed",
        "event_consumer": "started" if event_consumer else "not_started",
    }
    
    logger.info(f"Service initialization complete: {json.dumps(service_status)}")
    return all([qdrant_ok, neo4j_ok, query_engine_ok, rabbitmq_ok])

def run_api():
    """Run the FastAPI application."""
    uvicorn.run(
        "risk_analyzer:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )

if __name__ == "__main__":
    try:
        # Initialize services
        init_ok = init_services()
        
        if not init_ok:
            logger.error("Service initialization failed. Exiting.")
            sys.exit(1)
        
        # Start API server
        run_api()
    
    except KeyboardInterrupt:
        logger.info("Service shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
