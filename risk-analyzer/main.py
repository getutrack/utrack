import logging
import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from risk_analyzer.api.routes import api_router
from risk_analyzer.services.neo4j import init_neo4j
from risk_analyzer.services.qdrant import init_qdrant
from risk_analyzer.services.rabbitmq import init_rabbitmq_consumer, close_rabbitmq_connection

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Utrack Risk Analyzer",
    description="RAG-based risk analysis for Utrack projects",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "status": "healthy",
        "service": "Utrack Risk Analyzer",
        "version": "0.1.0",
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Risk Analyzer service")
    
    # Initialize Neo4j
    try:
        init_neo4j()
        logger.info("Neo4j connection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j connection: {e}")
        # Don't raise here, we'll continue even if Neo4j is not available
    
    # Initialize Qdrant
    try:
        await init_qdrant()
        logger.info("Qdrant connection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant connection: {e}")
        # Don't raise here, we'll continue even if Qdrant is not available
    
    # Initialize RabbitMQ
    try:
        await init_rabbitmq_consumer()
        logger.info("RabbitMQ consumer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RabbitMQ consumer: {e}")
        # Don't raise here, we'll continue even if RabbitMQ is not available

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Risk Analyzer service")
    
    # Close RabbitMQ connection
    try:
        await close_rabbitmq_connection()
        logger.info("RabbitMQ connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing RabbitMQ connection: {e}")
    
    logger.info("Risk Analyzer service shutdown complete") 