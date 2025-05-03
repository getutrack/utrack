import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

class LogConfig(BaseModel):
    """Logging configuration"""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Neo4jConfig(BaseModel):
    """Neo4j connection configuration"""
    uri: str = os.getenv("NEO4J_URI", "bolt://utrack-neo4j:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "utrackneo4j")


class QdrantConfig(BaseModel):
    """Qdrant connection configuration"""
    host: str = os.getenv("QDRANT_HOST", "utrack-qdrant")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    collection: str = os.getenv("QDRANT_COLLECTION", "utrack_vectors")


class RabbitMQConfig(BaseModel):
    """RabbitMQ connection configuration"""
    host: str = os.getenv("RABBITMQ_HOST", "utrack-mq")
    port: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    user: str = os.getenv("RABBITMQ_USER", "utrack")
    password: str = os.getenv("RABBITMQ_PASSWORD", "utrack")
    vhost: str = os.getenv("RABBITMQ_VHOST", "utrack")


class EventConfig(BaseModel):
    """Event exchange and queue configuration"""
    exchange: str = os.getenv("EVENT_EXCHANGE", "utrack_events")
    vector_queue: str = os.getenv("VECTOR_QUEUE", "vector_updates")
    graph_queue: str = os.getenv("GRAPH_QUEUE", "graph_updates")


class MLConfig(BaseModel):
    """Machine learning configuration"""
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))


class APIConfig(BaseModel):
    """API configuration"""
    workers: int = int(os.getenv("WORKERS", "2"))
    cors_origins: List[str] = ["*"]  # In production, replace with specific origins


class Settings(BaseModel):
    """Application settings"""
    log: LogConfig = LogConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    qdrant: QdrantConfig = QdrantConfig()
    rabbitmq: RabbitMQConfig = RabbitMQConfig()
    events: EventConfig = EventConfig()
    ml: MLConfig = MLConfig()
    api: APIConfig = APIConfig()


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings 