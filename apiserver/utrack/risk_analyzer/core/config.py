"""
Risk Analyzer Configuration

This file provides default configuration values for the risk analyzer module.
These values can be overridden in the Django settings.
"""
from django.conf import settings

# Default settings that can be overridden in Django settings

# Neo4j settings
NEO4J_URI = getattr(settings, "NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = getattr(settings, "NEO4J_USER", "neo4j")
NEO4J_PASSWORD = getattr(settings, "NEO4J_PASSWORD", "utrackneo4j")

# Qdrant settings
QDRANT_HOST = getattr(settings, "QDRANT_HOST", "localhost")
QDRANT_PORT = getattr(settings, "QDRANT_PORT", 6333)
QDRANT_GRPC_PORT = getattr(settings, "QDRANT_GRPC_PORT", 6334)
QDRANT_API_KEY = getattr(settings, "QDRANT_API_KEY", None)
QDRANT_COLLECTION = getattr(settings, "QDRANT_COLLECTION", "utrack_vectors")

# RabbitMQ settings
RABBITMQ_HOST = getattr(settings, "RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = getattr(settings, "RABBITMQ_PORT", 5672)
RABBITMQ_USER = getattr(settings, "RABBITMQ_USER", "utrack")
RABBITMQ_PASSWORD = getattr(settings, "RABBITMQ_PASSWORD", "utrack")
RABBITMQ_VHOST = getattr(settings, "RABBITMQ_VHOST", "utrack")

# Event settings
EVENT_EXCHANGE = getattr(settings, "EVENT_EXCHANGE", "utrack_events")
VECTOR_QUEUE = getattr(settings, "VECTOR_QUEUE", "vector_updates")
GRAPH_QUEUE = getattr(settings, "GRAPH_QUEUE", "graph_updates")

# ML settings
EMBEDDING_MODEL = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = getattr(settings, "EMBEDDING_DIMENSION", 384) 