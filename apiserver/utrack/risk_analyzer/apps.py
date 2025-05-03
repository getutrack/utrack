from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class RiskAnalyzerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apiserver.utrack.risk_analyzer'
    verbose_name = 'Utrack Risk Analyzer'
    
    def ready(self):
        """
        Initialize services when the Django app is ready.
        """
        import asyncio
        from .services.neo4j import init_neo4j
        
        # Initialize Neo4j
        try:
            init_neo4j()
            logger.info("Neo4j service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j service: {e}")
        
        # Initialize Qdrant and RabbitMQ asynchronously
        # Note: Django doesn't support async initialization directly,
        # so we'll need to initialize these services when they're first used
        # or implement a background task to initialize them 