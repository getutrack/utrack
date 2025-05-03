import logging
import json
import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable

import aio_pika
from aio_pika.abc import AbstractRobustConnection, AbstractChannel

from django.conf import settings
from app.services.embedding import generate_embedding
from app.services.qdrant import store_embeddings
from app.services.neo4j import (
    create_project_node,
    create_issue_node,
    create_comment_node,
    create_state_change_node,
)

logger = logging.getLogger(__name__)

# Global RabbitMQ connection and channel
_connection: Optional[AbstractRobustConnection] = None
_channel: Optional[AbstractChannel] = None

# Handler mapping for different event types
_vector_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[bool]]] = {}
_graph_handlers: Dict[str, Callable[[Dict[str, Any]], bool]] = {}


async def init_rabbitmq_consumer() -> None:
    """Initialize RabbitMQ connection and consumer."""
    global _connection, _channel
    
    # Connect to RabbitMQ
    _connection = await aio_pika.connect_robust(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        login=settings.RABBITMQ_USER,
        password=settings.RABBITMQ_PASSWORD,
        virtualhost=settings.RABBITMQ_VHOST,
    )
    
    # Create channel
    _channel = await _connection.channel()
    
    # Declare exchange
    exchange = await _channel.declare_exchange(
        settings.EVENT_EXCHANGE,
        aio_pika.ExchangeType.TOPIC,
        durable=True,
    )
    
    # Register handlers
    register_event_handlers()
    
    # Declare queues
    vector_queue = await _channel.declare_queue(
        settings.VECTOR_QUEUE,
        durable=True,
    )
    graph_queue = await _channel.declare_queue(
        settings.GRAPH_QUEUE,
        durable=True,
    )
    
    # Bind queues to exchange
    await vector_queue.bind(exchange, "issue.#")
    await vector_queue.bind(exchange, "comment.#")
    await vector_queue.bind(exchange, "file.#")
    
    await graph_queue.bind(exchange, "issue.#")
    await graph_queue.bind(exchange, "comment.#")
    await graph_queue.bind(exchange, "state.#")
    await graph_queue.bind(exchange, "user.#")
    await graph_queue.bind(exchange, "project.#")
    
    # Start consuming
    await vector_queue.consume(handle_vector_message)
    await graph_queue.consume(handle_graph_message)
    
    logger.info("RabbitMQ consumer initialized successfully")


async def close_rabbitmq_connection() -> None:
    """Close RabbitMQ connection."""
    global _connection, _channel
    
    if _channel:
        await _channel.close()
        _channel = None
    
    if _connection:
        await _connection.close()
        _connection = None
    
    logger.info("RabbitMQ connection closed")


def register_event_handlers() -> None:
    """Register handlers for different event types."""
    global _vector_handlers, _graph_handlers
    
    # Vector handlers
    _vector_handlers.update({
        "issue.create": handle_issue_vector_create,
        "issue.update": handle_issue_vector_update,
        "comment.create": handle_comment_vector_create,
        "comment.update": handle_comment_vector_update,
    })
    
    # Graph handlers
    _graph_handlers.update({
        "project.create": handle_project_create,
        "project.update": handle_project_update,
        "issue.create": handle_issue_create,
        "issue.update": handle_issue_update,
        "comment.create": handle_comment_create,
        "comment.update": handle_comment_update,
        "state.change": handle_state_change,
    })


async def handle_vector_message(message: aio_pika.IncomingMessage) -> None:
    """Handle messages for vector updates."""
    async with message.process():
        try:
            # Parse message
            event_data = json.loads(message.body.decode())
            event_type = message.routing_key
            
            logger.info(f"Received vector update event: {event_type}")
            
            # Get handler for event type
            handler = _vector_handlers.get(event_type)
            if handler:
                # Process event
                success = await handler(event_data)
                if success:
                    logger.info(f"Successfully processed vector event: {event_type}")
                else:
                    logger.error(f"Failed to process vector event: {event_type}")
            else:
                logger.warning(f"No handler for vector event type: {event_type}")
        
        except Exception as e:
            logger.error(f"Error processing vector message: {e}")


async def handle_graph_message(message: aio_pika.IncomingMessage) -> None:
    """Handle messages for graph updates."""
    async with message.process():
        try:
            # Parse message
            event_data = json.loads(message.body.decode())
            event_type = message.routing_key
            
            logger.info(f"Received graph update event: {event_type}")
            
            # Get handler for event type
            handler = _graph_handlers.get(event_type)
            if handler:
                # Process event
                success = handler(event_data)
                if success:
                    logger.info(f"Successfully processed graph event: {event_type}")
                else:
                    logger.error(f"Failed to process graph event: {event_type}")
            else:
                logger.warning(f"No handler for graph event type: {event_type}")
        
        except Exception as e:
            logger.error(f"Error processing graph message: {e}")


# Vector handlers

async def handle_issue_vector_create(data: Dict[str, Any]) -> bool:
    """Handle issue creation for vector storage."""
    try:
        # Generate text for embedding
        text = f"Issue: {data.get('title', '')}. Description: {data.get('description', '')}"
        
        # Generate embedding
        embedding = await generate_embedding(text)
        
        # Generate vector ID
        vector_id = f"issue_{data['id']}"
        
        # Create metadata
        metadata = {
            "type": "issue",
            "id": str(data.get('id')),
            "project_id": str(data.get('project_id')),
            "text": text,
            "created_at": data.get('created_at'),
            "state": data.get('state', ''),
            "title": data.get('title', ''),
        }
        
        # Store embedding
        success = await store_embeddings(
            vectors=[embedding],
            ids=[vector_id],
            metadata=[metadata],
        )
        
        return success
    except Exception as e:
        logger.error(f"Error creating issue vector: {e}")
        return False


async def handle_issue_vector_update(data: Dict[str, Any]) -> bool:
    """Handle issue update for vector storage."""
    # For updates, we just recreate the vector the same way as for creation
    return await handle_issue_vector_create(data)


async def handle_comment_vector_create(data: Dict[str, Any]) -> bool:
    """Handle comment creation for vector storage."""
    try:
        # Generate text for embedding
        text = f"Comment on issue {data.get('issue_id')}: {data.get('content', '')}"
        
        # Generate embedding
        embedding = await generate_embedding(text)
        
        # Generate vector ID
        vector_id = f"comment_{data['id']}"
        
        # Create metadata
        metadata = {
            "type": "comment",
            "id": str(data.get('id')),
            "issue_id": str(data.get('issue_id')),
            "project_id": str(data.get('project_id')),
            "text": text,
            "created_at": data.get('created_at'),
            "author_id": str(data.get('author_id', '')),
        }
        
        # Store embedding
        success = await store_embeddings(
            vectors=[embedding],
            ids=[vector_id],
            metadata=[metadata],
        )
        
        return success
    except Exception as e:
        logger.error(f"Error creating comment vector: {e}")
        return False


async def handle_comment_vector_update(data: Dict[str, Any]) -> bool:
    """Handle comment update for vector storage."""
    # For updates, we just recreate the vector the same way as for creation
    return await handle_comment_vector_create(data)


# Graph handlers

def handle_project_create(data: Dict[str, Any]) -> bool:
    """Handle project creation for Neo4j."""
    return create_project_node(data)


def handle_project_update(data: Dict[str, Any]) -> bool:
    """Handle project update for Neo4j."""
    return create_project_node(data)


def handle_issue_create(data: Dict[str, Any]) -> bool:
    """Handle issue creation for Neo4j."""
    return create_issue_node(data)


def handle_issue_update(data: Dict[str, Any]) -> bool:
    """Handle issue update for Neo4j."""
    return create_issue_node(data)


def handle_comment_create(data: Dict[str, Any]) -> bool:
    """Handle comment creation for Neo4j."""
    return create_comment_node(data)


def handle_comment_update(data: Dict[str, Any]) -> bool:
    """Handle comment update for Neo4j."""
    return create_comment_node(data)


def handle_state_change(data: Dict[str, Any]) -> bool:
    """Handle state change for Neo4j."""
    return create_state_change_node(data) 