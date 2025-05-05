"""
RabbitMQ Event Consumer for Risk Analyzer

This module implements a consumer that processes real-time events from RabbitMQ
and updates both Qdrant (vector database) and Neo4j (graph database) accordingly.

It follows an event-driven architecture to maintain consistency between the
application state and the analytics databases used for RAG functionality.
"""

import os
import json
import time
import signal
import logging
import threading
from typing import Dict, Any, Callable, Optional, List, Union
from functools import wraps
from datetime import datetime, timedelta

import pika
from pika import SelectConnection
from pika.channel import Channel
from pika.connection import Connection
from pika.exceptions import AMQPConnectionError, ChannelClosedByBroker
from dotenv import load_dotenv

# Import our modules
from embedding import EmbeddingPipeline
from graph_storage import GraphPipeline, Neo4jManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EventConsumerError(Exception):
    """Base exception for event consumer errors."""
    pass


class ConnectionError(EventConsumerError):
    """Exception raised for RabbitMQ connection errors."""
    pass


class ChannelError(EventConsumerError):
    """Exception raised for RabbitMQ channel errors."""
    pass


class EventProcessingError(EventConsumerError):
    """Exception raised for event processing errors."""
    pass


def retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    
    Returns:
        The decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_retries, delay
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    msg = f"{str(e)}, Retrying in {mdelay} seconds..."
                    if mtries - 1 > 0:
                        logger.warning(msg)
                        time.sleep(mdelay)
                        mtries -= 1
                        mdelay *= backoff
                    else:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise
        return wrapper
    return decorator


class EventConsumer:
    """
    Consumes events from RabbitMQ and processes them to update
    Qdrant and Neo4j databases.
    """
    
    def __init__(self):
        """Initialize the event consumer."""
        # RabbitMQ connection parameters
        self.host = os.getenv("RABBITMQ_HOST", "utrack-mq")
        self.port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.username = os.getenv("RABBITMQ_USER", "utrack")
        self.password = os.getenv("RABBITMQ_PASSWORD", "utrack")
        self.vhost = os.getenv("RABBITMQ_VHOST", "utrack")
        
        # RabbitMQ exchange and queue settings
        self.exchange = os.getenv("EVENT_EXCHANGE", "utrack_events")
        self.vector_queue = os.getenv("VECTOR_QUEUE", "vector_updates")
        self.graph_queue = os.getenv("GRAPH_QUEUE", "graph_updates")
        
        # RabbitMQ connection and channel
        self.connection = None
        self.channel = None
        
        # Event processing pipelines
        self.embedding_pipeline = None
        self.graph_pipeline = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
        
        # Threading event for shutdown coordination
        self.shutdown_event = threading.Event()
        
        # Initialize pipeline instances
        self._initialize_pipelines()
        
        logger.info("Event consumer initialized")
    
    def _initialize_pipelines(self):
        """Initialize the embedding and graph pipelines."""
        try:
            self.embedding_pipeline = EmbeddingPipeline()
            logger.info("Embedding pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding pipeline: {e}")
        
        try:
            self.graph_pipeline = GraphPipeline()
            logger.info("Graph pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize graph pipeline: {e}")
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()
        if self.connection:
            self.connection.ioloop.add_callback_threadsafe(self.connection.close)
    
    def _get_connection_parameters(self) -> pika.ConnectionParameters:
        """
        Get connection parameters for RabbitMQ.
        
        Returns:
            pika.ConnectionParameters: Connection parameters
        """
        credentials = pika.PlainCredentials(self.username, self.password)
        return pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
        )
    
    def connect(self) -> pika.SelectConnection:
        """
        Connect to RabbitMQ.
        
        Returns:
            pika.SelectConnection: Connection object
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            logger.info(f"Connecting to RabbitMQ at {self.host}:{self.port}")
            parameters = self._get_connection_parameters()
            return pika.SelectConnection(
                parameters=parameters,
                on_open_callback=self.on_connection_open,
                on_open_error_callback=self.on_connection_open_error,
                on_close_callback=self.on_connection_closed,
            )
        except AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise ConnectionError(f"Failed to connect to RabbitMQ: {e}")
    
    def on_connection_open(self, connection: Connection):
        """
        Called when connection is established.
        
        Args:
            connection: Connection object
        """
        logger.info("Connection to RabbitMQ established")
        self.connection = connection
        self.open_channel()
    
    def on_connection_open_error(self, connection: Connection, err: Exception):
        """
        Called if the connection can't be established.
        
        Args:
            connection: Connection object
            err: The exception
        """
        logger.error(f"Failed to open connection: {err}")
        if not self.shutdown_event.is_set():
            self.reconnect()
    
    def on_connection_closed(self, connection: Connection, reason: Exception):
        """
        Called when the connection is closed.
        
        Args:
            connection: Connection object
            reason: The reason for closure
        """
        self.channel = None
        if self.shutdown_event.is_set():
            logger.info("Connection closed due to shutdown")
            if connection.is_open:
                connection.close()
        else:
            logger.warning(f"Connection closed, reopening: {reason}")
            self.reconnect()
    
    def reconnect(self):
        """Reconnect to RabbitMQ with backoff."""
        if self.shutdown_event.is_set():
            return
            
        # Implement reconnection logic with backoff
        retry_delays = [1, 2, 5, 10, 30, 60]
        
        # We'll try to reconnect with increasing delay
        for delay in retry_delays:
            if self.shutdown_event.is_set():
                break
                
            logger.info(f"Attempting to reconnect in {delay} seconds...")
            time.sleep(delay)
            
            try:
                self.connection = self.connect()
                # Start the IO loop to process connection events
                self.connection.ioloop.start()
                break
            except ConnectionError:
                logger.warning("Reconnection attempt failed")
                continue
        
        if not self.connection or not self.connection.is_open:
            logger.error("Failed to reconnect after multiple attempts")
    
    def open_channel(self):
        """Open a new channel with RabbitMQ."""
        logger.info("Creating a new channel")
        self.connection.channel(on_open_callback=self.on_channel_open)
    
    def on_channel_open(self, channel: Channel):
        """
        Called when channel is opened.
        
        Args:
            channel: Channel object
        """
        logger.info("Channel opened")
        self.channel = channel
        self.channel.add_on_close_callback(self.on_channel_closed)
        self.setup_exchange()
    
    def on_channel_closed(self, channel: Channel, reason: Exception):
        """
        Called when channel is closed.
        
        Args:
            channel: Channel object
            reason: The reason for closure
        """
        logger.warning(f"Channel {channel} closed: {reason}")
        if not self.shutdown_event.is_set() and self.connection and self.connection.is_open:
            self.open_channel()
    
    def setup_exchange(self):
        """
        Set up the exchange on RabbitMQ.
        
        This declares a topic exchange for routing events.
        """
        logger.info(f"Declaring exchange {self.exchange}")
        # Use a durable, topic exchange
        self.channel.exchange_declare(
            exchange=self.exchange,
            exchange_type="topic",
            durable=True,
            callback=self.on_exchange_declared,
        )
    
    def on_exchange_declared(self, frame: Any):
        """
        Called when exchange declaration is complete.
        
        Args:
            frame: The response frame
        """
        logger.info(f"Exchange {self.exchange} declared")
        self.setup_queues()
    
    def setup_queues(self):
        """Set up the queues for vector and graph updates."""
        logger.info("Setting up queues")
        
        # Declare the vector updates queue
        self.channel.queue_declare(
            queue=self.vector_queue,
            durable=True,
            callback=self.on_vector_queue_declared,
        )
        
        # Declare the graph updates queue
        self.channel.queue_declare(
            queue=self.graph_queue,
            durable=True,
            callback=self.on_graph_queue_declared,
        )
    
    def on_vector_queue_declared(self, frame: Any):
        """
        Called when vector queue declaration is complete.
        
        Args:
            frame: The response frame
        """
        logger.info(f"Queue {self.vector_queue} declared")
        
        # Bind the vector queue to the exchange with routing keys
        routing_keys = [
            "issue.created",
            "issue.updated",
            "comment.created",
            "comment.updated",
            "document.uploaded",
            "document.updated",
        ]
        
        for key in routing_keys:
            self.channel.queue_bind(
                queue=self.vector_queue,
                exchange=self.exchange,
                routing_key=key,
            )
        
        logger.info(f"Bound {self.vector_queue} to {self.exchange} with keys: {routing_keys}")
    
    def on_graph_queue_declared(self, frame: Any):
        """
        Called when graph queue declaration is complete.
        
        Args:
            frame: The response frame
        """
        logger.info(f"Queue {self.graph_queue} declared")
        
        # Bind the graph queue to the exchange with routing keys
        routing_keys = [
            "project.created",
            "project.updated",
            "issue.created",
            "issue.updated",
            "issue.state_changed",
            "comment.created",
            "comment.updated",
            "document.uploaded",
            "document.updated",
            "user.created",
            "user.updated",
        ]
        
        for key in routing_keys:
            self.channel.queue_bind(
                queue=self.graph_queue,
                exchange=self.exchange,
                routing_key=key,
            )
        
        logger.info(f"Bound {self.graph_queue} to {self.exchange} with keys: {routing_keys}")
        
        # Start consuming from both queues
        self.start_consuming()
    
    def start_consuming(self):
        """Start consuming messages from the queues."""
        logger.info("Starting to consume messages")
        
        # Start consuming from vector queue
        self.channel.basic_consume(
            queue=self.vector_queue,
            on_message_callback=self.process_vector_message,
            auto_ack=False,
        )
        
        # Start consuming from graph queue
        self.channel.basic_consume(
            queue=self.graph_queue,
            on_message_callback=self.process_graph_message,
            auto_ack=False,
        )
        
        logger.info("Consumer started successfully")
    
    def process_vector_message(self, channel: Channel, method: Any, properties: Any, body: bytes):
        """
        Process messages from the vector queue.
        
        Args:
            channel: Channel object
            method: Method frame
            properties: Properties
            body: Message body
        """
        try:
            # Parse the message body
            message = json.loads(body)
            logger.info(f"Received vector message: {method.routing_key}")
            
            # Process based on the routing key
            routing_key = method.routing_key
            processed = False
            
            if routing_key.startswith("issue."):
                processed = self.process_issue_for_vector(message, routing_key)
            elif routing_key.startswith("comment."):
                processed = self.process_comment_for_vector(message, routing_key)
            elif routing_key.startswith("document."):
                processed = self.process_document_for_vector(message, routing_key)
            else:
                logger.warning(f"Unhandled vector routing key: {routing_key}")
            
            if processed:
                channel.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(f"Vector message processed successfully: {routing_key}")
            else:
                # Requeue the message
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                logger.warning(f"Failed to process vector message, requeued: {routing_key}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in vector message: {e}")
            # Don't requeue if the message is malformed
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
        except Exception as e:
            logger.error(f"Error processing vector message: {e}", exc_info=True)
            # Requeue the message to try again later
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def process_graph_message(self, channel: Channel, method: Any, properties: Any, body: bytes):
        """
        Process messages from the graph queue.
        
        Args:
            channel: Channel object
            method: Method frame
            properties: Properties
            body: Message body
        """
        try:
            # Parse the message body
            message = json.loads(body)
            logger.info(f"Received graph message: {method.routing_key}")
            
            # Process based on the routing key
            routing_key = method.routing_key
            processed = False
            
            if routing_key.startswith("project."):
                processed = self.process_project_for_graph(message, routing_key)
            elif routing_key.startswith("issue."):
                processed = self.process_issue_for_graph(message, routing_key)
            elif routing_key.startswith("comment."):
                processed = self.process_comment_for_graph(message, routing_key)
            elif routing_key.startswith("document."):
                processed = self.process_document_for_graph(message, routing_key)
            elif routing_key.startswith("user."):
                processed = self.process_user_for_graph(message, routing_key)
            else:
                logger.warning(f"Unhandled graph routing key: {routing_key}")
            
            if processed:
                channel.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(f"Graph message processed successfully: {routing_key}")
            else:
                # Requeue the message
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                logger.warning(f"Failed to process graph message, requeued: {routing_key}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in graph message: {e}")
            # Don't requeue if the message is malformed
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
        except Exception as e:
            logger.error(f"Error processing graph message: {e}", exc_info=True)
            # Requeue the message to try again later
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,))
    def process_issue_for_vector(self, message: Dict[str, Any], routing_key: str) -> bool:
        """
        Process an issue event for vector updates.
        
        Args:
            message: Message data
            routing_key: Routing key
        
        Returns:
            bool: True if processed successfully
        
        Raises:
            EventProcessingError: If processing fails
        """
        if not self.embedding_pipeline:
            logger.error("Embedding pipeline not initialized")
            return False
        
        try:
            # Extract issue data
            issue_data = message.get("data", {})
            issue_id = issue_data.get("id")
            
            if not issue_id:
                logger.error("Missing issue ID in message")
                return False
            
            # Create or update vector embedding
            vector_id = f"issue_{issue_id}"
            
            # Combine title and description for better embedding context
            text = f"{issue_data.get('title', '')} {issue_data.get('description', '')}"
            if not text.strip():
                logger.warning(f"Empty text for issue {issue_id}, skipping vector update")
                return True  # Consider it processed, but nothing to do
            
            # Generate and store embedding
            metadata = {
                "id": issue_id,
                "type": "issue",
                "title": issue_data.get("title", ""),
                "project_id": issue_data.get("project_id", ""),
                "updated_at": issue_data.get("updated_at", datetime.now().isoformat()),
            }
            
            result = self.embedding_pipeline.process_text(
                text=text,
                vector_id=vector_id,
                metadata=metadata
            )
            
            logger.info(f"Created/updated vector for issue {issue_id}: {result}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to process issue for vector update: {e}")
            raise EventProcessingError(f"Failed to process issue for vector: {e}")
    
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,))
    def process_comment_for_vector(self, message: Dict[str, Any], routing_key: str) -> bool:
        """
        Process a comment event for vector updates.
        
        Args:
            message: Message data
            routing_key: Routing key
        
        Returns:
            bool: True if processed successfully
        
        Raises:
            EventProcessingError: If processing fails
        """
        if not self.embedding_pipeline:
            logger.error("Embedding pipeline not initialized")
            return False
        
        try:
            # Extract comment data
            comment_data = message.get("data", {})
            comment_id = comment_data.get("id")
            
            if not comment_id:
                logger.error("Missing comment ID in message")
                return False
            
            # Create or update vector embedding
            vector_id = f"comment_{comment_id}"
            
            # Get the comment content
            text = comment_data.get("content", "")
            if not text.strip():
                logger.warning(f"Empty text for comment {comment_id}, skipping vector update")
                return True  # Consider it processed, but nothing to do
            
            # Generate and store embedding
            metadata = {
                "id": comment_id,
                "type": "comment",
                "issue_id": comment_data.get("issue_id", ""),
                "author_id": comment_data.get("author_id", ""),
                "created_at": comment_data.get("created_at", datetime.now().isoformat()),
            }
            
            result = self.embedding_pipeline.process_text(
                text=text,
                vector_id=vector_id,
                metadata=metadata
            )
            
            logger.info(f"Created/updated vector for comment {comment_id}: {result}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to process comment for vector update: {e}")
            raise EventProcessingError(f"Failed to process comment for vector: {e}")
    
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,))
    def process_document_for_vector(self, message: Dict[str, Any], routing_key: str) -> bool:
        """
        Process a document event for vector updates.
        
        Args:
            message: Message data
            routing_key: Routing key
        
        Returns:
            bool: True if processed successfully
        
        Raises:
            EventProcessingError: If processing fails
        """
        if not self.embedding_pipeline:
            logger.error("Embedding pipeline not initialized")
            return False
        
        try:
            # Extract document data
            document_data = message.get("data", {})
            document_id = document_data.get("id")
            
            if not document_id:
                logger.error("Missing document ID in message")
                return False
            
            # Extract document info
            name = document_data.get("name", "")
            content_type = document_data.get("content_type", "")
            bucket = document_data.get("bucket", os.getenv("AWS_S3_BUCKET_NAME", "uploads"))
            
            if not name:
                logger.error(f"Missing name for document {document_id}")
                return False
            
            # Create or update vector embedding from document content
            vector_id = f"document_{document_id}"
            
            # Process and extract text from the document
            text = self.embedding_pipeline.extract_document_text(
                document_name=name,
                bucket=bucket,
                content_type=content_type
            )
            
            if not text:
                logger.warning(f"No text extracted from document {document_id}, skipping vector update")
                return True  # Consider it processed, but nothing to do
            
            # Generate and store embedding
            metadata = {
                "id": document_id,
                "type": "document",
                "name": name,
                "content_type": content_type,
                "project_id": document_data.get("project_id", ""),
                "uploaded_by": document_data.get("uploaded_by", ""),
                "last_modified": document_data.get("last_modified", datetime.now().isoformat()),
            }
            
            result = self.embedding_pipeline.process_text(
                text=text,
                vector_id=vector_id,
                metadata=metadata
            )
            
            logger.info(f"Created/updated vector for document {document_id}: {result}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to process document for vector update: {e}")
            raise EventProcessingError(f"Failed to process document for vector: {e}")
    
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,))
    def process_project_for_graph(self, message: Dict[str, Any], routing_key: str) -> bool:
        """
        Process a project event for graph updates.
        
        Args:
            message: Message data
            routing_key: Routing key
        
        Returns:
            bool: True if processed successfully
        
        Raises:
            EventProcessingError: If processing fails
        """
        if not self.graph_pipeline:
            logger.error("Graph pipeline not initialized")
            return False
        
        try:
            # Extract project data
            project_data = message.get("data", {})
            project_id = project_data.get("id")
            
            if not project_id:
                logger.error("Missing project ID in message")
                return False
            
            # Create or update project in Neo4j
            self.graph_pipeline.process_projects([project_data])
            logger.info(f"Created/updated graph node for project {project_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to process project for graph update: {e}")
            raise EventProcessingError(f"Failed to process project for graph: {e}")
    
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,))
    def process_issue_for_graph(self, message: Dict[str, Any], routing_key: str) -> bool:
        """
        Process an issue event for graph updates.
        
        Args:
            message: Message data
            routing_key: Routing key
        
        Returns:
            bool: True if processed successfully
        
        Raises:
            EventProcessingError: If processing fails
        """
        if not self.graph_pipeline:
            logger.error("Graph pipeline not initialized")
            return False
        
        try:
            # Extract issue data
            issue_data = message.get("data", {})
            issue_id = issue_data.get("id")
            
            if not issue_id:
                logger.error("Missing issue ID in message")
                return False
            
            if routing_key == "issue.state_changed":
                # Handle state change event
                state_change_data = {
                    "issue_id": issue_id,
                    "from_state": message.get("from_state", ""),
                    "to_state": message.get("to_state", ""),
                    "user_id": message.get("changed_by", ""),
                    "timestamp": message.get("timestamp", datetime.now().isoformat()),
                }
                
                # Create state change node
                self.graph_pipeline.process_state_changes([state_change_data])
                logger.info(f"Created state change node for issue {issue_id}")
            else:
                # Handle create/update event
                # Create or update issue in Neo4j
                self.graph_pipeline.process_issues([issue_data])
                logger.info(f"Created/updated graph node for issue {issue_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to process issue for graph update: {e}")
            raise EventProcessingError(f"Failed to process issue for graph: {e}")
    
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,))
    def process_comment_for_graph(self, message: Dict[str, Any], routing_key: str) -> bool:
        """
        Process a comment event for graph updates.
        
        Args:
            message: Message data
            routing_key: Routing key
        
        Returns:
            bool: True if processed successfully
        
        Raises:
            EventProcessingError: If processing fails
        """
        if not self.graph_pipeline:
            logger.error("Graph pipeline not initialized")
            return False
        
        try:
            # Extract comment data
            comment_data = message.get("data", {})
            comment_id = comment_data.get("id")
            
            if not comment_id:
                logger.error("Missing comment ID in message")
                return False
            
            # Create or update comment in Neo4j
            self.graph_pipeline.process_comments([comment_data])
            logger.info(f"Created/updated graph node for comment {comment_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to process comment for graph update: {e}")
            raise EventProcessingError(f"Failed to process comment for graph: {e}")
    
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,))
    def process_document_for_graph(self, message: Dict[str, Any], routing_key: str) -> bool:
        """
        Process a document event for graph updates.
        
        Args:
            message: Message data
            routing_key: Routing key
        
        Returns:
            bool: True if processed successfully
        
        Raises:
            EventProcessingError: If processing fails
        """
        if not self.graph_pipeline:
            logger.error("Graph pipeline not initialized")
            return False
        
        try:
            # Extract document data
            document_data = message.get("data", {})
            document_id = document_data.get("id")
            
            if not document_id:
                logger.error("Missing document ID in message")
                return False
            
            # Create or update document in Neo4j
            self.graph_pipeline.process_documents([document_data])
            logger.info(f"Created/updated graph node for document {document_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to process document for graph update: {e}")
            raise EventProcessingError(f"Failed to process document for graph: {e}")
    
    @retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,))
    def process_user_for_graph(self, message: Dict[str, Any], routing_key: str) -> bool:
        """
        Process a user event for graph updates.
        
        Args:
            message: Message data
            routing_key: Routing key
        
        Returns:
            bool: True if processed successfully
        
        Raises:
            EventProcessingError: If processing fails
        """
        if not self.graph_pipeline:
            logger.error("Graph pipeline not initialized")
            return False
        
        try:
            # Extract user data
            user_data = message.get("data", {})
            user_id = user_data.get("id")
            
            if not user_id:
                logger.error("Missing user ID in message")
                return False
            
            # We don't process users directly as they are created as relationships
            # from issues, comments, etc.
            logger.info(f"User {user_id} event received, but no direct action needed")
            return True
        
        except Exception as e:
            logger.error(f"Failed to process user for graph update: {e}")
            raise EventProcessingError(f"Failed to process user for graph: {e}")
    
    def run(self):
        """Start the event consumer."""
        try:
            # Connect to RabbitMQ
            self.connection = self.connect()
            
            # Start the IO loop to process connection events
            self.connection.ioloop.start()
        except KeyboardInterrupt:
            logger.info("Interrupted by user, shutting down...")
            if self.connection and not self.connection.is_closed:
                # Close the connection
                self.connection.close()
                # Wait for the connection to close
                self.connection.ioloop.start()
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            if self.connection and not self.connection.is_closed:
                # Close the connection
                self.connection.close()
                # Wait for the connection to close
                self.connection.ioloop.start()


def main():
    """Main entry point for the event consumer."""
    try:
        # Create and run the event consumer
        consumer = EventConsumer()
        consumer.run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 