"""
RabbitMQ Event Producer for Django Models

This module provides functionality to publish Django model events to RabbitMQ.
It can be integrated with Django's signals to automatically publish events
when models are created, updated, or when state changes occur.

Usage:
    1. Import signal handlers in Django apps.py
    2. Connect signal handlers to model signals
    3. Use change_issue_state method for issue state transitions
"""

import json
import logging
import pika
import os
import time
import datetime
from functools import wraps
from threading import Thread, Lock
from django.conf import settings
from django.db import transaction
from django.forms.models import model_to_dict
from django.db.models import Model
from django.db.models.query import QuerySet
from django.dispatch import Signal

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Custom signal for state changes
issue_state_changed = Signal(providing_args=["issue", "from_state", "to_state", "changed_by"])

# RabbitMQ configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "utrack-mq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "utrack")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "utrack")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "utrack")
EVENT_EXCHANGE = os.getenv("EVENT_EXCHANGE", "utrack_events")
ENABLE_EVENT_PUBLISHING = os.getenv("ENABLE_EVENT_PUBLISHING", "True") == "True"

# Connection singleton
_connection = None
_channel = None
_lock = Lock()

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return super().default(obj)

def get_connection():
    """Get or create RabbitMQ connection"""
    global _connection, _channel
    with _lock:
        if _connection is None or _connection.is_closed:
            try:
                credentials = pika.PlainCredentials(
                    RABBITMQ_USER,
                    RABBITMQ_PASSWORD
                )
                parameters = pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    port=RABBITMQ_PORT,
                    virtual_host=RABBITMQ_VHOST,
                    credentials=credentials,
                    heartbeat=600,
                    blocked_connection_timeout=300,
                )
                _connection = pika.BlockingConnection(parameters)
                logger.info("RabbitMQ connection established")
                _channel = _connection.channel()
                
                # Ensure exchange exists
                _channel.exchange_declare(
                    exchange=EVENT_EXCHANGE,
                    exchange_type='topic',
                    durable=True
                )
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ: {e}")
                _connection = None
                _channel = None
                raise
            
    return _connection, _channel

def retry(max_retries=3, delay=1, backoff=2):
    """Retry decorator with exponential backoff for RabbitMQ operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip if event publishing is disabled
            if not ENABLE_EVENT_PUBLISHING:
                logger.debug("Event publishing is disabled, skipping")
                return True
                
            mtries, mdelay = max_retries, delay
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except pika.exceptions.AMQPConnectionError as e:
                    mtries -= 1
                    if mtries == 0:
                        logger.error(f"Failed to publish message after {max_retries} retries: {e}")
                        raise
                    logger.warning(f"Connection error, retrying in {mdelay}s: {e}")
                    time.sleep(mdelay)
                    mdelay *= backoff
                    # Reconnect
                    global _connection, _channel
                    with _lock:
                        if _connection and not _connection.is_closed:
                            try:
                                _connection.close()
                            except:
                                pass
                        _connection = None
                        _channel = None
        return wrapper
    return decorator

@retry(max_retries=3, delay=1, backoff=2)
def publish_event(routing_key, payload):
    """
    Publish an event to RabbitMQ with the given routing key
    
    Args:
        routing_key: The routing key for the event
        payload: The event payload
    
    Returns:
        bool: True if successful, raises exception otherwise
    """
    try:
        _, channel = get_connection()
        
        message_body = json.dumps(payload, cls=DateTimeEncoder).encode('utf-8')
        channel.basic_publish(
            exchange=EVENT_EXCHANGE,
            routing_key=routing_key,
            body=message_body,
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type='application/json',
            )
        )
        logger.info(f"Published event with routing key: {routing_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to publish event {routing_key}: {e}")
        raise

# Background Publisher for non-blocking operations
class BackgroundPublisher(Thread):
    def __init__(self, routing_key, payload):
        Thread.__init__(self)
        self.routing_key = routing_key
        self.payload = payload
        self.daemon = True
        
    def run(self):
        try:
            publish_event(self.routing_key, self.payload)
        except Exception as e:
            logger.error(f"Background publishing error: {e}")

def publish_event_async(routing_key, payload):
    """
    Publish an event in a background thread to avoid blocking
    
    Args:
        routing_key: The routing key for the event
        payload: The event payload
    
    Returns:
        bool: True if the background thread was started
    """
    publisher = BackgroundPublisher(routing_key, payload)
    publisher.start()
    return True

def serialize_model(instance, fields=None, exclude=None):
    """
    Convert a Django model instance to a serializable dictionary
    
    Args:
        instance: Model instance to serialize
        fields: Optional list of field names to include
        exclude: Optional list of field names to exclude
    
    Returns:
        Dict with serialized model data
    """
    if not isinstance(instance, Model):
        return instance
        
    # Use model_to_dict for basic serialization
    data = model_to_dict(instance, fields=fields, exclude=exclude)
    
    # Add the primary key (id) field if not already included
    pk_name = instance._meta.pk.name
    if pk_name not in data:
        data[pk_name] = instance.pk
        
    # Convert datetime objects to ISO format strings
    for key, value in list(data.items()):
        # Handle datetime objects
        if isinstance(value, (datetime.datetime, datetime.date)):
            data[key] = value.isoformat()
        # Handle related objects
        elif isinstance(value, Model):
            data[key] = value.pk
        # Handle QuerySets
        elif isinstance(value, QuerySet):
            data[key] = [item.pk for item in value]
            
    return data

#
# Signal Handlers
#

def handle_project_save(sender, instance, created, **kwargs):
    """
    Signal handler for Project post_save.
    
    Args:
        sender: Model class that sent the signal
        instance: The actual instance being saved
        created: Boolean; True if a new record was created
    """
    event_type = "project.created" if created else "project.updated"
    data = serialize_model(instance)
    
    # Use transaction.on_commit to ensure DB transaction completes first
    transaction.on_commit(
        lambda: publish_event_async(event_type, {"data": data})
    )
    logger.debug(f"Triggered {event_type} event for project {instance.pk}")

def handle_issue_save(sender, instance, created, **kwargs):
    """
    Signal handler for Issue post_save.
    
    Args:
        sender: Model class that sent the signal
        instance: The actual instance being saved
        created: Boolean; True if a new record was created
    """
    event_type = "issue.created" if created else "issue.updated"
    data = serialize_model(instance)
    
    transaction.on_commit(
        lambda: publish_event_async(event_type, {"data": data})
    )
    logger.debug(f"Triggered {event_type} event for issue {instance.pk}")

def handle_comment_save(sender, instance, created, **kwargs):
    """
    Signal handler for Comment post_save.
    
    Args:
        sender: Model class that sent the signal
        instance: The actual instance being saved
        created: Boolean; True if a new record was created
    """
    event_type = "comment.created" if created else "comment.updated"
    data = serialize_model(instance)
    
    transaction.on_commit(
        lambda: publish_event_async(event_type, {"data": data})
    )
    logger.debug(f"Triggered {event_type} event for comment {instance.pk}")

def handle_document_save(sender, instance, created, **kwargs):
    """
    Signal handler for Document post_save.
    
    Args:
        sender: Model class that sent the signal
        instance: The actual instance being saved
        created: Boolean; True if a new record was created
    """
    event_type = "document.uploaded" if created else "document.updated"
    data = serialize_model(instance)
    
    transaction.on_commit(
        lambda: publish_event_async(event_type, {"data": data})
    )
    logger.debug(f"Triggered {event_type} event for document {instance.pk}")

def handle_user_save(sender, instance, created, **kwargs):
    """
    Signal handler for User post_save.
    
    Args:
        sender: Model class that sent the signal
        instance: The actual instance being saved
        created: Boolean; True if a new record was created
    """
    event_type = "user.created" if created else "user.updated"
    data = serialize_model(instance)
    
    transaction.on_commit(
        lambda: publish_event_async(event_type, {"data": data})
    )
    logger.debug(f"Triggered {event_type} event for user {instance.pk}")

def handle_issue_state_change(sender, issue, from_state, to_state, changed_by, **kwargs):
    """
    Signal handler for issue_state_changed signal.
    
    Args:
        sender: Model class that sent the signal
        issue: The issue instance being changed
        from_state: Original state
        to_state: New state
        changed_by: User who made the change
    """
    data = {
        "data": serialize_model(issue),
        "from_state": from_state,
        "to_state": to_state,
        "changed_by": str(changed_by.id) if changed_by else None,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    transaction.on_commit(
        lambda: publish_event_async("issue.state_changed", data)
    )
    logger.debug(f"Triggered issue.state_changed event for issue {issue.pk}")

#
# Service Methods
#

def change_issue_state(issue, new_state, user=None):
    """
    Change the state of an issue and emit appropriate signals
    
    Args:
        issue: The Issue instance to update
        new_state: The new state value
        user: The User making the change (optional)
    
    Returns:
        bool: True if state was changed, False if no change needed
    """    
    old_state = issue.state
    
    # Only emit signal if state actually changes
    if old_state != new_state:
        # Save the old state before updating
        issue.state = new_state
        issue.save(update_fields=['state', 'updated_at'])
        
        # Fire the custom signal
        issue_state_changed.send(
            sender=issue.__class__,
            issue=issue,
            from_state=old_state,
            to_state=new_state,
            changed_by=user
        )
        
        return True
    return False

#
# Helper Functions
#

def setup_signal_handlers():
    """
    Connect the signal handlers to the Django models.
    
    This function should be called from the apps.py file to register the
    signal handlers when the Django application starts.
    
    Example:
        # In your apps.py file:
        from django.apps import AppConfig
        
        class YourAppConfig(AppConfig):
            name = 'your_app'
            
            def ready(self):
                from risk_analyzer.event_producer import setup_signal_handlers
                setup_signal_handlers()
    """
    from django.db.models.signals import post_save
    
    try:
        # Import models - adjust these imports based on your actual project structure
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        from projects.models import Project, Issue, Comment, Document
        
        # Connect signal handlers
        post_save.connect(handle_project_save, sender=Project)
        post_save.connect(handle_issue_save, sender=Issue)
        post_save.connect(handle_comment_save, sender=Comment)
        post_save.connect(handle_document_save, sender=Document)
        post_save.connect(handle_user_save, sender=User)
        
        # Connect custom signal
        issue_state_changed.connect(handle_issue_state_change)
        
        logger.info("Connected RabbitMQ event producer signal handlers")
        return True
    except ImportError as e:
        logger.error(f"Could not connect signal handlers: {e}")
        return False

def check_rabbitmq_health():
    """
    Check if RabbitMQ is available and we can connect
    
    Returns:
        tuple: (is_healthy, message)
    """
    try:
        # Create connection
        credentials = pika.PlainCredentials(
            RABBITMQ_USER,
            RABBITMQ_PASSWORD
        )
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            virtual_host=RABBITMQ_VHOST,
            credentials=credentials,
            connection_attempts=2,
            retry_delay=1
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        # Check if our exchange exists
        try:
            channel.exchange_declare(
                exchange=EVENT_EXCHANGE,
                exchange_type='topic',
                durable=True,
                passive=True  # Just check if it exists
            )
            exchange_exists = True
        except pika.exceptions.ChannelClosedByBroker:
            # Exchange doesn't exist
            exchange_exists = False
            # Re-open channel
            connection.close()
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
        
        # Close connection
        connection.close()
        
        if exchange_exists:
            return True, "RabbitMQ connection successful and exchange exists"
        else:
            return False, "RabbitMQ connection successful but exchange doesn't exist"
            
    except Exception as e:
        logger.error(f"RabbitMQ health check failed: {e}")
        return False, f"RabbitMQ connection failed: {str(e)}"

def setup_rabbitmq():
    """
    Set up RabbitMQ exchanges and queues.
    
    Returns:
        bool: True if setup was successful
    """
    try:
        # Create connection
        credentials = pika.PlainCredentials(
            RABBITMQ_USER, 
            RABBITMQ_PASSWORD
        )
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            virtual_host=RABBITMQ_VHOST,
            credentials=credentials
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        # Declare exchange
        logger.info(f"Declaring exchange: {EVENT_EXCHANGE}")
        channel.exchange_declare(
            exchange=EVENT_EXCHANGE,
            exchange_type='topic',
            durable=True
        )
        
        # Declare vector queue
        VECTOR_QUEUE = os.getenv("VECTOR_QUEUE", "vector_updates")
        logger.info(f"Declaring queue: {VECTOR_QUEUE}")
        channel.queue_declare(
            queue=VECTOR_QUEUE,
            durable=True
        )
        
        # Declare graph queue
        GRAPH_QUEUE = os.getenv("GRAPH_QUEUE", "graph_updates")
        logger.info(f"Declaring queue: {GRAPH_QUEUE}")
        channel.queue_declare(
            queue=GRAPH_QUEUE,
            durable=True
        )
        
        # Bind vector queue
        vector_routing_keys = [
            "issue.created",
            "issue.updated",
            "comment.created",
            "comment.updated",
            "document.uploaded",
            "document.updated",
        ]
        
        for key in vector_routing_keys:
            logger.info(f"Binding {VECTOR_QUEUE} to {key}")
            channel.queue_bind(
                queue=VECTOR_QUEUE,
                exchange=EVENT_EXCHANGE,
                routing_key=key
            )
        
        # Bind graph queue
        graph_routing_keys = [
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
        
        for key in graph_routing_keys:
            logger.info(f"Binding {GRAPH_QUEUE} to {key}")
            channel.queue_bind(
                queue=GRAPH_QUEUE,
                exchange=EVENT_EXCHANGE,
                routing_key=key
            )
        
        connection.close()
        logger.info("RabbitMQ setup complete")
        return True
    except Exception as e:
        logger.error(f"RabbitMQ setup failed: {e}")
        return False

if __name__ == "__main__":
    # When run as a script, set up RabbitMQ
    setup_rabbitmq() 