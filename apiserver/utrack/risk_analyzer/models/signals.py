import json
import logging
import asyncio
import aio_pika
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.conf import settings

from utrack.db.models.issues import Issue, Comment
from utrack.db.models.projects import Project
from utrack.db.models.state_change import StateChange

logger = logging.getLogger(__name__)

async def publish_event(routing_key, message_data):
    """Publish an event to RabbitMQ."""
    try:
        # Connect to RabbitMQ
        connection = await aio_pika.connect_robust(
            host=settings.RABBITMQ_HOST,
            port=int(settings.RABBITMQ_PORT),
            login=settings.RABBITMQ_USER,
            password=settings.RABBITMQ_PASSWORD,
            virtualhost=settings.RABBITMQ_VHOST,
        )
        
        # Get channel
        channel = await connection.channel()
        
        # Declare exchange
        exchange = await channel.declare_exchange(
            settings.EVENT_EXCHANGE,
            aio_pika.ExchangeType.TOPIC,
            durable=True,
        )
        
        # Create message
        message = aio_pika.Message(
            body=json.dumps(message_data).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        
        # Publish message
        await exchange.publish(message, routing_key=routing_key)
        
        # Close connection
        await connection.close()
        
        logger.info(f"Published event to {routing_key}: {message_data}")
        return True
    
    except Exception as e:
        logger.error(f"Error publishing event to RabbitMQ: {e}")
        return False


@receiver(post_save, sender=Project)
def project_save_handler(sender, instance, created, **kwargs):
    """Handle project create/update events."""
    # Prepare data
    project_data = {
        "id": str(instance.id),
        "name": instance.name,
        "description": instance.description,
        "created_at": instance.created_at.isoformat() if instance.created_at else None,
        "updated_at": instance.updated_at.isoformat() if instance.updated_at else None,
    }
    
    # Determine event type
    event_type = "project.create" if created else "project.update"
    
    # Run async publish in the background
    asyncio.run(publish_event(event_type, project_data))


@receiver(post_save, sender=Issue)
def issue_save_handler(sender, instance, created, **kwargs):
    """Handle issue create/update events."""
    # Prepare data
    issue_data = {
        "id": str(instance.id),
        "project_id": str(instance.project_id),
        "title": instance.title,
        "description": instance.description,
        "state": instance.state.name if instance.state else None,
        "created_at": instance.created_at.isoformat() if instance.created_at else None,
        "updated_at": instance.updated_at.isoformat() if instance.updated_at else None,
        "creator_id": str(instance.creator_id) if instance.creator_id else None,
        "assignee_id": str(instance.assignee_id) if instance.assignee_id else None,
    }
    
    # Determine event type
    event_type = "issue.create" if created else "issue.update"
    
    # Run async publish in the background
    asyncio.run(publish_event(event_type, issue_data))


@receiver(post_save, sender=Comment)
def comment_save_handler(sender, instance, created, **kwargs):
    """Handle comment create/update events."""
    # Prepare data
    comment_data = {
        "id": str(instance.id),
        "issue_id": str(instance.issue_id),
        "project_id": str(instance.issue.project_id) if instance.issue else None,
        "content": instance.content,
        "created_at": instance.created_at.isoformat() if instance.created_at else None,
        "updated_at": instance.updated_at.isoformat() if instance.updated_at else None,
        "author_id": str(instance.author_id) if instance.author_id else None,
    }
    
    # Determine event type
    event_type = "comment.create" if created else "comment.update"
    
    # Run async publish in the background
    asyncio.run(publish_event(event_type, comment_data))


@receiver(post_save, sender=StateChange)
def state_change_handler(sender, instance, created, **kwargs):
    """Handle state change events."""
    if not created:
        # Only handle new state changes
        return
    
    # Prepare data
    state_change_data = {
        "id": str(instance.id),
        "issue_id": str(instance.issue_id),
        "from_state": instance.from_state.name if instance.from_state else None,
        "to_state": instance.to_state.name if instance.to_state else None,
        "timestamp": instance.created_at.isoformat() if instance.created_at else None,
        "user_id": str(instance.created_by_id) if instance.created_by_id else None,
    }
    
    # Publish event
    asyncio.run(publish_event("state.change", state_change_data)) 