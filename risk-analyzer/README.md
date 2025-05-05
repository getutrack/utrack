# Risk Analyzer Module

The Risk Analyzer module is a component of the Utrack application that analyzes project data to identify potential risks, inefficiencies, and opportunities for improvement.

## Features

- Project risk analysis
- Issue risk assessment
- Team dynamics analysis
- Workflow optimization
- Vector search using Qdrant
- Graph-based analysis with Neo4j
- Hybrid search combining vector and graph databases
- Real-time event processing with RabbitMQ

## Architecture

The risk-analyzer is implemented as a standalone FastAPI service that integrates with the main Utrack application through:

1. Database access (PostgreSQL)
2. Object storage (Minio)
3. Cache (Redis)
4. Message queue (RabbitMQ)
5. Vector database (Qdrant)
6. Graph database (Neo4j)

## Event-Driven Processing

The Risk Analyzer implements an event-driven architecture using RabbitMQ to process changes in real-time:

### Event Consumer

The `event_consumer.py` module implements a robust RabbitMQ consumer that:

- Consumes events from two primary queues: `vector_updates` and `graph_updates`
- Updates Qdrant vector database with embeddings for text data
- Updates Neo4j graph database with entity relationships
- Handles connection issues and retries with exponential backoff
- Processes all entity types (projects, issues, comments, documents)

### Event Producer

The `event_producer.py` module provides a way for the Django application to publish events:

- Connects to model signals to automatically publish events on data changes 
- Serializes Django models to JSON for event payloads
- Provides utility functions for manual event publishing
- Implements transaction-aware event publishing
- Includes health checks and setup utilities

## Integration with Django

To integrate the event producer with your Django application:

1. Install the required dependencies:
   - `pip install pika`

2. Import and set up the signal handlers in your app's `apps.py`:
   ```python
   from django.apps import AppConfig

   class YourAppConfig(AppConfig):
       name = 'your_app'
       
       def ready(self):
           from risk_analyzer.event_producer import setup_signal_handlers
           setup_signal_handlers()
   ```

3. Use the state change service in your views:
   ```python
   from risk_analyzer.event_producer import change_issue_state
   
   # In your view or service...
   change_issue_state(issue, new_state, user)
   ```

4. Set up RabbitMQ queues and exchanges:
   ```bash
   python manage.py shell
   >>> from risk_analyzer.event_producer import setup_rabbitmq
   >>> setup_rabbitmq()
   ```

5. Start the event consumer:
   ```bash
   python -m risk_analyzer.event_consumer
   ```

See `integration_example.py` for more detailed integration examples.

## API Endpoints

The Risk Analyzer exposes the following API endpoints:

- `/health`: Health check endpoint
- `/api/v1/risk/project/{project_id}`: Project risk analysis
- `/api/v1/risk/issues/{project_id}`: Issue risk analysis
- `/api/v1/risk/team/{project_id}`: Team dynamics analysis
- `/api/v1/risk/workflow/{project_id}`: Workflow optimization
- `/api/v1/search/hybrid`: Hybrid search combining vector and graph
- `/api/v1/embedding`: Generate and manage embeddings

## Configuration

The Risk Analyzer can be configured through environment variables:

- `LOG_LEVEL`: Logging level (default: INFO)
- `RABBITMQ_HOST`, `RABBITMQ_PORT`, etc.: RabbitMQ connection settings
- `NEO4J_URI`, `NEO4J_USER`, etc.: Neo4j connection settings
- `QDRANT_HOST`, `QDRANT_PORT`, etc.: Qdrant connection settings
- `EVENT_EXCHANGE`: RabbitMQ exchange name (default: utrack_events)
- `VECTOR_QUEUE`: Queue for vector updates (default: vector_updates)
- `GRAPH_QUEUE`: Queue for graph updates (default: graph_updates)

## Development

To set up the development environment:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables in `.env`

3. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Run the event consumer (in a separate terminal):
   ```bash
   python -m risk_analyzer.event_consumer
   ``` 