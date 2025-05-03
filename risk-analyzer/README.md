# Utrack Risk Analyzer

A RAG-based risk analysis service for Utrack projects that uses a hybrid vector-graph database approach.

## Features

- **Hybrid Search**: Combines semantic similarity from vector embeddings with relationship data from graph structures
- **Risk Analysis**: Analyzes projects and issues to identify potential risks
- **Team Dynamics Analysis**: Examines team patterns to identify bottlenecks and collaboration opportunities
- **Workflow Optimization**: Suggests improvements based on state transition patterns

## Architecture

- **Neo4j**: Graph database for storing relationships between projects, issues, users, and states
- **Qdrant**: Vector database for storing text embeddings and enabling semantic search
- **FastAPI**: API framework for exposing endpoints
- **RabbitMQ**: Event bus for real-time updates and data synchronization

## API Endpoints

### Search
- `GET /api/v1/search/hybrid` - Hybrid search across vector and graph databases
- `GET /api/v1/search/semantic` - Semantic search using vector embeddings
- `GET /api/v1/search/graph/{issue_id}` - Graph data for a specific issue

### Risk Analysis
- `GET /api/v1/risk/project/{project_id}` - Project-level risk analysis
- `GET /api/v1/risk/issue/{issue_id}` - Issue-level risk analysis

### Team Analysis
- `GET /api/v1/analysis/team-dynamics/{project_id}` - Team dynamics analysis
- `GET /api/v1/analysis/workflow-optimization/{project_id}` - Workflow optimization suggestions

### Embedding
- `POST /api/v1/embedding/generate` - Generate embedding for a single text
- `POST /api/v1/embedding/batch` - Generate embeddings for multiple texts

## Development

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (or use .env file):
```bash
# Neo4j settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Qdrant settings
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

3. Run the service:
```bash
uvicorn main:app --reload
```

### Docker

Build and run with Docker:
```bash
docker build -t risk-analyzer .
docker run -p 8000:8000 risk-analyzer
```

## Integration with Utrack

The Risk Analyzer service integrates with Utrack through:

1. RabbitMQ events for real-time data synchronization
2. API endpoints for risk and team analytics
3. Shared Neo4j and Qdrant instances for data storage 