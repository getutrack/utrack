# Utrack Project Risk Analysis: RAG Implementation Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Storage Analysis](#data-storage-analysis)
3. [Vector Database Selection](#vector-database-selection)
4. [Data Extraction Pipeline](#data-extraction-pipeline)
5. [Vector Embedding Generation](#vector-embedding-generation)
6. [RAG Implementation](#rag-implementation)
7. [Risk Assessment Models](#risk-assessment-models)
8. [Deployment Strategy](#deployment-strategy)
9. [Evaluation and Monitoring](#evaluation-and-monitoring)
10. [Privacy and Ethical Considerations](#privacy-and-ethical-considerations)

## Architecture Overview

The proposed system will analyze data from Utrack's storage systems, process it through a RAG (Retrieval Augmented Generation) pipeline, and provide risk analysis and recommendations:

```
┌───────────────────┐      ┌─────────────────┐      ┌───────────────────┐
│   Data Sources    │      │  Processing     │      │  Analysis Engine  │
├───────────────────┤      ├─────────────────┤      ├───────────────────┤
│ - PostgreSQL DB   │──────┤ - Extraction    │──────┤ - RAG Module      │
│ - Minio Storage   │      │ - Embedding     │      │ - LLM Integration │
│ - Redis Cache     │      │ - Indexing      │      │ - Risk Analysis   │
└───────────────────┘      └─────────────────┘      └───────────────────┘
                                                             │
                                                             ▼
                                                    ┌───────────────────┐
                                                    │    Outputs        │
                                                    ├───────────────────┤
                                                    │ - Risk Dashboards │
                                                    │ - Alerts          │
                                                    │ - Recommendations │
                                                    └───────────────────┘
```

## Data Storage Analysis

### PostgreSQL Database

Utrack uses PostgreSQL as its primary data store:

1. **Key Tables to Extract**:
   - `issues`: Contains project issues and tasks 
   - `comments`: Communication and discussions
   - `projects`: Project metadata
   - `cycles`: Sprint/iteration data 
   - `users`: Team member information
   - `activity_logs`: Change history and activity

2. **Relationship Data**:
   - Issue relationships (parent-child, dependencies)
   - User assignments and responsibilities
   - Project hierarchies

### Minio Storage

Minio stores file attachments and documents:

1. **Content Types**:
   - Documents (*.pdf, *.docx, etc.)
   - Images that may contain text
   - Exported reports and data

2. **Structure**:
   - Typically organized by workspace/project/issue

### Redis Cache

Redis provides real-time features and caching:

1. **Relevant Data**:
   - Real-time user activity patterns
   - Notification streams
   - Session information

## Vector Database Selection

### Recommended: Qdrant

For this implementation, **Qdrant** is recommended as the vector database for the following reasons:

1. **Performance**: Excellent search performance with large datasets
2. **Filtering**: Advanced filtering capabilities during vector search
3. **Payload Management**: Flexible metadata storage
4. **Clustering**: Supports automatic vector clustering
5. **Self-Hosted Option**: Can be deployed in your infrastructure
6. **API Simplicity**: Clean and intuitive API

### Alternative Options

1. **Pinecone**: 
   - Pros: Fast, fully-managed service
   - Cons: Can be expensive at scale

2. **Weaviate**:
   - Pros: Combined vector and graph capabilities
   - Cons: More complex to set up

3. **pgvector with PostgreSQL**:
   - Pros: Uses existing PostgreSQL infrastructure
   - Cons: Limited performance compared to dedicated vector DBs

4. **Chroma**:
   - Pros: Simple and easy integration
   - Cons: Less mature for large-scale production

## Data Extraction Pipeline

### PostgreSQL Extraction

```python
import psycopg2
import pandas as pd
from datetime import datetime, timedelta

def extract_project_data(workspace_id, project_id, lookback_days=90):
    """Extract relevant project data from PostgreSQL"""
    conn = psycopg2.connect(
        host=os.environ["PGHOST"],
        database=os.environ["PGDATABASE"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"]
    )
    
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    # Extract issues with key metadata
    issues_query = """
        SELECT i.id, i.name, i.description, i.state_id, i.created_at, 
               i.created_by_id, i.assignee_id, i.target_date,
               s.name as state_name, s.group as state_group
        FROM issues i
        JOIN states s ON i.state_id = s.id
        WHERE i.project_id = %s AND i.created_at > %s
    """
    issues_df = pd.read_sql(issues_query, conn, params=[project_id, cutoff_date])
    
    # Extract comments
    comments_query = """
        SELECT c.id, c.issue_id, c.comment, c.created_at, c.created_by_id,
               u.name as author_name
        FROM comments c
        JOIN users u ON c.created_by_id = u.id
        JOIN issues i ON c.issue_id = i.id
        WHERE i.project_id = %s AND c.created_at > %s
    """
    comments_df = pd.read_sql(comments_query, conn, params=[project_id, cutoff_date])
    
    # Extract state transitions (activity logs)
    activity_query = """
        SELECT al.id, al.issue_id, al.field, al.old_value, al.new_value,
               al.created_at, al.created_by_id
        FROM activity_logs al
        JOIN issues i ON al.issue_id = i.id
        WHERE i.project_id = %s AND al.created_at > %s
              AND al.field = 'state'
    """
    activity_df = pd.read_sql(activity_query, conn, params=[project_id, cutoff_date])
    
    # Extract cycle data
    cycles_query = """
        SELECT c.id, c.name, c.description, c.start_date, c.end_date,
               c.status, c.progress
        FROM cycles c
        WHERE c.project_id = %s AND c.end_date > %s
    """
    cycles_df = pd.read_sql(cycles_query, conn, params=[project_id, cutoff_date])
    
    conn.close()
    
    return {
        "issues": issues_df,
        "comments": comments_df,
        "activity": activity_df,
        "cycles": cycles_df
    }
```

### Minio Extraction

```python
import minio
import io
import PyPDF2
import docx
from PIL import Image
import pytesseract

def extract_minio_content(project_id):
    """Extract text content from documents in Minio"""
    client = minio.Minio(
        endpoint=os.environ["AWS_S3_ENDPOINT_URL"].replace("http://", ""),
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        secure=False
    )
    
    bucket = os.environ["AWS_S3_BUCKET_NAME"]
    prefix = f"projects/{project_id}/"
    
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    content_data = []
    
    for obj in objects:
        object_name = obj.object_name
        try:
            response = client.get_object(bucket, object_name)
            file_data = io.BytesIO(response.read())
            
            if object_name.endswith('.pdf'):
                text = extract_text_from_pdf(file_data)
            elif object_name.endswith('.docx'):
                text = extract_text_from_docx(file_data)
            elif object_name.endswith(('.png', '.jpg', '.jpeg')):
                text = extract_text_from_image(file_data)
            else:
                continue
                
            content_data.append({
                "object_name": object_name,
                "content": text,
                "last_modified": obj.last_modified
            })
            
        except Exception as e:
            print(f"Error extracting {object_name}: {e}")
            continue
    
    return content_data

def extract_text_from_pdf(file_data):
    pdf_reader = PyPDF2.PdfReader(file_data)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_data):
    doc = docx.Document(file_data)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(file_data):
    image = Image.open(file_data)
    return pytesseract.image_to_string(image)
```

### Redis Data Extraction

```python
import redis
import json

def extract_redis_activity_data(project_id):
    """Extract recent activity patterns from Redis"""
    r = redis.Redis(
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        db=0
    )
    
    # Get activity patterns (Note: actual keys will depend on Utrack's Redis usage)
    activity_key = f"project:{project_id}:activity"
    notification_key = f"project:{project_id}:notifications"
    
    activity_data = []
    
    # Get activity if exists
    if r.exists(activity_key):
        activity_raw = r.lrange(activity_key, 0, -1)
        for item in activity_raw:
            try:
                activity_data.append(json.loads(item))
            except:
                continue
    
    # Get notification patterns if exists
    notification_data = []
    if r.exists(notification_key):
        notification_raw = r.lrange(notification_key, 0, -1)
        for item in notification_raw:
            try:
                notification_data.append(json.loads(item))
            except:
                continue
    
    return {
        "activity": activity_data,
        "notifications": notification_data
    }
```

## Vector Embedding Generation

### Text Preprocessing

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text for embedding"""
    if not text or not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin
    return " ".join(tokens)

def prepare_documents(data_dict):
    """Prepare documents for embedding"""
    documents = []
    
    # Process issues
    for _, issue in data_dict["issues"].iterrows():
        doc_text = f"Issue: {issue.name}. Description: {issue.description}"
        doc_text = preprocess_text(doc_text)
        if doc_text:
            documents.append({
                "text": doc_text,
                "metadata": {
                    "type": "issue",
                    "id": issue.id,
                    "created_at": issue.created_at,
                    "state": issue.state_name
                }
            })
    
    # Process comments
    for _, comment in data_dict["comments"].iterrows():
        doc_text = preprocess_text(comment.comment)
        if doc_text:
            documents.append({
                "text": doc_text,
                "metadata": {
                    "type": "comment",
                    "id": comment.id,
                    "issue_id": comment.issue_id,
                    "created_at": comment.created_at,
                    "author": comment.author_name
                }
            })
    
    # Process Minio documents
    for doc in data_dict.get("minio_content", []):
        doc_text = preprocess_text(doc["content"])
        if doc_text:
            documents.append({
                "text": doc_text,
                "metadata": {
                    "type": "attachment",
                    "object_name": doc["object_name"],
                    "last_modified": doc["last_modified"]
                }
            })
    
    return documents
```

### Embedding Generation with SentenceTransformers

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def create_embeddings(documents, model_name="all-MiniLM-L6-v2", batch_size=32):
    """Create embeddings for documents using SentenceTransformers"""
    model = SentenceTransformer(model_name)
    
    texts = [doc["text"] for doc in documents]
    embeddings = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings)
    
    # Add embeddings to documents
    for i, embedding in enumerate(embeddings):
        documents[i]["embedding"] = embedding.tolist()
    
    return documents
```

### Qdrant Vector Database Setup

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models

def setup_qdrant(collection_name="utrack_project_data"):
    """Set up Qdrant vector database"""
    client = QdrantClient("localhost", port=6333)
    
    # Check if collection exists, create if not
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    vector_size = 384  # For all-MiniLM-L6-v2
    
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
    
    return client

def store_documents_in_qdrant(client, documents, collection_name="utrack_project_data"):
    """Store documents and embeddings in Qdrant"""
    points = []
    
    for i, doc in enumerate(documents):
        points.append(
            models.PointStruct(
                id=i,
                vector=doc["embedding"],
                payload={
                    "text": doc["text"],
                    **doc["metadata"]
                }
            )
        )
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
    
    return len(points)
```

## RAG Implementation

### Retrieval Component

```python
def retrieve_relevant_documents(client, query, top_k=10, collection_name="utrack_project_data"):
    """Retrieve relevant documents from vector database"""
    # Create embedding for the query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()
    
    # Search for similar documents
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    
    # Format results
    results = []
    for scored_point in search_result:
        results.append({
            "text": scored_point.payload["text"],
            "type": scored_point.payload["type"],
            "score": scored_point.score,
            "metadata": {k: v for k, v in scored_point.payload.items() if k != "text"}
        })
    
    return results
```

### Context Preparation 

```python
def prepare_context(retrieved_docs):
    """Format retrieved documents into context for the LLM"""
    context = ""
    
    for i, doc in enumerate(retrieved_docs):
        doc_type = doc["type"]
        context += f"\n\n[Document {i+1} - {doc_type}]\n"
        
        # Add metadata based on document type
        if doc_type == "issue":
            context += f"Issue ID: {doc['metadata']['id']}\n"
            context += f"State: {doc['metadata']['state']}\n"
            context += f"Created: {doc['metadata']['created_at']}\n"
        elif doc_type == "comment":
            context += f"On Issue: {doc['metadata']['issue_id']}\n"
            context += f"From: {doc['metadata'].get('author', 'Unknown')}\n"
            context += f"Created: {doc['metadata']['created_at']}\n"
        elif doc_type == "attachment":
            context += f"File: {doc['metadata']['object_name']}\n"
            context += f"Modified: {doc['metadata']['last_modified']}\n"
        
        # Add content with clear separator
        context += f"Content: {doc['text']}\n"
        context += "-" * 50
    
    return context
```

## Risk Assessment Models

### LLM Integration with OpenAI

```python
import openai

def analyze_project_risks(retrieved_context, project_metadata):
    """Analyze project risks using LLM"""
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    # Create a prompt that includes project metadata and retrieved context
    prompt = f"""
    As a project risk analyst, evaluate the following project data to identify potential risks and team conflicts.

    PROJECT METADATA:
    - Name: {project_metadata['name']}
    - Started: {project_metadata['created_at']}
    - Status: {project_metadata['status']}
    - Team Size: {project_metadata['team_size']}
    
    HISTORICAL PROJECT DATA:
    {retrieved_context}
    
    Based on this information, please provide:
    
    1. Overall Project Health Score (1-10, with 10 being healthiest)
    2. Top 3 Risk Factors (with evidence from the context)
    3. Team Dynamics Assessment (communication patterns, potential conflicts)
    4. Timeline Assessment (likelihood of meeting deadlines)
    5. Recommended Interventions (specific actions to mitigate risks)
    
    Please format your analysis as JSON with the following structure:
    {
        "health_score": number,
        "risk_factors": [
            {"risk": "string", "evidence": "string", "severity": "HIGH|MEDIUM|LOW"},
            ...
        ],
        "team_dynamics": {
            "assessment": "string",
            "potential_conflicts": ["string", ...],
            "communication_quality": "string"
        },
        "timeline_assessment": {
            "on_track": boolean,
            "delay_risk": "HIGH|MEDIUM|LOW",
            "explanation": "string"
        },
        "recommendations": ["string", ...]
    }
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a project management expert specializing in risk analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    
    analysis = response.choices[0].message.content
    
    # Parse the JSON response
    try:
        import json
        analysis_json = json.loads(analysis)
        return analysis_json
    except:
        # Fallback if JSON parsing fails
        return {"error": "Failed to parse LLM output", "raw_output": analysis}
```

### Risk Trend Analysis

```python
def analyze_risk_trends(project_id, days_back=90, interval_days=7):
    """Analyze how project risks have changed over time"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    intervals = []
    current_date = start_date
    
    risk_scores = []
    
    while current_date <= end_date:
        interval_end = min(current_date + timedelta(days=interval_days), end_date)
        
        # Get data for this interval
        data_dict = extract_project_data(
            workspace_id=None,  # This would need to be filled
            project_id=project_id,
            lookback_days=(end_date - current_date).days
        )
        
        # Get project metadata
        project_metadata = {
            "name": "Project X",  # This would need to be filled
            "created_at": start_date,
            "status": "In Progress",
            "team_size": 5
        }
        
        # Prepare documents and create embeddings
        documents = prepare_documents(data_dict)
        documents = create_embeddings(documents)
        
        # Setup vector DB
        client = setup_qdrant(f"utrack_project_{project_id}_{current_date.strftime('%Y%m%d')}")
        store_documents_in_qdrant(client, documents)
        
        # Retrieve relevant context
        retrieved_docs = retrieve_relevant_documents(
            client, 
            "project risks team conflicts deadline issues",
            top_k=15
        )
        context = prepare_context(retrieved_docs)
        
        # Get risk analysis
        risk_analysis = analyze_project_risks(context, project_metadata)
        
        risk_scores.append({
            "date": current_date,
            "health_score": risk_analysis.get("health_score", 0),
            "risk_factors": risk_analysis.get("risk_factors", []),
            "on_track": risk_analysis.get("timeline_assessment", {}).get("on_track", False)
        })
        
        current_date = interval_end
    
    return risk_scores
```

## Deployment Strategy

### Docker Compose Setup

Create a `docker-compose.yml` file for the RAG system:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT_TELEMETRY_DISABLED=true
    restart: unless-stopped

  risk-analyzer:
    build:
      context: .
      dockerfile: Dockerfile
    depends_on:
      - qdrant
    environment:
      - POSTGRES_HOST=${POSTGRES_HOST}
      - POSTGRES_PORT=${POSTGRES_PORT}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
      - MINIO_ENDPOINT=${MINIO_ENDPOINT}
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
      - MINIO_BUCKET=${MINIO_BUCKET}
      - REDIS_HOST=${REDIS_HOST}
      - REDIS_PORT=${REDIS_PORT}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    restart: unless-stopped

volumes:
  qdrant_data:
```

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install additional dependencies for PDF and image processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

CMD ["python", "risk_analyzer.py"]
```

### Requirements.txt

```
psycopg2-binary==2.9.9
pandas==2.0.3
minio==7.1.15
redis==4.5.5
PyPDF2==3.0.1
python-docx==0.8.11
Pillow==9.5.0
pytesseract==0.3.10
nltk==3.8.1
sentence-transformers==2.2.2
qdrant-client==1.7.0
openai==1.3.5
fastapi==0.103.1
uvicorn==0.23.2
python-dotenv==1.0.0
```

## Evaluation and Monitoring

### Metrics Collection

```python
def collect_system_metrics(analysis_id, project_id, analysis_result):
    """Collect metrics about the RAG system"""
    metrics = {
        "analysis_id": analysis_id,
        "project_id": project_id,
        "timestamp": datetime.now(),
        "health_score": analysis_result.get("health_score"),
        "risk_count": len(analysis_result.get("risk_factors", [])),
        "high_severity_risks": sum(1 for r in analysis_result.get("risk_factors", []) 
                                  if r.get("severity") == "HIGH"),
        "on_track": analysis_result.get("timeline_assessment", {}).get("on_track", False),
        "recommendation_count": len(analysis_result.get("recommendations", [])),
        "processing_time": analysis_result.get("processing_metadata", {}).get("time_taken")
    }
    
    # Store metrics (e.g., in a database or file)
    return metrics

def evaluate_recommendation_quality(recommendations, actual_outcomes, time_window_days=30):
    """Evaluate the quality of recommendations against actual outcomes"""
    # This would be implemented after collecting enough data
    pass
```

## Privacy and Ethical Considerations

### Data Anonymization

```python
import hashlib
import re

def anonymize_sensitive_data(text):
    """Anonymize potentially sensitive information in text"""
    # Anonymize email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
    
    # Anonymize phone numbers
    text = re.sub(r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', '[PHONE]', text)
    
    # Anonymize URLs
    text = re.sub(r'https?://\S+', '[URL]', text)
    
    return text

def hash_user_identifiers(user_id):
    """Hash user identifiers to protect privacy"""
    return hashlib.sha256(str(user_id).encode()).hexdigest()

def apply_privacy_policies(documents):
    """Apply privacy policies to documents before embedding"""
    for doc in documents:
        doc["text"] = anonymize_sensitive_data(doc["text"])
        
        # Anonymize user IDs in metadata
        if "author" in doc["metadata"]:
            doc["metadata"]["author"] = hash_user_identifiers(doc["metadata"]["author"])
        
        if "created_by_id" in doc["metadata"]:
            doc["metadata"]["created_by_id"] = hash_user_identifiers(doc["metadata"]["created_by_id"])
    
    return documents
```

---

This implementation guide provides a comprehensive framework for building a RAG-based risk analysis system for Utrack. The recommended approach using Qdrant as the vector database offers an optimal balance of performance, flexibility, and ease of implementation.

For best results, integrate this system with Utrack's existing notification and reporting mechanisms to provide timely insights and actionable recommendations to project managers and team leads.

Start with a pilot project to validate the approach, refine the risk assessment prompts, and calibrate the sensitivity of the analysis before deploying across all projects. 