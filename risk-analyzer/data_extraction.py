"""
Data Extraction Pipeline for Risk Analyzer RAG Implementation

This module handles the extraction of data from various sources:
- PostgreSQL database for structured data (issues, projects, comments)
- Minio storage for document files (PDFs, Word documents, images)
- Redis cache for metadata and previously processed data

The extracted data is prepared for further processing by the embedding pipeline.
"""

import logging
import os
import json
import io
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from minio import Minio
from minio.error import S3Error
import PyPDF2
from docx import Document
import pandas as pd
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataExtractionError(Exception):
    """Base exception for data extraction errors."""
    pass


class DatabaseConnectionError(DataExtractionError):
    """Exception raised for database connection errors."""
    pass


class DatabaseQueryError(DataExtractionError):
    """Exception raised for database query errors."""
    pass


class MinioConnectionError(DataExtractionError):
    """Exception raised for Minio connection errors."""
    pass


class MinioDataError(DataExtractionError):
    """Exception raised for Minio data retrieval errors."""
    pass


class RedisConnectionError(DataExtractionError):
    """Exception raised for Redis connection errors."""
    pass


class RedisDataError(DataExtractionError):
    """Exception raised for Redis data retrieval errors."""
    pass


class DocumentParsingError(DataExtractionError):
    """Exception raised for document parsing errors."""
    pass


# PostgreSQL extraction functions
def get_postgres_connection():
    """
    Establish a connection to the PostgreSQL database using environment variables.
    
    Returns:
        psycopg2.connection: Database connection object
        
    Raises:
        DatabaseConnectionError: If connection fails
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "utrack-db"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            dbname=os.getenv("PGDATABASE", "utrack"),
            user=os.getenv("POSTGRES_USER", "utrack"),
            password=os.getenv("POSTGRES_PASSWORD", "utrack"),
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise DatabaseConnectionError(f"Database connection failed: {e}")


def extract_projects(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Extract project data from PostgreSQL.
    
    Args:
        limit: Maximum number of projects to extract
        offset: Number of projects to skip
        
    Returns:
        List of project dictionaries
        
    Raises:
        DatabaseQueryError: If query execution fails
    """
    try:
        conn = get_postgres_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT 
                    p.id, 
                    p.name, 
                    p.description, 
                    p.created_at,
                    p.updated_at,
                    p.is_active,
                    p.owner_id,
                    COUNT(i.id) as issue_count
                FROM 
                    projects p
                LEFT JOIN 
                    issues i ON i.project_id = p.id
                GROUP BY 
                    p.id
                ORDER BY 
                    p.updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            projects = cursor.fetchall()
        conn.close()
        logger.info(f"Extracted {len(projects)} projects from PostgreSQL")
        return projects
    except psycopg2.Error as e:
        logger.error(f"Failed to extract projects: {e}")
        raise DatabaseQueryError(f"Project extraction failed: {e}")


def extract_issues(
    project_id: Optional[str] = None,
    days: int = 30,
    limit: int = 1000,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Extract issue data from PostgreSQL.
    
    Args:
        project_id: If provided, extract issues only for this project
        days: Extract issues updated in the last N days
        limit: Maximum number of issues to extract
        offset: Number of issues to skip
        
    Returns:
        List of issue dictionaries
        
    Raises:
        DatabaseQueryError: If query execution fails
    """
    try:
        conn = get_postgres_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT 
                    i.id, 
                    i.title, 
                    i.description, 
                    i.state,
                    i.created_at,
                    i.updated_at,
                    i.project_id,
                    i.creator_id,
                    i.assignee_id,
                    p.name as project_name
                FROM 
                    issues i
                JOIN 
                    projects p ON i.project_id = p.id
                WHERE 
                    i.updated_at >= NOW() - INTERVAL '%s days'
            """
            params = [days]
            
            if project_id:
                query += " AND i.project_id = %s"
                params.append(project_id)
                
            query += """
                ORDER BY 
                    i.updated_at DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            issues = cursor.fetchall()
        conn.close()
        logger.info(f"Extracted {len(issues)} issues from PostgreSQL")
        return issues
    except psycopg2.Error as e:
        logger.error(f"Failed to extract issues: {e}")
        raise DatabaseQueryError(f"Issue extraction failed: {e}")


def extract_comments(
    issue_id: Optional[str] = None,
    days: int = 30,
    limit: int = 1000,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Extract comment data from PostgreSQL.
    
    Args:
        issue_id: If provided, extract comments only for this issue
        days: Extract comments created in the last N days
        limit: Maximum number of comments to extract
        offset: Number of comments to skip
        
    Returns:
        List of comment dictionaries
        
    Raises:
        DatabaseQueryError: If query execution fails
    """
    try:
        conn = get_postgres_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT 
                    c.id, 
                    c.content, 
                    c.created_at,
                    c.updated_at,
                    c.issue_id,
                    c.author_id,
                    i.title as issue_title,
                    i.project_id
                FROM 
                    comments c
                JOIN 
                    issues i ON c.issue_id = i.id
                WHERE 
                    c.created_at >= NOW() - INTERVAL '%s days'
            """
            params = [days]
            
            if issue_id:
                query += " AND c.issue_id = %s"
                params.append(issue_id)
                
            query += """
                ORDER BY 
                    c.created_at DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            comments = cursor.fetchall()
        conn.close()
        logger.info(f"Extracted {len(comments)} comments from PostgreSQL")
        return comments
    except psycopg2.Error as e:
        logger.error(f"Failed to extract comments: {e}")
        raise DatabaseQueryError(f"Comment extraction failed: {e}")


# Minio extraction functions
def get_minio_client():
    """
    Initialize a Minio client using environment variables.
    
    Returns:
        Minio: Initialized Minio client
        
    Raises:
        MinioConnectionError: If connection initialization fails
    """
    try:
        client = Minio(
            endpoint=os.getenv("AWS_S3_ENDPOINT_URL", "utrack-minio:9000").replace("http://", ""),
            access_key=os.getenv("AWS_ACCESS_KEY_ID", "access-key"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", "secret-key"),
            secure=os.getenv("AWS_S3_ENDPOINT_URL", "").startswith("https://")
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Minio client: {e}")
        raise MinioConnectionError(f"Minio connection failed: {e}")


def extract_document(bucket_name: str, object_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract document content and metadata from Minio.
    
    Args:
        bucket_name: Name of the bucket
        object_name: Name of the object (file path)
        
    Returns:
        Tuple containing:
            - Document text content
            - Dictionary of document metadata
        
    Raises:
        MinioDataError: If data retrieval fails
        DocumentParsingError: If document parsing fails
    """
    try:
        client = get_minio_client()
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME", bucket_name)
        
        # Check if bucket exists
        if not client.bucket_exists(bucket_name):
            raise MinioDataError(f"Bucket {bucket_name} does not exist")
        
        # Get object data and metadata
        response = client.get_object(bucket_name, object_name)
        metadata = client.stat_object(bucket_name, object_name).metadata
        
        # Extract text based on file type
        file_extension = object_name.split(".")[-1].lower()
        content = ""
        
        if file_extension == "pdf":
            content = _extract_pdf_text(response)
        elif file_extension in ["docx", "doc"]:
            content = _extract_docx_text(response)
        elif file_extension in ["jpg", "jpeg", "png"]:
            content = _extract_image_text(response)
        elif file_extension == "csv":
            content = _extract_csv_text(response)
        elif file_extension == "txt":
            content = response.read().decode("utf-8", errors="replace")
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            content = f"Unsupported file type: {file_extension}"
        
        # Close the response to release resources
        response.close()
        response.release_conn()
        
        logger.info(f"Extracted content from {object_name}")
        return content, metadata
    except S3Error as e:
        logger.error(f"Minio error extracting {object_name}: {e}")
        raise MinioDataError(f"Failed to retrieve object: {e}")
    except Exception as e:
        logger.error(f"Error processing document {object_name}: {e}")
        raise DocumentParsingError(f"Document parsing failed: {e}")


def list_documents(
    bucket_name: str,
    prefix: str = "",
    recursive: bool = True
) -> List[Dict[str, Any]]:
    """
    List documents in a Minio bucket.
    
    Args:
        bucket_name: Name of the bucket
        prefix: Object name prefix
        recursive: If True, recursively list objects
        
    Returns:
        List of dictionaries containing document metadata
        
    Raises:
        MinioDataError: If listing fails
    """
    try:
        client = get_minio_client()
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME", bucket_name)
        
        # Check if bucket exists
        if not client.bucket_exists(bucket_name):
            raise MinioDataError(f"Bucket {bucket_name} does not exist")
        
        # List objects
        objects = list(client.list_objects(bucket_name, prefix=prefix, recursive=recursive))
        
        # Get metadata for each object
        document_list = []
        for obj in objects:
            document_list.append({
                "name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified,
                "etag": obj.etag,
                "content_type": client.stat_object(bucket_name, obj.object_name).content_type
            })
        
        logger.info(f"Listed {len(document_list)} documents in {bucket_name}/{prefix}")
        return document_list
    except S3Error as e:
        logger.error(f"Minio error listing documents: {e}")
        raise MinioDataError(f"Failed to list documents: {e}")


def _extract_pdf_text(file_obj: BinaryIO) -> str:
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise DocumentParsingError(f"PDF parsing failed: {e}")


def _extract_docx_text(file_obj: BinaryIO) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = Document(file_obj)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise DocumentParsingError(f"DOCX parsing failed: {e}")


def _extract_image_text(file_obj: BinaryIO) -> str:
    """Extract text from an image using OCR."""
    try:
        image_bytes = file_obj.read()
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error extracting image text: {e}")
        raise DocumentParsingError(f"Image OCR failed: {e}")


def _extract_csv_text(file_obj: BinaryIO) -> str:
    """Extract text from a CSV file."""
    try:
        df = pd.read_csv(file_obj)
        return df.to_string()
    except Exception as e:
        logger.error(f"Error extracting CSV text: {e}")
        raise DocumentParsingError(f"CSV parsing failed: {e}")


# Redis extraction functions
def get_redis_connection():
    """
    Establish a connection to Redis using environment variables.
    
    Returns:
        redis.Redis: Redis connection object
        
    Raises:
        RedisConnectionError: If connection fails
    """
    try:
        conn = redis.Redis(
            host=os.getenv("REDIS_HOST", "utrack-redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=0,
            decode_responses=True  # Automatically decode bytes to strings
        )
        conn.ping()  # Test connection
        return conn
    except redis.RedisError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise RedisConnectionError(f"Redis connection failed: {e}")


def extract_from_redis(key: str) -> Any:
    """
    Extract a value from Redis by key.
    
    Args:
        key: Redis key
        
    Returns:
        Value associated with the key (automatically converted from JSON if possible)
        
    Raises:
        RedisDataError: If data retrieval fails
    """
    try:
        redis_conn = get_redis_connection()
        value = redis_conn.get(key)
        
        if value is None:
            logger.warning(f"No data found for key: {key}")
            return None
        
        # Try to parse as JSON if it looks like JSON
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        logger.info(f"Extracted data for key: {key}")
        return value
    except redis.RedisError as e:
        logger.error(f"Redis error extracting {key}: {e}")
        raise RedisDataError(f"Failed to retrieve data: {e}")


def extract_keys_by_pattern(pattern: str) -> List[str]:
    """
    Extract keys matching a pattern from Redis.
    
    Args:
        pattern: Redis key pattern (e.g., "project:*:metadata")
        
    Returns:
        List of matching keys
        
    Raises:
        RedisDataError: If data retrieval fails
    """
    try:
        redis_conn = get_redis_connection()
        keys = redis_conn.keys(pattern)
        
        logger.info(f"Found {len(keys)} keys matching pattern: {pattern}")
        return keys
    except redis.RedisError as e:
        logger.error(f"Redis error extracting keys by pattern: {e}")
        raise RedisDataError(f"Failed to retrieve keys: {e}")


def extract_hash_from_redis(key: str) -> Dict[str, Any]:
    """
    Extract a hash (dictionary) from Redis.
    
    Args:
        key: Redis key
        
    Returns:
        Dictionary of hash values
        
    Raises:
        RedisDataError: If data retrieval fails
    """
    try:
        redis_conn = get_redis_connection()
        hash_data = redis_conn.hgetall(key)
        
        if not hash_data:
            logger.warning(f"No hash data found for key: {key}")
            return {}
        
        # Try to parse JSON values
        for field, value in hash_data.items():
            if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                try:
                    hash_data[field] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        
        logger.info(f"Extracted hash data for key: {key}")
        return hash_data
    except redis.RedisError as e:
        logger.error(f"Redis error extracting hash {key}: {e}")
        raise RedisDataError(f"Failed to retrieve hash data: {e}")


# Combined data extraction function
def extract_data_for_processing(
    project_id: Optional[str] = None,
    days: int = 30,
    include_documents: bool = True
) -> Dict[str, Any]:
    """
    Extract and combine data from all sources for processing.
    
    Args:
        project_id: If provided, extract data only for this project
        days: Extract data updated in the last N days
        include_documents: Whether to include document content
        
    Returns:
        Dictionary containing structured data from all sources
        
    Raises:
        DataExtractionError: If data extraction fails
    """
    try:
        data = {
            "issues": [],
            "comments": [],
            "documents": [],
            "metadata": {}
        }
        
        # Extract issues
        data["issues"] = extract_issues(project_id=project_id, days=days)
        
        # Extract comments for the issues
        issue_ids = [issue["id"] for issue in data["issues"]]
        for issue_id in issue_ids:
            comments = extract_comments(issue_id=issue_id, days=days)
            data["comments"].extend(comments)
        
        # Extract documents if needed
        if include_documents and project_id:
            bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "uploads")
            document_prefix = f"projects/{project_id}/"
            documents = list_documents(bucket_name, prefix=document_prefix)
            
            # Extract content for each document
            for document in documents:
                try:
                    content, metadata = extract_document(bucket_name, document["name"])
                    document["content"] = content
                    document["metadata"] = metadata
                    data["documents"].append(document)
                except (MinioDataError, DocumentParsingError) as e:
                    logger.warning(f"Skipping document {document['name']}: {e}")
                    continue
        
        # Get metadata from Redis if available
        if project_id:
            try:
                metadata_key = f"project:{project_id}:metadata"
                metadata = extract_hash_from_redis(metadata_key)
                data["metadata"] = metadata
            except RedisDataError as e:
                logger.warning(f"Could not get Redis metadata: {e}")
        
        logger.info(
            f"Extracted {len(data['issues'])} issues, "
            f"{len(data['comments'])} comments, "
            f"{len(data['documents'])} documents"
        )
        return data
    except (DatabaseQueryError, MinioDataError, RedisDataError) as e:
        logger.error(f"Failed to extract data: {e}")
        raise DataExtractionError(f"Data extraction pipeline failed: {e}")


if __name__ == "__main__":
    # Example usage
    try:
        # Extract data for a specific project
        project_id = "example-project-id"  # Replace with actual ID
        data = extract_data_for_processing(project_id=project_id, days=30)
        print(f"Extracted data: {len(data['issues'])} issues, {len(data['comments'])} comments")
    except DataExtractionError as e:
        print(f"Error: {e}") 