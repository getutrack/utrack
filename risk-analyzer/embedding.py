"""
Vector Embedding Module for Risk Analyzer RAG Implementation

This module handles:
- Text preprocessing and normalization
- Embedding generation using SentenceTransformers
- Efficient batching of documents for vectorization
- Storage of embeddings in Qdrant vector database

It is designed to work with data extracted by the data_extraction module.
"""

import os
import re
import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple, Generator
from datetime import datetime
import time

import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class ModelInitializationError(EmbeddingError):
    """Exception raised for model initialization errors."""
    pass


class EmbeddingGenerationError(EmbeddingError):
    """Exception raised for embedding generation errors."""
    pass


class QdrantConnectionError(EmbeddingError):
    """Exception raised for Qdrant connection errors."""
    pass


class QdrantOperationError(EmbeddingError):
    """Exception raised for Qdrant operation errors."""
    pass


class TextPreprocessor:
    """Class for preprocessing text before generating embeddings."""
    
    def __init__(self, remove_stopwords: bool = True, min_token_length: int = 3):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            min_token_length: Minimum token length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        
        # Initialize NLTK resources
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer")
            nltk.download("punkt", quiet=True)
            
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK stopwords")
            nltk.download("stopwords", quiet=True)
            
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        logger.info("Text preprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for embedding.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces and alphanumerics
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text for embedding (cleaning + stopword removal).
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        if not self.remove_stopwords:
            return cleaned_text
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(cleaned_text)
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) >= self.min_token_length
        ]
        
        # Join tokens back into text
        return " ".join(filtered_tokens)
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


class EmbeddingGenerator:
    """Class for generating embeddings from text using SentenceTransformers."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            preprocessor: TextPreprocessor instance
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.preprocessor = preprocessor or TextPreprocessor()
        self.device = device
        
        # Initialize the model
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise ModelInitializationError(f"Failed to initialize embedding model: {e}")
    
    def generate_embedding(self, text: str, preprocess: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            preprocess: Whether to preprocess the text
            
        Returns:
            Embedding vector as a list of floats
            
        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        try:
            if not text:
                # Return a zero vector for empty text
                return [0.0] * self.embedding_dimension
            
            processed_text = self.preprocessor.preprocess(text) if preprocess else text
            embedding = self.model.encode(processed_text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise EmbeddingGenerationError(f"Failed to generate embedding: {e}")
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        preprocess: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with batching.
        
        Args:
            texts: List of texts to embed
            preprocess: Whether to preprocess the texts
            batch_size: Batch size for processing
            show_progress: Whether to show a progress bar
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        try:
            if not texts:
                return []
            
            processed_texts = (
                self.preprocessor.batch_preprocess(texts) if preprocess else texts
            )
            
            # Filter out empty texts and keep track of indices
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(processed_texts):
                if text:
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            # Generate embeddings for valid texts with batching
            all_embeddings = [None] * len(texts)
            
            # Create iterator with optional progress bar
            if show_progress:
                batches = self._batch_iterator(valid_texts, batch_size, show_progress=True, desc="Generating embeddings")
            else:
                batches = self._batch_iterator(valid_texts, batch_size)
            
            # Process each batch
            for i, batch in enumerate(batches):
                batch_embeddings = self.model.encode(batch, normalize_embeddings=True)
                
                # Assign embeddings to their original positions
                for j, embedding in enumerate(batch_embeddings):
                    original_idx = valid_indices[i * batch_size + j]
                    all_embeddings[original_idx] = embedding.tolist()
            
            # Fill in zero vectors for any empty texts
            zero_vector = [0.0] * self.embedding_dimension
            all_embeddings = [emb if emb is not None else zero_vector for emb in all_embeddings]
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingGenerationError(f"Failed to generate embeddings: {e}")
    
    def _batch_iterator(
        self, 
        items: List[Any], 
        batch_size: int,
        show_progress: bool = False,
        desc: str = "Processing",
    ) -> Generator[List[Any], None, None]:
        """Create batches from a list of items with optional progress bar."""
        num_batches = (len(items) + batch_size - 1) // batch_size
        
        if show_progress:
            with tqdm(total=len(items), desc=desc) as pbar:
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(items))
                    yield items[start_idx:end_idx]
                    pbar.update(end_idx - start_idx)
        else:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(items))
                yield items[start_idx:end_idx]


class QdrantManager:
    """Class for managing Qdrant vector database operations."""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        distance: str = "Cosine",
    ):
        """
        Initialize the Qdrant manager.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_dimension: Dimension of the embeddings
            distance: Distance metric to use
        """
        self.host = os.getenv("QDRANT_HOST", "utrack-qdrant")
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "utrack_vectors")
        self.embedding_dimension = embedding_dimension or int(os.getenv("EMBEDDING_DIMENSION", "384"))
        self.distance = distance
        
        # Initialize Qdrant client
        try:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
            )
            logger.info("Qdrant connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise QdrantConnectionError(f"Failed to connect to Qdrant: {e}")
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create a Qdrant collection for storing embeddings.
        
        Args:
            recreate: Whether to recreate the collection if it exists
            
        Returns:
            True if collection was created or already exists
            
        Raises:
            QdrantOperationError: If collection creation fails
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                if recreate:
                    logger.info(f"Recreating collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection already exists: {self.collection_name}")
                    return True
            
            # Create collection with the specified parameters
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE if self.distance == "Cosine" else Distance.EUCLID,
                ),
            )
            
            # Create payload indexes for efficient filtering
            self._create_indexes()
            
            logger.info(f"Collection created: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise QdrantOperationError(f"Failed to create collection: {e}")
    
    def _create_indexes(self):
        """Create payload indexes for efficient filtering."""
        try:
            indexes = [
                ("metadata.project_id", models.PayloadSchemaType.KEYWORD),
                ("metadata.type", models.PayloadSchemaType.KEYWORD),
                ("metadata.created_at", models.PayloadSchemaType.DATETIME),
                ("metadata.issue_id", models.PayloadSchemaType.KEYWORD),
                ("metadata.author_id", models.PayloadSchemaType.KEYWORD),
            ]
            
            for field_name, field_schema in indexes:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            
            logger.info(f"Created payload indexes for {self.collection_name}")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def store_embeddings(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]],
        batch_size: int = 100,
        show_progress: bool = False,
    ) -> bool:
        """
        Store embeddings in Qdrant.
        
        Args:
            vectors: List of embedding vectors
            ids: List of unique IDs for the vectors
            metadata: List of metadata dictionaries
            batch_size: Batch size for upsert operations
            show_progress: Whether to show a progress bar
            
        Returns:
            True if embeddings were stored successfully
            
        Raises:
            QdrantOperationError: If storing embeddings fails
        """
        if not vectors or not ids or not metadata:
            logger.warning("Empty data provided for storage")
            return False
        
        if not (len(vectors) == len(ids) == len(metadata)):
            raise ValueError("Vectors, IDs, and metadata must have the same length")
        
        try:
            # Create points for upsert
            points = [
                PointStruct(
                    id=id,
                    vector=vector,
                    payload=payload,
                )
                for id, vector, payload in zip(ids, vectors, metadata)
            ]
            
            # Insert in batches
            total_points = len(points)
            total_batches = (total_points + batch_size - 1) // batch_size
            
            if show_progress:
                with tqdm(total=total_points, desc="Storing embeddings") as pbar:
                    for i in range(total_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, total_points)
                        batch = points[start_idx:end_idx]
                        
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=batch,
                        )
                        
                        pbar.update(end_idx - start_idx)
            else:
                for i in range(total_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, total_points)
                    batch = points[start_idx:end_idx]
                    
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                    )
            
            logger.info(f"Stored {total_points} embeddings in {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise QdrantOperationError(f"Failed to store embeddings: {e}")
    
    def search_similar(
        self,
        query_vector: List[float],
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant.
        
        Args:
            query_vector: Embedding vector to search for
            filter_conditions: Dictionary of filter conditions
            limit: Maximum number of results
            
        Returns:
            List of dictionaries containing search results
            
        Raises:
            QdrantOperationError: If search fails
        """
        try:
            # Create filter if conditions provided
            filter_query = None
            if filter_conditions:
                must_conditions = []
                
                for key, value in filter_conditions.items():
                    if isinstance(value, list):
                        # Handle list values (any match)
                        should_conditions = []
                        for v in value:
                            should_conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchValue(value=v),
                                )
                            )
                        must_conditions.append(models.Filter(should=should_conditions))
                    else:
                        # Handle single values
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value),
                            )
                        )
                
                filter_query = models.Filter(must=must_conditions)
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filter_query,
                limit=limit,
                with_payload=True,
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                })
            
            logger.info(f"Found {len(formatted_results)} similar vectors")
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search similar vectors: {e}")
            raise QdrantOperationError(f"Failed to search similar vectors: {e}")


class EmbeddingPipeline:
    """Pipeline for processing data and generating embeddings."""
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        qdrant_manager: Optional[QdrantManager] = None,
    ):
        """
        Initialize the embedding pipeline.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            qdrant_manager: QdrantManager instance
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Use the embedding dimension from the generator for the Qdrant manager
        self.qdrant_manager = qdrant_manager or QdrantManager(
            embedding_dimension=self.embedding_generator.embedding_dimension
        )
        
        # Ensure the collection exists
        self.qdrant_manager.create_collection()
        
        logger.info("Embedding pipeline initialized")
    
    def process_issues(
        self,
        issues: List[Dict[str, Any]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> bool:
        """
        Process issues and store their embeddings.
        
        Args:
            issues: List of issue dictionaries
            batch_size: Batch size for processing
            show_progress: Whether to show a progress bar
            
        Returns:
            True if processing was successful
            
        Raises:
            EmbeddingError: If processing fails
        """
        try:
            if not issues:
                logger.warning("No issues to process")
                return False
            
            # Prepare texts, IDs, and metadata
            texts = []
            ids = []
            metadata = []
            
            for issue in issues:
                # Create combined text from title and description
                title = issue.get("title", "")
                description = issue.get("description", "")
                combined_text = f"Title: {title}\nDescription: {description}"
                
                # Generate a unique ID if not provided
                issue_id = str(issue.get("id", str(uuid.uuid4())))
                vector_id = f"issue_{issue_id}"
                
                # Extract metadata
                issue_metadata = {
                    "type": "issue",
                    "id": issue_id,
                    "project_id": issue.get("project_id"),
                    "title": title,
                    "state": issue.get("state"),
                    "created_at": issue.get("created_at"),
                    "updated_at": issue.get("updated_at"),
                    "creator_id": issue.get("creator_id"),
                    "assignee_id": issue.get("assignee_id"),
                }
                
                texts.append(combined_text)
                ids.append(vector_id)
                metadata.append(issue_metadata)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} issues")
            embeddings = self.embedding_generator.generate_embeddings(
                texts, preprocess=True, batch_size=batch_size, show_progress=show_progress
            )
            
            # Store embeddings
            logger.info(f"Storing {len(embeddings)} issue embeddings in Qdrant")
            self.qdrant_manager.store_embeddings(
                vectors=embeddings,
                ids=ids,
                metadata=metadata,
                batch_size=batch_size,
                show_progress=show_progress,
            )
            
            logger.info(f"Successfully processed {len(issues)} issues")
            return True
        except Exception as e:
            logger.error(f"Failed to process issues: {e}")
            raise EmbeddingError(f"Failed to process issues: {e}")
    
    def process_comments(
        self,
        comments: List[Dict[str, Any]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> bool:
        """
        Process comments and store their embeddings.
        
        Args:
            comments: List of comment dictionaries
            batch_size: Batch size for processing
            show_progress: Whether to show a progress bar
            
        Returns:
            True if processing was successful
            
        Raises:
            EmbeddingError: If processing fails
        """
        try:
            if not comments:
                logger.warning("No comments to process")
                return False
            
            # Prepare texts, IDs, and metadata
            texts = []
            ids = []
            metadata = []
            
            for comment in comments:
                # Get the comment content
                content = comment.get("content", "")
                
                # Generate a unique ID if not provided
                comment_id = str(comment.get("id", str(uuid.uuid4())))
                vector_id = f"comment_{comment_id}"
                
                # Extract metadata
                comment_metadata = {
                    "type": "comment",
                    "id": comment_id,
                    "issue_id": comment.get("issue_id"),
                    "project_id": comment.get("project_id"),
                    "created_at": comment.get("created_at"),
                    "updated_at": comment.get("updated_at"),
                    "author_id": comment.get("author_id"),
                    "issue_title": comment.get("issue_title"),
                }
                
                texts.append(content)
                ids.append(vector_id)
                metadata.append(comment_metadata)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} comments")
            embeddings = self.embedding_generator.generate_embeddings(
                texts, preprocess=True, batch_size=batch_size, show_progress=show_progress
            )
            
            # Store embeddings
            logger.info(f"Storing {len(embeddings)} comment embeddings in Qdrant")
            self.qdrant_manager.store_embeddings(
                vectors=embeddings,
                ids=ids,
                metadata=metadata,
                batch_size=batch_size,
                show_progress=show_progress,
            )
            
            logger.info(f"Successfully processed {len(comments)} comments")
            return True
        except Exception as e:
            logger.error(f"Failed to process comments: {e}")
            raise EmbeddingError(f"Failed to process comments: {e}")
    
    def process_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 16,  # Smaller batch size for potentially larger texts
        show_progress: bool = False,
    ) -> bool:
        """
        Process documents and store their embeddings.
        
        Args:
            documents: List of document dictionaries
            batch_size: Batch size for processing
            show_progress: Whether to show a progress bar
            
        Returns:
            True if processing was successful
            
        Raises:
            EmbeddingError: If processing fails
        """
        try:
            if not documents:
                logger.warning("No documents to process")
                return False
            
            # Prepare texts, IDs, and metadata
            texts = []
            ids = []
            metadata = []
            
            for document in documents:
                # Get the document content
                content = document.get("content", "")
                
                # Skip empty documents
                if not content or len(content.strip()) < 10:
                    logger.warning(f"Skipping document {document.get('name')} with insufficient content")
                    continue
                
                # Extract document info
                doc_name = document.get("name", "")
                doc_id = document.get("id", str(uuid.uuid4()))
                vector_id = f"document_{doc_id}"
                
                # Parse project_id from the document path if available
                project_id = None
                if "projects/" in doc_name:
                    parts = doc_name.split("/")
                    for i, part in enumerate(parts):
                        if part == "projects" and i < len(parts) - 1:
                            project_id = parts[i + 1]
                            break
                
                # Extract metadata
                doc_metadata = {
                    "type": "document",
                    "id": str(doc_id),
                    "name": doc_name,
                    "project_id": project_id,
                    "content_type": document.get("content_type"),
                    "size": document.get("size"),
                    "last_modified": document.get("last_modified"),
                }
                
                # Add additional metadata if present
                if "metadata" in document and isinstance(document["metadata"], dict):
                    doc_metadata.update(document["metadata"])
                
                texts.append(content)
                ids.append(vector_id)
                metadata.append(doc_metadata)
            
            if not texts:
                logger.warning("No valid documents to process after filtering")
                return False
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embedding_generator.generate_embeddings(
                texts, preprocess=True, batch_size=batch_size, show_progress=show_progress
            )
            
            # Store embeddings
            logger.info(f"Storing {len(embeddings)} document embeddings in Qdrant")
            self.qdrant_manager.store_embeddings(
                vectors=embeddings,
                ids=ids,
                metadata=metadata,
                batch_size=batch_size,
                show_progress=show_progress,
            )
            
            logger.info(f"Successfully processed {len(texts)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to process documents: {e}")
            raise EmbeddingError(f"Failed to process documents: {e}")
    
    def process_all_data(
        self,
        data: Dict[str, Any],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Dict[str, bool]:
        """
        Process all data types from a data dictionary.
        
        Args:
            data: Dictionary containing issues, comments, and documents
            batch_size: Batch size for processing
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary with processing status for each data type
        """
        results = {
            "issues": False,
            "comments": False,
            "documents": False,
        }
        
        # Process issues
        if "issues" in data and data["issues"]:
            try:
                results["issues"] = self.process_issues(
                    data["issues"], batch_size=batch_size, show_progress=show_progress
                )
            except Exception as e:
                logger.error(f"Error processing issues: {e}")
        
        # Process comments
        if "comments" in data and data["comments"]:
            try:
                results["comments"] = self.process_comments(
                    data["comments"], batch_size=batch_size, show_progress=show_progress
                )
            except Exception as e:
                logger.error(f"Error processing comments: {e}")
        
        # Process documents
        if "documents" in data and data["documents"]:
            try:
                results["documents"] = self.process_documents(
                    data["documents"], batch_size=batch_size, show_progress=show_progress
                )
            except Exception as e:
                logger.error(f"Error processing documents: {e}")
        
        logger.info(f"Data processing results: {results}")
        return results
    
    def search_similar(
        self,
        query_text: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items to a query text.
        
        Args:
            query_text: Text to search for
            filter_conditions: Dictionary of filter conditions
            limit: Maximum number of results
            
        Returns:
            List of dictionaries containing search results
            
        Raises:
            EmbeddingError: If search fails
        """
        try:
            # Generate embedding for the query text
            query_vector = self.embedding_generator.generate_embedding(query_text, preprocess=True)
            
            # Search for similar vectors
            results = self.qdrant_manager.search_similar(
                query_vector=query_vector,
                filter_conditions=filter_conditions,
                limit=limit,
            )
            
            return results
        except Exception as e:
            logger.error(f"Failed to search similar items: {e}")
            raise EmbeddingError(f"Failed to search similar items: {e}")


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize the embedding pipeline
        pipeline = EmbeddingPipeline()
        
        # Search for similar items
        query = "User authentication issues"
        results = pipeline.search_similar(
            query_text=query,
            filter_conditions={"metadata.type": "issue"},
            limit=5,
        )
        
        print(f"Query: {query}")
        print(f"Found {len(results)} similar items")
        for result in results:
            print(f"Score: {result['score']:.4f}, Title: {result['payload'].get('title')}")
    except Exception as e:
        print(f"Error: {e}") 