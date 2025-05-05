#!/usr/bin/env python3
"""
Hybrid Query Engine

This module implements a hybrid query engine that combines results from
vector search (Qdrant) and graph search (Neo4j) to provide comprehensive
and context-aware search results for risk analysis.
"""

import os
import logging
import time
import functools
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import hashlib

# Redis for caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Local imports
from embedding import TextPreprocessor, EmbeddingGenerator, QdrantManager
from graph_storage import Neo4jManager

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class QueryCache:
    """Cache for query results to improve performance."""
    
    def __init__(self):
        """Initialize the cache."""
        self.use_redis = REDIS_AVAILABLE and os.getenv("USE_REDIS_CACHE", "false").lower() == "true"
        self.enabled = CACHE_ENABLED
        self.ttl = CACHE_TTL
        
        # Initialize cache storage
        if self.use_redis:
            try:
                self.redis_client = redis.from_url(REDIS_URL)
                self.redis_client.ping()  # Test connection
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis, falling back to in-memory cache: {e}")
                self.use_redis = False
                self.cache = {}
                self.cache_times = {}
        else:
            self.cache = {}
            self.cache_times = {}
        
        logger.info(f"Query cache initialized. Enabled: {self.enabled}, Redis: {self.use_redis}")
    
    def _generate_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key based on query and parameters."""
        # Convert parameters to a stable string representation
        params_str = json.dumps(params, sort_keys=True)
        key_data = f"{query}:{params_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result for a query if available and not expired."""
        if not self.enabled:
            return None
        
        key = self._generate_key(query, params)
        
        if self.use_redis:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
            return None
        else:
            # Check if key exists and not expired
            if key in self.cache:
                cache_time = self.cache_times.get(key, 0)
                if time.time() - cache_time <= self.ttl:
                    return self.cache[key]
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    del self.cache_times[key]
            return None
    
    def set(self, query: str, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Store a query result in the cache."""
        if not self.enabled:
            return
        
        key = self._generate_key(query, params)
        
        if self.use_redis:
            self.redis_client.setex(key, self.ttl, json.dumps(result))
        else:
            # If cache is full, remove oldest entry
            if len(self.cache) >= CACHE_MAX_SIZE:
                oldest_key = min(self.cache_times, key=self.cache_times.get)
                del self.cache[oldest_key]
                del self.cache_times[oldest_key]
            
            self.cache[key] = result
            self.cache_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cached entries."""
        if self.use_redis:
            self.redis_client.flushdb()
        else:
            self.cache = {}
            self.cache_times = {}

def cached_query(func):
    """Decorator to cache query results."""
    @functools.wraps(func)
    def wrapper(self, query_text: str, *args, **kwargs):
        # Try to get from cache
        cache_key = query_text
        cache_params = kwargs
        cached_result = self.cache.get(cache_key, cache_params)
        
        if cached_result is not None:
            logger.debug(f"Cache hit for query: {query_text}")
            return cached_result
        
        # Not in cache, execute query
        result = func(self, query_text, *args, **kwargs)
        
        # Store in cache
        self.cache.set(cache_key, cache_params, result)
        
        return result
    return wrapper

class QueryEngine:
    """
    Hybrid query engine that combines vector and graph search results.
    
    This engine provides various query methods for different analysis types,
    combining the semantic understanding from vector search with the
    relationship understanding from graph search.
    """
    
    def __init__(
        self,
        qdrant_manager: Optional[QdrantManager] = None,
        neo4j_manager: Optional[Neo4jManager] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        text_preprocessor: Optional[TextPreprocessor] = None,
    ):
        """
        Initialize the query engine.
        
        Args:
            qdrant_manager: Manager for Qdrant operations
            neo4j_manager: Manager for Neo4j operations
            embedding_generator: Generator for text embeddings
            text_preprocessor: Preprocessor for query text
        """
        # Initialize components if not provided
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        
        self.embedding_generator = embedding_generator
        if not self.embedding_generator:
            # Set up embedding generator with model name from env or default
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.embedding_generator = EmbeddingGenerator(model_name=model_name)
        
        self.qdrant_manager = qdrant_manager
        if not self.qdrant_manager:
            # Connect to Qdrant if not provided
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            collection_name = os.getenv("QDRANT_COLLECTION", "utrack_vectors")
            self.qdrant_manager = QdrantManager(host=host, port=port, collection_name=collection_name)
        
        self.neo4j_manager = neo4j_manager
        if not self.neo4j_manager:
            # Connect to Neo4j if not provided
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            self.neo4j_manager = Neo4jManager(uri=uri, user=user, password=password)
        
        # Initialize cache
        self.cache = QueryCache()
        
        # Default weights for hybrid search
        self.default_vector_weight = float(os.getenv("DEFAULT_VECTOR_WEIGHT", "0.7"))
        self.default_graph_weight = float(os.getenv("DEFAULT_GRAPH_WEIGHT", "0.3"))
        
        logger.info("Query engine initialized successfully")
    
    def _preprocess_query(self, query_text: str) -> str:
        """Preprocess the query text for search."""
        return self.text_preprocessor.clean_text(query_text)
    
    def _generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for the query text."""
        return self.embedding_generator.generate_embedding(query_text)
    
    def _search_vector_db(
        self,
        query_text: str,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar items in Qdrant.
        
        Args:
            query_text: The search query
            filter_params: Optional filter parameters
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        # Preprocess query
        processed_query = self._preprocess_query(query_text)
        
        # Generate embedding
        try:
            query_embedding = self._generate_query_embedding(processed_query)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []
        
        # Search Qdrant
        try:
            results = self.qdrant_manager.search_vectors(
                query_vector=query_embedding,
                limit=limit,
                filter_params=filter_params,
            )
            logger.debug(f"Vector search found {len(results)} results for query: {query_text}")
            return results
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def _search_graph_db(
        self,
        query_text: str,
        project_id: Optional[str] = None,
        node_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant nodes in Neo4j graph database.
        
        Args:
            query_text: The search query
            project_id: Optional project ID to filter
            node_types: Types of nodes to search
            limit: Maximum number of results
            
        Returns:
            List of graph search results
        """
        try:
            # Extract keywords from query for keyword-based search
            keywords = self.text_preprocessor.extract_keywords(query_text)
            
            # Prepare query parameters
            params = {
                "keywords": keywords,
                "limit": limit,
            }
            
            if project_id:
                params["project_id"] = project_id
            
            # Default node types if not specified
            if not node_types:
                node_types = ["Issue", "Comment", "Document"]
            
            # Build Cypher query based on node types
            node_match_clauses = []
            for node_type in node_types:
                if node_type == "Issue":
                    node_match_clauses.append("""
                    MATCH (i:Issue) 
                    WHERE i.title CONTAINS $keywords[0] OR i.description CONTAINS $keywords[0]
                    AND ($project_id IS NULL OR i.project_id = $project_id)
                    RETURN i AS node, 'issue' AS type, i.title AS title, 
                           i.description AS content, i.id AS id,
                           toFloat(5 * count {
                               k IN $keywords WHERE 
                               i.title CONTAINS k OR i.description CONTAINS k
                           } / size($keywords)) AS relevance
                    """)
                elif node_type == "Comment":
                    node_match_clauses.append("""
                    MATCH (c:Comment)
                    WHERE c.content CONTAINS $keywords[0]
                    AND ($project_id IS NULL OR c.project_id = $project_id)
                    RETURN c AS node, 'comment' AS type, '' AS title, 
                           c.content AS content, c.id AS id,
                           toFloat(5 * count {
                               k IN $keywords WHERE c.content CONTAINS k
                           } / size($keywords)) AS relevance
                    """)
                elif node_type == "Document":
                    node_match_clauses.append("""
                    MATCH (d:Document)
                    WHERE d.content CONTAINS $keywords[0]
                    AND ($project_id IS NULL OR d.project_id = $project_id)
                    RETURN d AS node, 'document' AS type, d.name AS title, 
                           d.content AS content, d.id AS id,
                           toFloat(5 * count {
                               k IN $keywords WHERE d.content CONTAINS k
                           } / size($keywords)) AS relevance
                    """)
            
            # Combine all node match clauses with UNION
            query = " UNION ".join(node_match_clauses)
            query += " ORDER BY relevance DESC LIMIT $limit"
            
            # Execute query
            results = self.neo4j_manager.query(query, params)
            
            # Format results
            formatted_results = []
            for record in results:
                node_data = dict(record["node"])
                formatted_result = {
                    "id": record["id"],
                    "type": record["type"],
                    "title": record["title"],
                    "content": record["content"],
                    "score": record["relevance"],
                    "source": "graph",
                    "metadata": node_data,
                }
                formatted_results.append(formatted_result)
            
            logger.debug(f"Graph search found {len(formatted_results)} results for query: {query_text}")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching graph database: {e}")
            return []
    
    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        vector_weight: float = None,
        graph_weight: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Merge and rank results from vector and graph searches.
        
        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
            vector_weight: Weight for vector search scores
            graph_weight: Weight for graph search scores
            
        Returns:
            Combined and ranked list of results
        """
        # Use default weights if not specified
        vector_weight = vector_weight or self.default_vector_weight
        graph_weight = graph_weight or self.default_graph_weight
        
        # Create a dictionary to track unique items by ID and type
        merged_items = {}
        
        # Process vector results
        for item in vector_results:
            key = f"{item.get('type')}_{item.get('id')}"
            vector_score = item.get("score", 0) * vector_weight
            
            merged_items[key] = {
                "id": item.get("id"),
                "type": item.get("type"),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "vector_score": vector_score,
                "graph_score": 0,
                "final_score": vector_score,
                "source": ["vector"],
                "metadata": item.get("metadata", {}),
            }
        
        # Process graph results
        for item in graph_results:
            key = f"{item.get('type')}_{item.get('id')}"
            graph_score = item.get("score", 0) * graph_weight
            
            if key in merged_items:
                # Update existing item
                merged_items[key]["graph_score"] = graph_score
                merged_items[key]["final_score"] = merged_items[key]["vector_score"] + graph_score
                merged_items[key]["source"].append("graph")
                
                # Merge metadata
                for k, v in item.get("metadata", {}).items():
                    if k not in merged_items[key]["metadata"]:
                        merged_items[key]["metadata"][k] = v
            else:
                # Add new item
                merged_items[key] = {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "vector_score": 0,
                    "graph_score": graph_score,
                    "final_score": graph_score,
                    "source": ["graph"],
                    "metadata": item.get("metadata", {}),
                }
        
        # Convert to list and sort by final score
        result_list = list(merged_items.values())
        result_list.sort(key=lambda x: x["final_score"], reverse=True)
        
        return result_list
    
    @cached_query
    def hybrid_search(
        self,
        query_text: str,
        project_id: Optional[str] = None,
        limit: int = 10,
        vector_weight: Optional[float] = None,
        graph_weight: Optional[float] = None,
        use_vector: bool = True,
        use_graph: bool = True,
        node_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a hybrid search using both vector and graph databases.
        
        Args:
            query_text: The search query
            project_id: Optional project ID to filter results
            limit: Maximum number of results
            vector_weight: Weight for vector search scores
            graph_weight: Weight for graph search scores
            use_vector: Whether to include vector search
            use_graph: Whether to include graph search
            node_types: Types of nodes to search in graph
            
        Returns:
            Dictionary containing search results and metadata
        """
        start_time = time.time()
        vector_results = []
        graph_results = []
        
        # Perform vector search if enabled
        if use_vector:
            filter_params = {"project_id": project_id} if project_id else None
            vector_results = self._search_vector_db(
                query_text,
                filter_params=filter_params,
                limit=limit,
            )
        
        # Perform graph search if enabled
        if use_graph:
            graph_results = self._search_graph_db(
                query_text,
                project_id=project_id,
                node_types=node_types,
                limit=limit,
            )
        
        # Merge results
        merged_results = self._merge_results(
            vector_results,
            graph_results,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
        )
        
        # Calculate metrics
        end_time = time.time()
        search_time = end_time - start_time
        
        return {
            "query": query_text,
            "results": merged_results[:limit],  # Limit final results
            "count": len(merged_results[:limit]),
            "total_found": len(merged_results),
            "vector_count": len(vector_results),
            "graph_count": len(graph_results),
            "search_time": search_time,
            "timestamp": datetime.now().isoformat(),
        }
    
    @cached_query
    def risk_analysis_search(
        self,
        query_text: str,
        project_id: str,
        days: int = 30,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Specialized search for risk analysis.
        
        This search puts higher emphasis on recent issues, state changes,
        and specific risk factors.
        
        Args:
            query_text: The risk-related query
            project_id: Project ID to analyze
            days: Number of days to look back
            limit: Maximum number of results
            
        Returns:
            Dictionary containing risk-related search results
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Enhanced query with risk-specific terms
        enhanced_query = f"{query_text} risk issues delays blockers problems"
        
        # Vector search with recency boost
        vector_results = self._search_vector_db(
            enhanced_query,
            filter_params={
                "project_id": project_id,
                "created_at": {"$gte": start_date.isoformat()},
            },
            limit=limit * 2,  # Get more results for filtering
        )
        
        # Graph search with focus on risk-related patterns
        graph_query = """
        MATCH (i:Issue)-[:HAS_STATE]->(s:State)
        WHERE i.project_id = $project_id
        AND s.name IN ['Blocked', 'Failed', 'Delayed']
        AND datetime(i.created_at) >= datetime($start_date)
        AND datetime(i.created_at) <= datetime($end_date)
        RETURN i AS node, 'issue' AS type, i.title AS title,
               i.description AS content, i.id AS id,
               CASE s.name
                 WHEN 'Blocked' THEN 0.9
                 WHEN 'Failed' THEN 0.8
                 WHEN 'Delayed' THEN 0.7
                 ELSE 0.5
               END AS relevance
        UNION
        MATCH (i:Issue)-[:HAS_COMMENT]->(c:Comment)
        WHERE i.project_id = $project_id
        AND any(keyword IN $risk_keywords WHERE c.content CONTAINS keyword)
        AND datetime(c.created_at) >= datetime($start_date)
        AND datetime(c.created_at) <= datetime($end_date)
        RETURN c AS node, 'comment' AS type, '' AS title,
               c.content AS content, c.id AS id,
               0.6 AS relevance
        ORDER BY relevance DESC
        LIMIT $limit
        """
        
        risk_keywords = [
            "risk", "delay", "block", "fail", "issue", "problem", 
            "critical", "urgent", "escalate", "concern"
        ]
        
        graph_params = {
            "project_id": project_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "risk_keywords": risk_keywords,
            "limit": limit,
        }
        
        try:
            graph_results_raw = self.neo4j_manager.query(graph_query, graph_params)
            
            # Format graph results
            graph_results = []
            for record in graph_results_raw:
                node_data = dict(record["node"])
                formatted_result = {
                    "id": record["id"],
                    "type": record["type"],
                    "title": record["title"],
                    "content": record["content"],
                    "score": record["relevance"],
                    "source": "graph",
                    "metadata": node_data,
                }
                graph_results.append(formatted_result)
        except Exception as e:
            logger.error(f"Error in risk analysis graph search: {e}")
            graph_results = []
        
        # Merge with higher weight for graph results
        merged_results = self._merge_results(
            vector_results,
            graph_results,
            vector_weight=0.4,  # Lower weight for vector
            graph_weight=0.6,   # Higher weight for graph
        )
        
        # Extract risk factors from results
        risk_factors = self._extract_risk_factors(merged_results, project_id)
        
        return {
            "query": query_text,
            "project_id": project_id,
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days,
            },
            "results": merged_results[:limit],
            "count": len(merged_results[:limit]),
            "risk_factors": risk_factors,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _extract_risk_factors(
        self,
        results: List[Dict[str, Any]],
        project_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract risk factors from search results.
        
        Args:
            results: Search results containing risk information
            project_id: Project ID
            
        Returns:
            List of identified risk factors
        """
        # Placeholder for actual risk factor extraction logic
        # In a real implementation, this would analyze the results
        # and extract patterns indicating risks
        
        risk_categories = {
            "delayed": {
                "name": "Delayed Issues",
                "count": 0,
                "score": 0.0,
                "examples": [],
            },
            "blocked": {
                "name": "Blocked Issues",
                "count": 0,
                "score": 0.0,
                "examples": [],
            },
            "resource": {
                "name": "Resource Constraints",
                "count": 0,
                "score": 0.0,
                "examples": [],
            },
            "technical": {
                "name": "Technical Challenges",
                "count": 0,
                "score": 0.0,
                "examples": [],
            },
        }
        
        # Process results to identify risk factors
        for item in results:
            content = (item.get("title", "") + " " + item.get("content", "")).lower()
            
            if "delay" in content or "behind schedule" in content:
                risk_categories["delayed"]["count"] += 1
                risk_categories["delayed"]["score"] = min(1.0, risk_categories["delayed"]["score"] + 0.1)
                if len(risk_categories["delayed"]["examples"]) < 3:
                    risk_categories["delayed"]["examples"].append(item)
            
            if "block" in content or "depend" in content or "waiting" in content:
                risk_categories["blocked"]["count"] += 1
                risk_categories["blocked"]["score"] = min(1.0, risk_categories["blocked"]["score"] + 0.1)
                if len(risk_categories["blocked"]["examples"]) < 3:
                    risk_categories["blocked"]["examples"].append(item)
            
            if "resource" in content or "capacity" in content or "overloaded" in content:
                risk_categories["resource"]["count"] += 1
                risk_categories["resource"]["score"] = min(1.0, risk_categories["resource"]["score"] + 0.1)
                if len(risk_categories["resource"]["examples"]) < 3:
                    risk_categories["resource"]["examples"].append(item)
            
            if "complex" in content or "difficult" in content or "challenge" in content:
                risk_categories["technical"]["count"] += 1
                risk_categories["technical"]["score"] = min(1.0, risk_categories["technical"]["score"] + 0.1)
                if len(risk_categories["technical"]["examples"]) < 3:
                    risk_categories["technical"]["examples"].append(item)
        
        # Convert to list of risk factors
        risk_factors = []
        for category, data in risk_categories.items():
            if data["count"] > 0:
                factor = {
                    "name": data["name"],
                    "score": data["score"],
                    "count": data["count"],
                    "examples": [
                        {
                            "id": ex.get("id"),
                            "type": ex.get("type"),
                            "title": ex.get("title", ""),
                            "snippet": ex.get("content", "")[:100] + "...",
                        }
                        for ex in data["examples"]
                    ],
                }
                risk_factors.append(factor)
        
        # Sort by score descending
        risk_factors.sort(key=lambda x: x["score"], reverse=True)
        
        return risk_factors
    
    @cached_query
    def issue_detail_search(
        self,
        issue_id: str,
        query_text: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Search for details related to a specific issue.
        
        Args:
            issue_id: The issue ID to search for
            query_text: Optional additional query text
            limit: Maximum number of results
            
        Returns:
            Dictionary containing issue-related search results
        """
        # Get issue details from Neo4j
        issue_query = """
        MATCH (i:Issue {id: $issue_id})
        RETURN i AS issue
        """
        
        issue_result = self.neo4j_manager.query(issue_query, {"issue_id": issue_id})
        if not issue_result:
            return {
                "error": f"Issue {issue_id} not found",
                "results": [],
                "count": 0,
            }
        
        issue_data = dict(issue_result[0]["issue"])
        
        # Get related comments
        comments_query = """
        MATCH (i:Issue {id: $issue_id})-[:HAS_COMMENT]->(c:Comment)
        RETURN c AS node, 'comment' AS type, '' AS title,
               c.content AS content, c.id AS id,
               datetime(c.created_at) AS created_at
        ORDER BY created_at DESC
        """
        
        comments_result = self.neo4j_manager.query(comments_query, {"issue_id": issue_id})
        comments = [
            {
                "id": record["id"],
                "type": record["type"],
                "content": record["content"],
                "created_at": record["created_at"],
                "metadata": dict(record["node"]),
            }
            for record in comments_result
        ]
        
        # Get state changes
        states_query = """
        MATCH (i:Issue {id: $issue_id})-[:HAS_STATE]->(s:State)
        RETURN s AS node, 'state' AS type, s.name AS title,
               s.description AS content, s.id AS id,
               datetime(s.created_at) AS created_at
        ORDER BY created_at DESC
        """
        
        states_result = self.neo4j_manager.query(states_query, {"issue_id": issue_id})
        states = [
            {
                "id": record["id"],
                "type": record["type"],
                "title": record["title"],
                "content": record["content"],
                "created_at": record["created_at"],
                "metadata": dict(record["node"]),
            }
            for record in states_result
        ]
        
        # If query text is provided, search for similar issues
        similar_issues = []
        if query_text:
            # Create a search query combining issue title and query
            search_text = f"{issue_data.get('title', '')} {query_text}"
            
            # Vector search for similar issues
            vector_results = self._search_vector_db(
                search_text,
                filter_params={"type": "issue", "id": {"$ne": issue_id}},
                limit=limit,
            )
            
            similar_issues = [
                {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0),
                    "metadata": item.get("metadata", {}),
                }
                for item in vector_results
            ]
        
        return {
            "issue_id": issue_id,
            "issue_data": issue_data,
            "comments": comments,
            "states": states,
            "similar_issues": similar_issues,
            "query": query_text,
            "count": {
                "comments": len(comments),
                "states": len(states),
                "similar_issues": len(similar_issues),
            },
            "timestamp": datetime.now().isoformat(),
        }
    
    @cached_query
    def document_search(
        self,
        query_text: str,
        document_types: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Search for documents matching the query.
        
        Args:
            query_text: The search query
            document_types: Types of documents to search
            project_id: Optional project ID filter
            limit: Maximum number of results
            
        Returns:
            Dictionary containing document search results
        """
        filter_params = {"type": "document"}
        
        if project_id:
            filter_params["project_id"] = project_id
        
        if document_types:
            filter_params["document_type"] = {"$in": document_types}
        
        # Vector search for documents
        vector_results = self._search_vector_db(
            query_text,
            filter_params=filter_params,
            limit=limit,
        )
        
        # Format results
        formatted_results = [
            {
                "id": item.get("id"),
                "type": "document",
                "title": item.get("metadata", {}).get("name", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
                "document_type": item.get("metadata", {}).get("document_type", "unknown"),
                "created_at": item.get("metadata", {}).get("created_at", ""),
                "metadata": item.get("metadata", {}),
            }
            for item in vector_results
        ]
        
        return {
            "query": query_text,
            "results": formatted_results,
            "count": len(formatted_results),
            "document_types": document_types,
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
        }

if __name__ == "__main__":
    # Example usage
    import dotenv
    
    # Load environment variables from .env file if available
    dotenv.load_dotenv()
    
    # Create query engine
    query_engine = QueryEngine()
    
    # Example hybrid search
    results = query_engine.hybrid_search(
        query_text="performance issues with database queries",
        limit=5,
    )
    
    print(f"Found {results['count']} results for query: {results['query']}")
    print(f"Search time: {results['search_time']:.2f} seconds")
    
    for i, result in enumerate(results['results']):
        print(f"\nResult {i+1}:")
        print(f"  Type: {result['type']}")
        print(f"  ID: {result['id']}")
        print(f"  Score: {result['final_score']:.4f}")
        print(f"  Source: {', '.join(result['source'])}")
        print(f"  Title: {result['title']}")
        print(f"  Content: {result['content'][:100]}..." if len(result['content']) > 100 else f"  Content: {result['content']}") 