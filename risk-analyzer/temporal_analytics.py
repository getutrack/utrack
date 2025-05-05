#!/usr/bin/env python3
"""
Temporal Analytics Module

This module provides temporal analytics capabilities for the Risk Analyzer system.
It implements time-series data capture, temporal collection management in Qdrant,
Neo4j temporal query patterns, and risk evolution analysis over time.
"""

import os
import logging
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import calendar
from enum import Enum

# Database imports
from qdrant_client.models import CollectionStatus

# Local imports
from embedding import EmbeddingGenerator, QdrantManager, TextPreprocessor
from graph_storage import Neo4jManager
from query_engine import QueryEngine

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class TimeGranularity(Enum):
    """Enumeration for time granularity options."""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

class TemporalAnalytics:
    """
    Temporal Analytics Manager
    
    This class provides utilities for managing and analyzing time-series data
    across both vector and graph databases.
    """
    
    def __init__(
        self,
        qdrant_manager: Optional[QdrantManager] = None,
        neo4j_manager: Optional[Neo4jManager] = None,
        query_engine: Optional[QueryEngine] = None,
        base_collection_name: str = "utrack_vectors",
    ):
        """
        Initialize the Temporal Analytics manager.
        
        Args:
            qdrant_manager: Manager for Qdrant operations
            neo4j_manager: Manager for Neo4j operations
            query_engine: Query engine for searches
            base_collection_name: Base name for collections
        """
        # Set up managers
        self.qdrant_manager = qdrant_manager
        if not self.qdrant_manager:
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            self.qdrant_manager = QdrantManager(
                host=host, 
                port=port,
                collection_name=base_collection_name
            )
        
        self.neo4j_manager = neo4j_manager
        if not self.neo4j_manager:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            self.neo4j_manager = Neo4jManager(uri=uri, user=user, password=password)
        
        self.query_engine = query_engine
        if not self.query_engine:
            from embedding import TextPreprocessor, EmbeddingGenerator
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            embedding_generator = EmbeddingGenerator(model_name=model_name)
            self.query_engine = QueryEngine(
                qdrant_manager=self.qdrant_manager,
                neo4j_manager=self.neo4j_manager,
                embedding_generator=embedding_generator,
                text_preprocessor=TextPreprocessor()
            )
        
        self.base_collection_name = base_collection_name
        
        logger.info("Temporal Analytics initialized successfully")
    
    def _generate_period_name(self, timestamp: datetime, granularity: TimeGranularity) -> str:
        """
        Generate a standardized period name based on timestamp and granularity.
        
        Args:
            timestamp: The datetime to generate a period for
            granularity: The granularity level (day, week, month, etc.)
            
        Returns:
            String representation of the period
        """
        if granularity == TimeGranularity.DAY:
            return timestamp.strftime("%Y-%m-%d")
        elif granularity == TimeGranularity.WEEK:
            # Get the week number (1-53)
            week_num = timestamp.isocalendar()[1]
            return f"{timestamp.year}-W{week_num:02d}"
        elif granularity == TimeGranularity.MONTH:
            return timestamp.strftime("%Y-%m")
        elif granularity == TimeGranularity.QUARTER:
            # Calculate quarter (1-4)
            quarter = (timestamp.month - 1) // 3 + 1
            return f"{timestamp.year}-Q{quarter}"
        elif granularity == TimeGranularity.YEAR:
            return str(timestamp.year)
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

    def _generate_collection_name(self, timestamp: datetime, granularity: TimeGranularity) -> str:
        """
        Generate a collection name for a specific time period.
        
        Args:
            timestamp: The datetime to generate a collection name for
            granularity: The granularity level (day, week, month, etc.)
            
        Returns:
            Collection name for the period
        """
        period = self._generate_period_name(timestamp, granularity)
        return f"{self.base_collection_name}_{granularity.value}_{period}"

    def create_temporal_collection(
        self, 
        timestamp: datetime,
        granularity: TimeGranularity,
        vector_size: int = 384,  # Default for all-MiniLM-L6-v2
        overwrite: bool = False
    ) -> str:
        """
        Create a temporal collection in Qdrant for a specific time period.
        
        Args:
            timestamp: The datetime for the collection
            granularity: The granularity level (day, week, month, etc.)
            vector_size: Size of the embedding vectors
            overwrite: Whether to overwrite existing collection
            
        Returns:
            Name of the created collection
        """
        collection_name = self._generate_collection_name(timestamp, granularity)
        
        # Check if collection already exists
        client = self.qdrant_manager.get_client()
        collections_response = client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]
        
        if collection_name in existing_collections:
            if overwrite:
                logger.info(f"Recreating existing collection: {collection_name}")
                client.delete_collection(collection_name)
            else:
                logger.info(f"Collection already exists: {collection_name}")
                return collection_name
        
        # Create the collection
        self.qdrant_manager.create_collection(
            collection_name=collection_name,
            vector_size=vector_size
        )
        
        logger.info(f"Created temporal collection: {collection_name}")
        return collection_name

    def list_temporal_collections(self, granularity: Optional[TimeGranularity] = None) -> List[Dict[str, Any]]:
        """
        List all temporal collections for a specific granularity.
        
        Args:
            granularity: Optional granularity filter
            
        Returns:
            List of collection information dictionaries
        """
        client = self.qdrant_manager.get_client()
        collections_response = client.get_collections()
        
        prefix = f"{self.base_collection_name}_"
        if granularity:
            prefix = f"{self.base_collection_name}_{granularity.value}_"
        
        temporal_collections = []
        for collection in collections_response.collections:
            if collection.name.startswith(prefix):
                # Get collection info
                try:
                    collection_info = client.get_collection(collection.name)
                    vectors_count = client.count(collection.name).count
                    
                    # Extract period from collection name
                    parts = collection.name.split('_')
                    if len(parts) > 2:
                        period_info = '_'.join(parts[2:])
                    else:
                        period_info = "unknown"
                    
                    temporal_collections.append({
                        "name": collection.name,
                        "vectors_count": vectors_count,
                        "status": collection_info.status,
                        "period": period_info,
                        "granularity": parts[1] if len(parts) > 1 else "unknown"
                    })
                except Exception as e:
                    logger.error(f"Error getting info for collection {collection.name}: {e}")
        
        # Sort by name (which will sort by time period)
        temporal_collections.sort(key=lambda x: x["name"])
        return temporal_collections

    def store_embedding_with_timestamp(
        self,
        text: str,
        vector_id: str,
        timestamp: datetime,
        metadata: Dict[str, Any],
        granularities: List[TimeGranularity] = [TimeGranularity.MONTH]
    ) -> Dict[str, bool]:
        """
        Store an embedding in temporal collections based on timestamp.
        
        Args:
            text: The text to embed
            vector_id: Unique ID for the vector
            timestamp: The datetime associated with the vector
            metadata: Additional metadata for the vector
            granularities: List of time granularities to store at
            
        Returns:
            Dictionary of collection names and success status
        """
        # Create embedding
        from embedding import EmbeddingGenerator
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        embedding_generator = EmbeddingGenerator(model_name=model_name)
        
        try:
            embedding = embedding_generator.generate_embedding(text)
            
            # Store in each temporal collection
            results = {}
            for granularity in granularities:
                # Ensure collection exists
                collection_name = self.create_temporal_collection(timestamp, granularity)
                
                # Add timestamp to metadata
                enriched_metadata = metadata.copy()
                enriched_metadata["timestamp"] = timestamp.isoformat()
                enriched_metadata["text"] = text
                
                # Store in collection
                original_collection = self.qdrant_manager.collection_name
                try:
                    self.qdrant_manager.collection_name = collection_name
                    success = self.qdrant_manager.store_embedding(
                        vector_id=vector_id,
                        vector=embedding,
                        metadata=enriched_metadata
                    )
                    results[collection_name] = success
                finally:
                    # Restore original collection
                    self.qdrant_manager.collection_name = original_collection
            
            return results
            
        except Exception as e:
            logger.error(f"Error storing embedding with timestamp: {e}")
            return {}

    def get_period_date_range(
        self, 
        period: str, 
        granularity: TimeGranularity
    ) -> Tuple[datetime, datetime]:
        """
        Get the start and end dates for a specific period.
        
        Args:
            period: Period string (format depends on granularity)
            granularity: Time granularity
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if granularity == TimeGranularity.DAY:
            start_date = datetime.strptime(period, "%Y-%m-%d")
            end_date = start_date + timedelta(days=1) - timedelta(microseconds=1)
        
        elif granularity == TimeGranularity.WEEK:
            # Format: YYYY-WNN (e.g., 2023-W01)
            year, week = period.split('-W')
            # Find the first day of the week
            first_day = datetime.strptime(f"{year}-{week}-1", "%Y-%W-%w")
            start_date = first_day
            end_date = start_date + timedelta(days=7) - timedelta(microseconds=1)
        
        elif granularity == TimeGranularity.MONTH:
            # Format: YYYY-MM
            year, month = period.split('-')
            start_date = datetime(int(year), int(month), 1)
            # Find the last day of the month
            last_day = calendar.monthrange(int(year), int(month))[1]
            end_date = datetime(int(year), int(month), last_day, 23, 59, 59, 999999)
        
        elif granularity == TimeGranularity.QUARTER:
            # Format: YYYY-QN
            year, quarter = period.split('-Q')
            quarter_month = (int(quarter) - 1) * 3 + 1
            start_date = datetime(int(year), quarter_month, 1)
            if quarter_month == 10:  # Q4
                end_date = datetime(int(year), 12, 31, 23, 59, 59, 999999)
            else:
                end_date = datetime(int(year), quarter_month + 3, 1) - timedelta(microseconds=1)
        
        elif granularity == TimeGranularity.YEAR:
            # Format: YYYY
            year = int(period)
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31, 23, 59, 59, 999999)
        
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
        
        return (start_date, end_date)

    def search_temporal_collection(
        self,
        query_text: str,
        timestamp: Optional[datetime] = None,
        granularity: TimeGranularity = TimeGranularity.MONTH,
        period: Optional[str] = None,
        limit: int = 10,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar items in a specific time period.
        
        Args:
            query_text: Query text to search for
            timestamp: Optional timestamp to determine the period
            granularity: Time granularity
            period: Optional explicit period string
            limit: Maximum number of results
            filter_params: Additional filter parameters
            
        Returns:
            Search results
        """
        # Determine collection name
        if timestamp and not period:
            period = self._generate_period_name(timestamp, granularity)
        
        if not period:
            raise ValueError("Either timestamp or period must be provided")
        
        collection_name = f"{self.base_collection_name}_{granularity.value}_{period}"
        
        # Check if collection exists
        client = self.qdrant_manager.get_client()
        collections_response = client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]
        
        if collection_name not in existing_collections:
            logger.warning(f"Collection does not exist: {collection_name}")
            return {
                "results": [],
                "count": 0,
                "period": period,
                "granularity": granularity.value,
                "collection": collection_name,
                "error": "Collection does not exist"
            }
        
        # Use query engine to search the collection
        # Set the collection name temporarily
        original_collection = self.qdrant_manager.collection_name
        try:
            self.qdrant_manager.collection_name = collection_name
            
            # Perform search
            vector_results = self.query_engine._search_vector_db(
                query_text=query_text,
                filter_params=filter_params,
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for item in vector_results:
                result = {
                    "id": item.get("id"),
                    "type": item.get("type", "unknown"),
                    "score": item.get("score", 0),
                    "content": item.get("content", ""),
                    "timestamp": item.get("metadata", {}).get("timestamp"),
                    "metadata": item.get("metadata", {})
                }
                formatted_results.append(result)
            
            return {
                "query": query_text,
                "results": formatted_results,
                "count": len(formatted_results),
                "period": period,
                "granularity": granularity.value,
                "collection": collection_name
            }
        
        finally:
            # Restore original collection
            self.qdrant_manager.collection_name = original_collection

    def temporal_trend_analysis(
        self,
        query_text: str,
        start_period: str,
        end_period: str,
        granularity: TimeGranularity,
        filter_params: Optional[Dict[str, Any]] = None,
        limit_per_period: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze trends over time for a specific query.
        
        Args:
            query_text: Query text to search for
            start_period: Starting period
            end_period: Ending period
            granularity: Time granularity
            filter_params: Additional filter parameters
            limit_per_period: Maximum number of results per period
            
        Returns:
            Trend analysis results
        """
        # Get date ranges
        start_date, _ = self.get_period_date_range(start_period, granularity)
        _, end_date = self.get_period_date_range(end_period, granularity)
        
        # Generate list of all periods between start and end
        periods = []
        current_date = start_date
        
        while current_date <= end_date:
            period = self._generate_period_name(current_date, granularity)
            periods.append(period)
            
            # Move to next period
            if granularity == TimeGranularity.DAY:
                current_date += timedelta(days=1)
            elif granularity == TimeGranularity.WEEK:
                current_date += timedelta(days=7)
            elif granularity == TimeGranularity.MONTH:
                # Move to first day of next month
                if current_date.month == 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(current_date.year, current_date.month + 1, 1)
            elif granularity == TimeGranularity.QUARTER:
                # Move to first day of next quarter
                quarter_month = (current_date.month - 1) // 3 * 3 + 1
                next_quarter_month = quarter_month + 3
                if next_quarter_month > 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(current_date.year, next_quarter_month, 1)
            elif granularity == TimeGranularity.YEAR:
                current_date = datetime(current_date.year + 1, 1, 1)
        
        # Search each period
        period_results = {}
        for period in periods:
            try:
                results = self.search_temporal_collection(
                    query_text=query_text,
                    period=period,
                    granularity=granularity,
                    limit=limit_per_period,
                    filter_params=filter_params
                )
                # Skip periods with no results
                if results.get("count", 0) > 0:
                    period_results[period] = results
            except Exception as e:
                logger.error(f"Error searching period {period}: {e}")
        
        # Calculate trend metrics
        count_trend = {}
        for period, results in period_results.items():
            count_trend[period] = results.get("count", 0)
        
        # Calculate average score per period
        score_trend = {}
        for period, results in period_results.items():
            scores = [item.get("score", 0) for item in results.get("results", [])]
            if scores:
                score_trend[period] = sum(scores) / len(scores)
            else:
                score_trend[period] = 0
        
        return {
            "query": query_text,
            "periods": periods,
            "granularity": granularity.value,
            "count_trend": count_trend,
            "score_trend": score_trend,
            "period_results": period_results,
            "filter_params": filter_params
        }

    def create_neo4j_temporal_indices(self):
        """Create Neo4j indices for efficient temporal queries."""
        # Create indices for timestamp properties
        queries = [
            # Index for Issue created_at timestamp
            """
            CREATE INDEX issue_created_at IF NOT EXISTS
            FOR (i:Issue)
            ON (i.created_at)
            """,
            
            # Index for Comment created_at timestamp
            """
            CREATE INDEX comment_created_at IF NOT EXISTS
            FOR (c:Comment)
            ON (c.created_at)
            """,
            
            # Index for State created_at timestamp
            """
            CREATE INDEX state_created_at IF NOT EXISTS
            FOR (s:State)
            ON (s.created_at)
            """,
            
            # Index for Document created_at timestamp
            """
            CREATE INDEX document_created_at IF NOT EXISTS
            FOR (d:Document)
            ON (d.created_at)
            """
        ]
        
        for query in queries:
            try:
                self.neo4j_manager.query(query, {})
                logger.info(f"Created Neo4j temporal index: {query.strip()}")
            except Exception as e:
                logger.error(f"Error creating Neo4j index: {e}")

    def graph_temporal_query(
        self,
        start_date: datetime,
        end_date: datetime,
        project_id: Optional[str] = None,
        node_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Query Neo4j for nodes within a specific time range.
        
        Args:
            start_date: Start date for the query
            end_date: End date for the query
            project_id: Optional project ID filter
            node_types: Types of nodes to include
            limit: Maximum number of results
            
        Returns:
            Query results
        """
        if not node_types:
            node_types = ["Issue", "Comment", "State", "Document"]
        
        # Convert dates to ISO format for Neo4j query
        start_iso = start_date.isoformat()
        end_iso = end_date.isoformat()
        
        # Build query based on node types
        match_clauses = []
        return_clauses = []
        
        type_mappings = {
            "Issue": "i",
            "Comment": "c",
            "State": "s",
            "Document": "d"
        }
        
        for node_type in node_types:
            if node_type not in type_mappings:
                continue
                
            var = type_mappings[node_type]
            
            # Match clause with temporal and project filters
            project_filter = f"AND {var}.project_id = $project_id" if project_id else ""
            match_clauses.append(f"""
            MATCH ({var}:{node_type})
            WHERE datetime({var}.created_at) >= datetime($start_date)
            AND datetime({var}.created_at) <= datetime($end_date)
            {project_filter}
            """)
            
            # Return clause
            return_clauses.append(f"""
            RETURN {var} AS node, "{node_type.lower()}" AS type, 
            {var}.id AS id, datetime({var}.created_at) AS created_at
            """)
        
        # Combine all queries with UNION
        query = " UNION ".join([m + r for m, r in zip(match_clauses, return_clauses)])
        query += " ORDER BY created_at LIMIT $limit"
        
        params = {
            "start_date": start_iso,
            "end_date": end_iso,
            "limit": limit
        }
        
        if project_id:
            params["project_id"] = project_id
        
        try:
            results = self.neo4j_manager.query(query, params)
            
            # Format results
            formatted_results = []
            for record in results:
                node_data = dict(record["node"])
                formatted_result = {
                    "id": record["id"],
                    "type": record["type"],
                    "created_at": record["created_at"],
                    "properties": node_data
                }
                formatted_results.append(formatted_result)
            
            return {
                "period": {
                    "start": start_iso,
                    "end": end_iso,
                },
                "results": formatted_results,
                "count": len(formatted_results),
                "node_types": node_types,
                "project_id": project_id
            }
            
        except Exception as e:
            logger.error(f"Error in graph temporal query: {e}")
            return {
                "period": {
                    "start": start_iso,
                    "end": end_iso,
                },
                "results": [],
                "count": 0,
                "error": str(e)
            }

    def project_evolution_analysis(
        self,
        project_id: str,
        start_date: datetime,
        end_date: datetime,
        granularity: TimeGranularity = TimeGranularity.MONTH
    ) -> Dict[str, Any]:
        """
        Analyze the evolution of a project over time.
        
        Args:
            project_id: Project ID to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            granularity: Time granularity
            
        Returns:
            Project evolution analysis
        """
        # Generate periods for analysis
        periods = []
        period_date_ranges = []
        
        current_date = start_date
        while current_date <= end_date:
            period = self._generate_period_name(current_date, granularity)
            period_start, period_end = self.get_period_date_range(period, granularity)
            
            # Only include periods that overlap with our date range
            if period_end >= start_date and period_start <= end_date:
                periods.append(period)
                period_date_ranges.append((period_start, period_end))
            
            # Move to next period
            if granularity == TimeGranularity.DAY:
                current_date += timedelta(days=1)
            elif granularity == TimeGranularity.WEEK:
                current_date += timedelta(days=7)
            elif granularity == TimeGranularity.MONTH:
                if current_date.month == 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(current_date.year, current_date.month + 1, 1)
            elif granularity == TimeGranularity.QUARTER:
                quarter_month = (current_date.month - 1) // 3 * 3 + 1
                next_quarter_month = quarter_month + 3
                if next_quarter_month > 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(current_date.year, next_quarter_month, 1)
            elif granularity == TimeGranularity.YEAR:
                current_date = datetime(current_date.year + 1, 1, 1)
        
        # Analyze activity for each period
        period_stats = {}
        for i, period in enumerate(periods):
            period_start, period_end = period_date_ranges[i]
            
            # Get graph data for this period
            graph_data = self.graph_temporal_query(
                start_date=period_start,
                end_date=period_end,
                project_id=project_id,
                node_types=["Issue", "Comment", "State", "Document"],
                limit=1000  # Higher limit for stats calculation
            )
            
            # Count items by type
            type_counts = {}
            for item in graph_data.get("results", []):
                item_type = item.get("type", "unknown")
                if item_type not in type_counts:
                    type_counts[item_type] = 0
                type_counts[item_type] += 1
            
            # Calculate state transitions
            state_transitions = self._calculate_state_transitions(
                graph_data.get("results", []), 
                project_id
            )
            
            # Store stats for this period
            period_stats[period] = {
                "type_counts": type_counts,
                "total_count": len(graph_data.get("results", [])),
                "state_transitions": state_transitions,
                "period": period
            }
        
        # Calculate project evolution metrics
        issue_trend = [period_stats.get(period, {}).get("type_counts", {}).get("issue", 0) 
                       for period in periods]
        
        comment_trend = [period_stats.get(period, {}).get("type_counts", {}).get("comment", 0) 
                        for period in periods]
        
        activity_trend = [period_stats.get(period, {}).get("total_count", 0) 
                         for period in periods]
        
        return {
            "project_id": project_id,
            "period_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "granularity": granularity.value
            },
            "periods": periods,
            "issue_trend": dict(zip(periods, issue_trend)),
            "comment_trend": dict(zip(periods, comment_trend)),
            "activity_trend": dict(zip(periods, activity_trend)),
            "period_stats": period_stats
        }

    def _calculate_state_transitions(
        self,
        temporal_data: List[Dict[str, Any]],
        project_id: str
    ) -> Dict[str, int]:
        """
        Calculate state transitions from temporal data.
        
        Args:
            temporal_data: List of temporal nodes
            project_id: Project ID
            
        Returns:
            Dictionary of state transition counts
        """
        # Filter for state changes
        state_nodes = [item for item in temporal_data if item.get("type") == "state"]
        
        # Get state transition counts with Neo4j
        query = """
        MATCH (i:Issue {project_id: $project_id})-[:HAS_STATE]->(s1:State)
        MATCH (i)-[:HAS_STATE]->(s2:State)
        WHERE s1 <> s2 AND datetime(s1.created_at) < datetime(s2.created_at)
        WITH s1.name as from_state, s2.name as to_state, count(*) as transition_count
        RETURN from_state, to_state, transition_count
        ORDER BY transition_count DESC
        """
        
        try:
            transitions = self.neo4j_manager.query(query, {"project_id": project_id})
            
            # Format results
            transition_counts = {}
            for record in transitions:
                from_state = record["from_state"]
                to_state = record["to_state"]
                count = record["transition_count"]
                transition_key = f"{from_state}_to_{to_state}"
                transition_counts[transition_key] = count
            
            return transition_counts
            
        except Exception as e:
            logger.error(f"Error calculating state transitions: {e}")
            return {}

    def risk_evolution_analysis(
        self,
        project_id: str,
        start_date: datetime,
        end_date: datetime,
        granularity: TimeGranularity = TimeGranularity.MONTH
    ) -> Dict[str, Any]:
        """
        Analyze the evolution of project risks over time.
        
        Args:
            project_id: Project ID to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            granularity: Time granularity
            
        Returns:
            Risk evolution analysis
        """
        # Generate periods for analysis
        periods = []
        
        current_date = start_date
        while current_date <= end_date:
            period = self._generate_period_name(current_date, granularity)
            periods.append(period)
            
            # Move to next period
            if granularity == TimeGranularity.DAY:
                current_date += timedelta(days=1)
            elif granularity == TimeGranularity.WEEK:
                current_date += timedelta(days=7)
            elif granularity == TimeGranularity.MONTH:
                if current_date.month == 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(current_date.year, current_date.month + 1, 1)
            elif granularity == TimeGranularity.QUARTER:
                quarter_month = (current_date.month - 1) // 3 * 3 + 1
                next_quarter_month = quarter_month + 3
                if next_quarter_month > 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(current_date.year, next_quarter_month, 1)
            elif granularity == TimeGranularity.YEAR:
                current_date = datetime(current_date.year + 1, 1, 1)
        
        # Calculate risk factors for each period
        period_risk_factors = {}
        risk_score_trend = {}
        
        risk_categories = ["delayed", "blocked", "resource", "technical"]
        category_trends = {category: {} for category in risk_categories}
        
        for period in periods:
            period_start, period_end = self.get_period_date_range(period, granularity)
            
            # Run risk analysis for this period
            try:
                # Use query engine's risk analysis with date range
                risk_query = "project risks delays blockers issues"
                period_days = (period_end - period_start).days + 1
                
                # Since we're analyzing a specific period, use the date range
                # in the filter rather than query engine's default days
                results = self.query_engine.risk_analysis_search(
                    query_text=risk_query,
                    project_id=project_id,
                    days=period_days
                )
                
                risk_factors = results.get("risk_factors", [])
                
                # Calculate overall risk score for the period
                risk_score = 0.0
                if risk_factors:
                    risk_score = sum(factor["score"] for factor in risk_factors) / len(risk_factors)
                
                risk_score_trend[period] = risk_score
                period_risk_factors[period] = risk_factors
                
                # Update category trends
                for category in risk_categories:
                    category_score = 0.0
                    for factor in risk_factors:
                        if category == "delayed" and factor["name"] == "Delayed Issues":
                            category_score = factor["score"]
                        elif category == "blocked" and factor["name"] == "Blocked Issues":
                            category_score = factor["score"]
                        elif category == "resource" and factor["name"] == "Resource Constraints":
                            category_score = factor["score"]
                        elif category == "technical" and factor["name"] == "Technical Challenges":
                            category_score = factor["score"]
                    
                    category_trends[category][period] = category_score
                
            except Exception as e:
                logger.error(f"Error in risk analysis for period {period}: {e}")
                risk_score_trend[period] = 0.0
                period_risk_factors[period] = []
                for category in risk_categories:
                    category_trends[category][period] = 0.0
        
        # Calculate risk change rates
        risk_change_rates = {}
        for i in range(1, len(periods)):
            prev_period = periods[i-1]
            curr_period = periods[i]
            
            prev_score = risk_score_trend.get(prev_period, 0.0)
            curr_score = risk_score_trend.get(curr_period, 0.0)
            
            # Avoid division by zero
            if prev_score > 0:
                change_rate = (curr_score - prev_score) / prev_score
            else:
                change_rate = 0 if curr_score == 0 else float('inf')
            
            risk_change_rates[curr_period] = change_rate
        
        return {
            "project_id": project_id,
            "period_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "granularity": granularity.value
            },
            "periods": periods,
            "risk_score_trend": risk_score_trend,
            "risk_change_rates": risk_change_rates,
            "risk_category_trends": category_trends,
            "period_risk_factors": period_risk_factors
        }

if __name__ == "__main__":
    # Example usage
    import dotenv
    from datetime import datetime, timedelta
    
    # Load environment variables from .env file if available
    dotenv.load_dotenv()
    
    # Create temporal analytics manager
    ta = TemporalAnalytics()
    
    # Create temporal indices in Neo4j
    ta.create_neo4j_temporal_indices()
    
    # Example: List temporal collections
    collections = ta.list_temporal_collections()
    print(f"Found {len(collections)} temporal collections")
    
    # Example: Store an embedding with timestamp
    now = datetime.now()
    sample_text = "This is a sample text for temporal analytics testing"
    result = ta.store_embedding_with_timestamp(
        text=sample_text,
        vector_id=f"test_{int(time.time())}",
        timestamp=now,
        metadata={"type": "test", "id": "test_1"},
        granularities=[TimeGranularity.DAY, TimeGranularity.MONTH]
    )
    print(f"Stored embedding in collections: {result}")
    
    # Example: Search in temporal collection
    this_month = ta._generate_period_name(now, TimeGranularity.MONTH)
    search_results = ta.search_temporal_collection(
        query_text="sample text testing",
        period=this_month,
        granularity=TimeGranularity.MONTH,
        limit=5
    )
    print(f"Found {search_results.get('count', 0)} results in period {this_month}")
    
    # Example: Project evolution analysis
    start_date = now - timedelta(days=90)
    end_date = now
    project_id = "test_project"  # Replace with actual project ID
    
    evolution = ta.project_evolution_analysis(
        project_id=project_id,
        start_date=start_date,
        end_date=end_date,
        granularity=TimeGranularity.MONTH
    )
    print(f"Analyzed evolution across {len(evolution.get('periods', []))} periods") 