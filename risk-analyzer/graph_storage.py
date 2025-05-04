"""
Neo4j Graph Storage Module for Risk Analyzer RAG Implementation

This module handles:
- Setting up Neo4j database connections
- Creating constraints and indexes for performance
- Storing issues, comments, and other entities as graph nodes
- Creating relationships between entities
- Linking Neo4j nodes with vector embeddings in Qdrant

It is designed to work with the data_extraction and embedding modules.
"""

import os
import logging
import uuid
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone

from neo4j import GraphDatabase, Driver, Session, Result
from neo4j.exceptions import ServiceUnavailable, AuthError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GraphStorageError(Exception):
    """Base exception for graph storage errors."""
    pass


class Neo4jConnectionError(GraphStorageError):
    """Exception raised for Neo4j connection errors."""
    pass


class Neo4jQueryError(GraphStorageError):
    """Exception raised for Neo4j query errors."""
    pass


class Neo4jConstraintError(GraphStorageError):
    """Exception raised for Neo4j constraint errors."""
    pass


class Neo4jManager:
    """Class for managing Neo4j operations."""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initialize the Neo4j manager.
        
        Args:
            uri: Neo4j URI (e.g., bolt://localhost:7687)
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name (default is Neo4j)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://utrack-neo4j:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "utrackneo4j")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = None
        self.initialized = False
        
        logger.info(f"Neo4j manager initialized with URI: {self.uri}")
    
    def connect(self) -> Driver:
        """
        Connect to Neo4j and return the driver.
        
        Returns:
            neo4j.Driver: Neo4j driver instance
            
        Raises:
            Neo4jConnectionError: If connection fails
        """
        if self.driver is not None:
            return self.driver
            
        try:
            logger.info(f"Connecting to Neo4j at {self.uri}")
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 'Connection successful' AS message")
                message = result.single()["message"]
                logger.info(f"Neo4j connection test: {message}")
            
            return self.driver
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise Neo4jConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")
    
    def setup_constraints(self) -> bool:
        """
        Set up Neo4j constraints and indexes for performance.
        
        Returns:
            bool: True if all constraints and indexes were created successfully
            
        Raises:
            Neo4jConstraintError: If constraint creation fails
        """
        try:
            self.connect()  # Ensure connected
            
            with self.driver.session(database=self.database) as session:
                # Create constraints for uniqueness
                constraints = [
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue) REQUIRE i.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:State) REQUIRE s.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (sc:StateChange) REQUIRE sc.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                ]
                
                # Create indexes for faster querying
                indexes = [
                    "CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.name)",
                    "CREATE INDEX IF NOT EXISTS FOR (i:Issue) ON (i.created_at)",
                    "CREATE INDEX IF NOT EXISTS FOR (i:Issue) ON (i.vector_id)",
                    "CREATE INDEX IF NOT EXISTS FOR (c:Comment) ON (c.created_at)",
                    "CREATE INDEX IF NOT EXISTS FOR (c:Comment) ON (c.vector_id)",
                    "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.name)",
                    "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.vector_id)",
                ]
                
                # Execute all constraints
                for constraint in constraints:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
                
                # Execute all indexes
                for index in indexes:
                    session.run(index)
                    logger.info(f"Created index: {index.split('FOR')[1].split('ON')[0].strip()}")
                
            self.initialized = True
            logger.info("Neo4j constraints and indexes set up successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to set up Neo4j constraints and indexes: {e}")
            raise Neo4jConstraintError(f"Failed to set up Neo4j constraints: {e}")
    
    def run_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Result:
        """
        Run a Cypher query and return the result.
        
        Args:
            query: Cypher query
            parameters: Query parameters
            database: Database name (defaults to self.database)
            
        Returns:
            neo4j.Result: Query result
            
        Raises:
            Neo4jQueryError: If query execution fails
        """
        try:
            self.connect()  # Ensure connected
            
            db = database or self.database
            with self.driver.session(database=db) as session:
                result = session.run(query, parameters or {})
                return result
        except Exception as e:
            logger.error(f"Failed to execute Neo4j query: {e}")
            raise Neo4jQueryError(f"Query execution failed: {e}\nQuery: {query}")
    
    def create_project(self, project_data: Dict[str, Any]) -> str:
        """
        Create or update a Project node.
        
        Args:
            project_data: Dictionary containing project data
            
        Returns:
            str: ID of the created/updated project
            
        Raises:
            Neo4jQueryError: If node creation fails
        """
        try:
            # Extract required properties
            project_id = str(project_data.get("id", ""))
            if not project_id:
                project_id = str(uuid.uuid4())
                logger.warning(f"Generated new project ID: {project_id}")
            
            # Ensure datetime is in the correct format
            created_at = project_data.get("created_at")
            updated_at = project_data.get("updated_at")
            
            # Convert datetime objects to ISO format strings
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            if isinstance(updated_at, datetime):
                updated_at = updated_at.isoformat()
            
            # Prepare parameters
            params = {
                "id": project_id,
                "name": project_data.get("name", ""),
                "description": project_data.get("description", ""),
                "created_at": created_at,
                "updated_at": updated_at,
                "is_active": project_data.get("is_active", True),
                "owner_id": project_data.get("owner_id"),
            }
            
            # Create or update project node
            query = """
            MERGE (p:Project {id: $id})
            SET p.name = $name,
                p.description = $description,
                p.created_at = CASE WHEN $created_at IS NOT NULL THEN datetime($created_at) ELSE p.created_at END,
                p.updated_at = CASE WHEN $updated_at IS NOT NULL THEN datetime($updated_at) ELSE p.updated_at END,
                p.is_active = $is_active
            
            WITH p, $owner_id AS owner_id
            WHERE owner_id IS NOT NULL
            
            MERGE (u:User {id: owner_id})
            MERGE (u)-[:OWNS]->(p)
            
            RETURN p.id AS id
            """
            
            result = self.run_query(query, params)
            record = result.single()
            
            if not record:
                raise Neo4jQueryError(f"Failed to create Project node for ID: {project_id}")
            
            logger.info(f"Created/updated Project node with ID: {record['id']}")
            return record["id"]
        except Exception as e:
            logger.error(f"Failed to create Project node: {e}")
            raise Neo4jQueryError(f"Failed to create Project node: {e}")
    
    def create_issue(self, issue_data: Dict[str, Any]) -> str:
        """
        Create or update an Issue node with relationships.
        
        Args:
            issue_data: Dictionary containing issue data
            
        Returns:
            str: ID of the created/updated issue
            
        Raises:
            Neo4jQueryError: If node creation fails
        """
        try:
            # Extract required properties
            issue_id = str(issue_data.get("id", ""))
            if not issue_id:
                issue_id = str(uuid.uuid4())
                logger.warning(f"Generated new issue ID: {issue_id}")
            
            project_id = issue_data.get("project_id")
            if not project_id:
                raise ValueError("Project ID is required for creating an issue")
            
            # Generate vector ID if not provided
            vector_id = issue_data.get("vector_id")
            if not vector_id:
                vector_id = f"issue_{issue_id}"
            
            # Ensure datetime is in the correct format
            created_at = issue_data.get("created_at")
            updated_at = issue_data.get("updated_at")
            
            # Convert datetime objects to ISO format strings
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            if isinstance(updated_at, datetime):
                updated_at = updated_at.isoformat()
            
            # Prepare parameters
            params = {
                "id": issue_id,
                "project_id": project_id,
                "title": issue_data.get("title", ""),
                "description": issue_data.get("description", ""),
                "state": issue_data.get("state", ""),
                "created_at": created_at,
                "updated_at": updated_at,
                "priority": issue_data.get("priority"),
                "vector_id": vector_id,
                "creator_id": issue_data.get("creator_id"),
                "assignee_id": issue_data.get("assignee_id"),
            }
            
            # Create or update issue node with relationships
            query = """
            MATCH (p:Project {id: $project_id})
            
            MERGE (i:Issue {id: $id})
            SET i.title = $title,
                i.description = $description,
                i.state = $state,
                i.created_at = CASE WHEN $created_at IS NOT NULL THEN datetime($created_at) ELSE i.created_at END,
                i.updated_at = CASE WHEN $updated_at IS NOT NULL THEN datetime($updated_at) ELSE i.updated_at END,
                i.priority = $priority,
                i.vector_id = $vector_id
                
            MERGE (i)-[:BELONGS_TO]->(p)
            
            // Create or update current state
            MERGE (s:State {name: $state})
            MERGE (i)-[:HAS_STATE]->(s)
            
            // Handle creator relationship
            WITH i, p, $creator_id AS creator_id
            WHERE creator_id IS NOT NULL
            
            MERGE (creator:User {id: creator_id})
            MERGE (creator)-[:CREATED]->(i)
            
            // Handle assignee relationship
            WITH i, p, $assignee_id AS assignee_id
            WHERE assignee_id IS NOT NULL
            
            MERGE (assignee:User {id: assignee_id})
            MERGE (assignee)-[:ASSIGNED_TO]->(i)
            
            RETURN i.id AS id
            """
            
            result = self.run_query(query, params)
            record = result.single()
            
            if not record:
                raise Neo4jQueryError(f"Failed to create Issue node for ID: {issue_id}")
            
            logger.info(f"Created/updated Issue node with ID: {record['id']}")
            return record["id"]
        except Exception as e:
            logger.error(f"Failed to create Issue node: {e}")
            raise Neo4jQueryError(f"Failed to create Issue node: {e}")
    
    def create_comment(self, comment_data: Dict[str, Any]) -> str:
        """
        Create or update a Comment node with relationships.
        
        Args:
            comment_data: Dictionary containing comment data
            
        Returns:
            str: ID of the created/updated comment
            
        Raises:
            Neo4jQueryError: If node creation fails
        """
        try:
            # Extract required properties
            comment_id = str(comment_data.get("id", ""))
            if not comment_id:
                comment_id = str(uuid.uuid4())
                logger.warning(f"Generated new comment ID: {comment_id}")
            
            issue_id = comment_data.get("issue_id")
            if not issue_id:
                raise ValueError("Issue ID is required for creating a comment")
            
            # Generate vector ID if not provided
            vector_id = comment_data.get("vector_id")
            if not vector_id:
                vector_id = f"comment_{comment_id}"
            
            # Ensure datetime is in the correct format
            created_at = comment_data.get("created_at")
            updated_at = comment_data.get("updated_at")
            
            # Convert datetime objects to ISO format strings
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            if isinstance(updated_at, datetime):
                updated_at = updated_at.isoformat()
            
            # Prepare parameters
            params = {
                "id": comment_id,
                "issue_id": issue_id,
                "content": comment_data.get("content", ""),
                "created_at": created_at,
                "updated_at": updated_at,
                "vector_id": vector_id,
                "author_id": comment_data.get("author_id"),
            }
            
            # Create or update comment node with relationships
            query = """
            MATCH (i:Issue {id: $issue_id})
            
            MERGE (c:Comment {id: $id})
            SET c.content = $content,
                c.created_at = CASE WHEN $created_at IS NOT NULL THEN datetime($created_at) ELSE c.created_at END,
                c.updated_at = CASE WHEN $updated_at IS NOT NULL THEN datetime($updated_at) ELSE c.updated_at END,
                c.vector_id = $vector_id
                
            MERGE (c)-[:COMMENTS_ON]->(i)
            
            // Handle author relationship
            WITH c, i, $author_id AS author_id
            WHERE author_id IS NOT NULL
            
            MERGE (author:User {id: author_id})
            MERGE (author)-[:AUTHORED]->(c)
            
            RETURN c.id AS id
            """
            
            result = self.run_query(query, params)
            record = result.single()
            
            if not record:
                raise Neo4jQueryError(f"Failed to create Comment node for ID: {comment_id}")
            
            logger.info(f"Created/updated Comment node with ID: {record['id']}")
            return record["id"]
        except Exception as e:
            logger.error(f"Failed to create Comment node: {e}")
            raise Neo4jQueryError(f"Failed to create Comment node: {e}")
    
    def create_state_change(self, state_change_data: Dict[str, Any]) -> str:
        """
        Create a StateChange node with relationships.
        
        Args:
            state_change_data: Dictionary containing state change data
            
        Returns:
            str: ID of the created state change
            
        Raises:
            Neo4jQueryError: If node creation fails
        """
        try:
            # Extract required properties
            sc_id = str(state_change_data.get("id", str(uuid.uuid4())))
            
            issue_id = state_change_data.get("issue_id")
            if not issue_id:
                raise ValueError("Issue ID is required for creating a state change")
            
            from_state = state_change_data.get("from_state")
            to_state = state_change_data.get("to_state")
            if not from_state or not to_state:
                raise ValueError("From and To states are required for state change")
            
            # Ensure timestamp is in the correct format
            timestamp = state_change_data.get("timestamp")
            
            # Convert datetime objects to ISO format strings
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            
            # If no timestamp provided, use current time
            if not timestamp:
                timestamp = datetime.now(timezone.utc).isoformat()
            
            # Prepare parameters
            params = {
                "id": sc_id,
                "issue_id": issue_id,
                "from_state": from_state,
                "to_state": to_state,
                "timestamp": timestamp,
                "user_id": state_change_data.get("user_id"),
            }
            
            # Create state change node with relationships
            query = """
            MATCH (i:Issue {id: $issue_id})
            
            MERGE (from_s:State {name: $from_state})
            MERGE (to_s:State {name: $to_state})
            
            CREATE (sc:StateChange {
                id: $id,
                timestamp: CASE WHEN $timestamp IS NOT NULL THEN datetime($timestamp) ELSE datetime() END
            })
            
            CREATE (sc)-[:FROM]->(from_s)
            CREATE (sc)-[:TO]->(to_s)
            CREATE (sc)-[:CHANGES]->(i)
            
            // Update issue state
            SET i.state = $to_state
            
            // Remove old HAS_STATE relationship and create new one
            OPTIONAL MATCH (i)-[old_rel:HAS_STATE]->(:State)
            DELETE old_rel
            MERGE (i)-[:HAS_STATE]->(to_s)
            
            // Handle user relationship if provided
            WITH sc, i, to_s, $user_id AS user_id
            WHERE user_id IS NOT NULL
            
            MERGE (u:User {id: user_id})
            CREATE (u)-[:PERFORMED]->(sc)
            
            RETURN sc.id AS id
            """
            
            result = self.run_query(query, params)
            record = result.single()
            
            if not record:
                raise Neo4jQueryError(f"Failed to create StateChange node for ID: {sc_id}")
            
            logger.info(f"Created StateChange node with ID: {record['id']}")
            return record["id"]
        except Exception as e:
            logger.error(f"Failed to create StateChange node: {e}")
            raise Neo4jQueryError(f"Failed to create StateChange node: {e}")
    
    def create_document(self, document_data: Dict[str, Any]) -> str:
        """
        Create or update a Document node with relationships.
        
        Args:
            document_data: Dictionary containing document data
            
        Returns:
            str: ID of the created/updated document
            
        Raises:
            Neo4jQueryError: If node creation fails
        """
        try:
            # Extract required properties
            doc_id = str(document_data.get("id", ""))
            if not doc_id:
                doc_id = str(uuid.uuid4())
                logger.warning(f"Generated new document ID: {doc_id}")
            
            # Get project_id, either directly or from the name path
            project_id = document_data.get("project_id")
            if not project_id and "name" in document_data:
                name = document_data["name"]
                if "projects/" in name:
                    parts = name.split("/")
                    for i, part in enumerate(parts):
                        if part == "projects" and i < len(parts) - 1:
                            project_id = parts[i + 1]
                            break
            
            # Generate vector ID if not provided
            vector_id = document_data.get("vector_id")
            if not vector_id:
                vector_id = f"document_{doc_id}"
            
            # Get last_modified as a datetime
            last_modified = document_data.get("last_modified")
            
            # Convert datetime objects to ISO format strings
            if isinstance(last_modified, datetime):
                last_modified = last_modified.isoformat()
            
            # Prepare parameters
            params = {
                "id": doc_id,
                "name": document_data.get("name", ""),
                "content_type": document_data.get("content_type", ""),
                "size": document_data.get("size", 0),
                "last_modified": last_modified,
                "vector_id": vector_id,
                "project_id": project_id,
                "uploader_id": document_data.get("uploader_id"),
            }
            
            # Create document node and relationships
            query = """
            MERGE (d:Document {id: $id})
            SET d.name = $name,
                d.content_type = $content_type,
                d.size = $size,
                d.last_modified = CASE WHEN $last_modified IS NOT NULL THEN datetime($last_modified) ELSE d.last_modified END,
                d.vector_id = $vector_id
            
            // Create relationship to project if project_id is provided
            WITH d, $project_id AS project_id
            WHERE project_id IS NOT NULL
            
            MATCH (p:Project {id: project_id})
            MERGE (d)-[:BELONGS_TO]->(p)
            
            // Create relationship to uploader if uploader_id is provided
            WITH d, $uploader_id AS uploader_id
            WHERE uploader_id IS NOT NULL
            
            MERGE (u:User {id: uploader_id})
            MERGE (u)-[:UPLOADED]->(d)
            
            RETURN d.id AS id
            """
            
            result = self.run_query(query, params)
            record = result.single()
            
            if not record:
                raise Neo4jQueryError(f"Failed to create Document node for ID: {doc_id}")
            
            logger.info(f"Created/updated Document node with ID: {record['id']}")
            return record["id"]
        except Exception as e:
            logger.error(f"Failed to create Document node: {e}")
            raise Neo4jQueryError(f"Failed to create Document node: {e}")
    
    def link_issue_references(self, issue_id: str, referenced_issues: List[str]) -> int:
        """
        Create REFERENCES relationships between issues.
        
        Args:
            issue_id: ID of the source issue
            referenced_issues: List of referenced issue IDs
            
        Returns:
            int: Number of relationships created
            
        Raises:
            Neo4jQueryError: If relationship creation fails
        """
        if not referenced_issues:
            return 0
            
        try:
            # Prepare parameters
            params = {
                "issue_id": issue_id,
                "referenced_ids": referenced_issues,
            }
            
            # Create relationships
            query = """
            MATCH (source:Issue {id: $issue_id})
            UNWIND $referenced_ids AS ref_id
            MATCH (target:Issue {id: ref_id})
            WHERE source <> target
            MERGE (source)-[r:REFERENCES]->(target)
            RETURN count(r) AS rel_count
            """
            
            result = self.run_query(query, params)
            record = result.single()
            
            if not record:
                return 0
                
            rel_count = record["rel_count"]
            logger.info(f"Created {rel_count} REFERENCES relationships for issue {issue_id}")
            return rel_count
        except Exception as e:
            logger.error(f"Failed to create issue references: {e}")
            raise Neo4jQueryError(f"Failed to create issue references: {e}")
    
    def set_vector_id(self, node_label: str, node_id: str, vector_id: str) -> bool:
        """
        Set the vector_id property on a node to link it with Qdrant.
        
        Args:
            node_label: Node label (e.g., 'Issue', 'Comment')
            node_id: Node ID
            vector_id: Vector ID in Qdrant
            
        Returns:
            bool: True if successful
            
        Raises:
            Neo4jQueryError: If update fails
        """
        try:
            # Validate inputs
            if not node_label or not node_id or not vector_id:
                raise ValueError("Node label, node ID, and vector ID are required")
                
            # Prepare parameters
            params = {
                "node_id": node_id,
                "vector_id": vector_id,
            }
            
            # Update node with vector ID
            query = f"""
            MATCH (n:{node_label} {{id: $node_id}})
            SET n.vector_id = $vector_id
            RETURN n.id AS id, n.vector_id AS vector_id
            """
            
            result = self.run_query(query, params)
            record = result.single()
            
            if not record:
                logger.warning(f"No {node_label} node found with ID: {node_id}")
                return False
                
            logger.info(f"Set vector_id={vector_id} on {node_label} node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set vector ID: {e}")
            raise Neo4jQueryError(f"Failed to set vector ID: {e}")
    
    def get_items_without_vectors(
        self, 
        node_label: str, 
        limit: int = 100,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get nodes that don't have a vector_id set.
        
        Args:
            node_label: Node label (e.g., 'Issue', 'Comment')
            limit: Maximum number of nodes to return
            days: Only consider nodes from the last N days
            
        Returns:
            List of node dictionaries
            
        Raises:
            Neo4jQueryError: If query fails
        """
        try:
            # Calculate date limit
            date_limit = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Prepare parameters
            params = {
                "date_limit": date_limit,
                "limit": limit,
            }
            
            # Query nodes without vector_id
            query = f"""
            MATCH (n:{node_label})
            WHERE (n.vector_id IS NULL OR n.vector_id = '')
              AND n.created_at >= datetime($date_limit)
            RETURN n
            LIMIT $limit
            """
            
            result = self.run_query(query, params)
            
            nodes = []
            for record in result:
                node = dict(record["n"])
                nodes.append(node)
                
            logger.info(f"Found {len(nodes)} {node_label} nodes without vector IDs")
            return nodes
        except Exception as e:
            logger.error(f"Failed to get items without vectors: {e}")
            raise Neo4jQueryError(f"Failed to get items without vectors: {e}")
    
    def get_related_items(
        self, 
        item_id: str, 
        item_type: str,
        relationship_types: Optional[List[str]] = None,
        target_labels: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get items related to a node through specified relationships.
        
        Args:
            item_id: ID of the source node
            item_type: Label of the source node
            relationship_types: List of relationship types to traverse
            target_labels: List of target node labels to filter
            limit: Maximum number of related items to return
            
        Returns:
            List of related node dictionaries
            
        Raises:
            Neo4jQueryError: If query fails
        """
        try:
            # Default relationship types if not provided
            if not relationship_types:
                relationship_types = ["REFERENCES", "COMMENTS_ON", "BELONGS_TO"]
                
            # Prepare parameters
            params = {
                "item_id": item_id,
                "rel_types": relationship_types,
                "limit": limit,
            }
            
            # Build the query
            target_filter = ""
            if target_labels:
                labels_str = " OR ".join([f"target:{label}" for label in target_labels])
                target_filter = f"WHERE {labels_str}"
            
            query = f"""
            MATCH (source:{item_type} {{id: $item_id}})
            MATCH (source)-[r:{ '|'.join('`' + rt + '`' for rt in relationship_types) }]->(target)
            {target_filter}
            RETURN target, type(r) AS relationship_type
            LIMIT $limit
            """
            
            result = self.run_query(query, params)
            
            related_items = []
            for record in result:
                related_item = dict(record["target"])
                related_item["relationship_type"] = record["relationship_type"]
                # Add the item's labels as a property
                related_item["labels"] = list(record["target"].labels)
                related_items.append(related_item)
                
            logger.info(f"Found {len(related_items)} items related to {item_type} {item_id}")
            return related_items
        except Exception as e:
            logger.error(f"Failed to get related items: {e}")
            raise Neo4jQueryError(f"Failed to get related items: {e}")


class GraphPipeline:
    """Pipeline for processing and storing data in the graph database."""
    
    def __init__(self, neo4j_manager: Optional[Neo4jManager] = None):
        """
        Initialize the graph pipeline.
        
        Args:
            neo4j_manager: Neo4jManager instance
        """
        self.neo4j_manager = neo4j_manager or Neo4jManager()
        
        # Ensure constraints are set up
        try:
            self.neo4j_manager.setup_constraints()
        except Neo4jConstraintError as e:
            logger.warning(f"Could not set up all constraints: {e}")
            
        logger.info("Graph pipeline initialized")
    
    def process_projects(
        self, 
        projects: List[Dict[str, Any]],
        show_progress: bool = False,
    ) -> List[str]:
        """
        Process projects and store them in Neo4j.
        
        Args:
            projects: List of project dictionaries
            show_progress: Whether to show progress
            
        Returns:
            List of created/updated project IDs
        """
        if not projects:
            logger.warning("No projects to process")
            return []
        
        created_ids = []
        
        if show_progress:
            import tqdm
            projects_iter = tqdm.tqdm(projects, desc="Processing projects")
        else:
            projects_iter = projects
        
        for project in projects_iter:
            try:
                project_id = self.neo4j_manager.create_project(project)
                created_ids.append(project_id)
            except Neo4jQueryError as e:
                logger.error(f"Failed to process project {project.get('id')}: {e}")
                # Continue with other projects
        
        logger.info(f"Successfully processed {len(created_ids)} projects")
        return created_ids
    
    def process_issues(
        self, 
        issues: List[Dict[str, Any]],
        show_progress: bool = False,
        process_references: bool = True,
    ) -> List[str]:
        """
        Process issues and store them in Neo4j.
        
        Args:
            issues: List of issue dictionaries
            show_progress: Whether to show progress
            process_references: Whether to process issue references
            
        Returns:
            List of created/updated issue IDs
        """
        if not issues:
            logger.warning("No issues to process")
            return []
        
        created_ids = []
        
        if show_progress:
            import tqdm
            issues_iter = tqdm.tqdm(issues, desc="Processing issues")
        else:
            issues_iter = issues
        
        for issue in issues_iter:
            try:
                # Create issue node
                issue_id = self.neo4j_manager.create_issue(issue)
                created_ids.append(issue_id)
                
                # Process references to other issues if needed
                if process_references and "references" in issue and isinstance(issue["references"], list):
                    self.neo4j_manager.link_issue_references(issue_id, issue["references"])
            except Neo4jQueryError as e:
                logger.error(f"Failed to process issue {issue.get('id')}: {e}")
                # Continue with other issues
        
        logger.info(f"Successfully processed {len(created_ids)} issues")
        return created_ids
    
    def process_comments(
        self, 
        comments: List[Dict[str, Any]],
        show_progress: bool = False,
    ) -> List[str]:
        """
        Process comments and store them in Neo4j.
        
        Args:
            comments: List of comment dictionaries
            show_progress: Whether to show progress
            
        Returns:
            List of created/updated comment IDs
        """
        if not comments:
            logger.warning("No comments to process")
            return []
        
        created_ids = []
        
        if show_progress:
            import tqdm
            comments_iter = tqdm.tqdm(comments, desc="Processing comments")
        else:
            comments_iter = comments
        
        for comment in comments_iter:
            try:
                comment_id = self.neo4j_manager.create_comment(comment)
                created_ids.append(comment_id)
            except Neo4jQueryError as e:
                logger.error(f"Failed to process comment {comment.get('id')}: {e}")
                # Continue with other comments
        
        logger.info(f"Successfully processed {len(created_ids)} comments")
        return created_ids
    
    def process_state_changes(
        self, 
        state_changes: List[Dict[str, Any]],
        show_progress: bool = False,
    ) -> List[str]:
        """
        Process state changes and store them in Neo4j.
        
        Args:
            state_changes: List of state change dictionaries
            show_progress: Whether to show progress
            
        Returns:
            List of created state change IDs
        """
        if not state_changes:
            logger.warning("No state changes to process")
            return []
        
        created_ids = []
        
        if show_progress:
            import tqdm
            changes_iter = tqdm.tqdm(state_changes, desc="Processing state changes")
        else:
            changes_iter = state_changes
        
        for change in changes_iter:
            try:
                change_id = self.neo4j_manager.create_state_change(change)
                created_ids.append(change_id)
            except Neo4jQueryError as e:
                logger.error(f"Failed to process state change {change.get('id')}: {e}")
                # Continue with other state changes
        
        logger.info(f"Successfully processed {len(created_ids)} state changes")
        return created_ids
    
    def process_documents(
        self, 
        documents: List[Dict[str, Any]],
        show_progress: bool = False,
    ) -> List[str]:
        """
        Process documents and store them in Neo4j.
        
        Args:
            documents: List of document dictionaries
            show_progress: Whether to show progress
            
        Returns:
            List of created/updated document IDs
        """
        if not documents:
            logger.warning("No documents to process")
            return []
        
        created_ids = []
        
        if show_progress:
            import tqdm
            docs_iter = tqdm.tqdm(documents, desc="Processing documents")
        else:
            docs_iter = documents
        
        for document in docs_iter:
            try:
                doc_id = self.neo4j_manager.create_document(document)
                created_ids.append(doc_id)
            except Neo4jQueryError as e:
                logger.error(f"Failed to process document {document.get('id')}: {e}")
                # Continue with other documents
        
        logger.info(f"Successfully processed {len(created_ids)} documents")
        return created_ids
    
    def process_all_data(
        self,
        data: Dict[str, Any],
        show_progress: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Process all data types from a data dictionary.
        
        Args:
            data: Dictionary containing projects, issues, comments, etc.
            show_progress: Whether to show progress
            
        Returns:
            Dictionary with lists of created/updated IDs for each data type
        """
        results = {
            "projects": [],
            "issues": [],
            "comments": [],
            "state_changes": [],
            "documents": [],
        }
        
        # Process projects
        if "projects" in data and data["projects"]:
            results["projects"] = self.process_projects(
                data["projects"], 
                show_progress=show_progress
            )
        
        # Process issues
        if "issues" in data and data["issues"]:
            results["issues"] = self.process_issues(
                data["issues"], 
                show_progress=show_progress
            )
        
        # Process comments
        if "comments" in data and data["comments"]:
            results["comments"] = self.process_comments(
                data["comments"], 
                show_progress=show_progress
            )
        
        # Process state changes
        if "state_changes" in data and data["state_changes"]:
            results["state_changes"] = self.process_state_changes(
                data["state_changes"], 
                show_progress=show_progress
            )
        
        # Process documents
        if "documents" in data and data["documents"]:
            results["documents"] = self.process_documents(
                data["documents"], 
                show_progress=show_progress
            )
        
        logger.info(f"Processed data summary: {', '.join([f'{k}: {len(v)}' for k, v in results.items()])}")
        return results
    
    def link_vectors_to_nodes(
        self,
        vector_mappings: Dict[str, Dict[str, str]],
    ) -> Dict[str, int]:
        """
        Link vector IDs with their corresponding Neo4j nodes.
        
        Args:
            vector_mappings: Dictionary mapping node types to {node_id: vector_id} dictionaries
            
        Returns:
            Dictionary with count of successful links by node type
        """
        results = {}
        
        for node_type, mappings in vector_mappings.items():
            success_count = 0
            for node_id, vector_id in mappings.items():
                try:
                    if self.neo4j_manager.set_vector_id(node_type, node_id, vector_id):
                        success_count += 1
                except Neo4jQueryError as e:
                    logger.error(f"Failed to link vector {vector_id} to {node_type} {node_id}: {e}")
            
            results[node_type] = success_count
            logger.info(f"Linked {success_count} vector IDs to {node_type} nodes")
        
        return results
    
    def get_items_requiring_vectors(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get items from Neo4j that need to have vector embeddings generated.
        
        Returns:
            Dictionary mapping node types to lists of nodes
        """
        node_types = ["Issue", "Comment", "Document"]
        results = {}
        
        for node_type in node_types:
            try:
                nodes = self.neo4j_manager.get_items_without_vectors(node_type, limit=100)
                results[node_type] = nodes
            except Neo4jQueryError as e:
                logger.error(f"Failed to get {node_type} nodes needing vectors: {e}")
                results[node_type] = []
        
        return results


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize the graph pipeline
        pipeline = GraphPipeline()
        
        # Create a test project
        test_project = {
            "id": "test-project-1",
            "name": "Test Project",
            "description": "A test project for graph storage",
            "created_at": datetime.now().isoformat(),
        }
        
        # Store in Neo4j
        pipeline.process_projects([test_project])
        
        # Create a test issue
        test_issue = {
            "id": "test-issue-1",
            "project_id": "test-project-1",
            "title": "Test Issue",
            "description": "A test issue for graph storage",
            "state": "Open",
            "created_at": datetime.now().isoformat(),
        }
        
        # Store in Neo4j
        pipeline.process_issues([test_issue])
        
        print("Test data stored successfully")
    except Exception as e:
        print(f"Error: {e}")
