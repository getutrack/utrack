import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)

# Configuration from environment variables (to be loaded from .env)
NEO4J_URI = "bolt://utrack-neo4j:7687"  # Default value, override with environment
NEO4J_USER = "neo4j"  
NEO4J_PASSWORD = "utrackneo4j"

# Global Neo4j driver instance
_driver: Optional[Driver] = None

def init_neo4j() -> Driver:
    """Initialize Neo4j driver and set up constraints/indexes."""
    global _driver
    
    # Create driver
    _driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    
    # Test connection
    with _driver.session() as session:
        result = session.run("RETURN 'Connection successful' AS message")
        message = result.single()["message"]
        logger.info(f"Neo4j connection test: {message}")
    
    # Set up constraints and indexes
    setup_constraints()
    
    logger.info("Neo4j initialized successfully")
    return _driver


def get_neo4j_driver() -> Driver:
    """Get the Neo4j driver instance."""
    if _driver is None:
        raise RuntimeError("Neo4j driver not initialized. Call init_neo4j() first.")
    return _driver


def setup_constraints():
    """Set up Neo4j constraints and indexes for performance."""
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # Create constraints for nodes with unique IDs
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:State) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Risk) REQUIRE r.id IS UNIQUE",
        ]
        
        # Create indexes for improved query performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (i:Issue) ON (i.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR (i:Issue) ON (i.state)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Comment) ON (c.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.name)",
            "CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.name)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Risk) ON (r.severity)",
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.info(f"Created constraint: {constraint}")
            except Exception as e:
                logger.error(f"Error creating constraint: {e}")
        
        for index in indexes:
            try:
                session.run(index)
                logger.info(f"Created index: {index}")
            except Exception as e:
                logger.error(f"Error creating index: {e}")


def create_project_node(project_data: Dict[str, Any]) -> bool:
    """Create or update a Project node in Neo4j."""
    driver = get_neo4j_driver()
    
    query = """
    MERGE (p:Project {id: $id})
    SET 
        p.name = $name,
        p.description = $description,
        p.created_at = datetime($created_at),
        p.updated_at = datetime($updated_at)
    RETURN p.id as id
    """
    
    with driver.session() as session:
        try:
            result = session.run(
                query,
                id=project_data["id"],
                name=project_data.get("name", ""),
                description=project_data.get("description", ""),
                created_at=project_data.get("created_at"),
                updated_at=project_data.get("updated_at"),
            )
            record = result.single()
            logger.info(f"Created/updated Project node with ID: {record['id']}")
            return True
        except Exception as e:
            logger.error(f"Error creating Project node: {e}")
            return False


def create_issue_node(issue_data: Dict[str, Any]) -> bool:
    """Create or update an Issue node in Neo4j with relationships."""
    driver = get_neo4j_driver()
    
    query = """
    MATCH (p:Project {id: $project_id})
    
    MERGE (i:Issue {id: $id})
    SET 
        i.title = $title,
        i.description = $description,
        i.state = $state,
        i.created_at = datetime($created_at),
        i.updated_at = datetime($updated_at),
        i.vector_id = $vector_id
    
    MERGE (i)-[:BELONGS_TO]->(p)
    
    WITH i, $creator_id as creator_id
    WHERE creator_id IS NOT NULL
    MERGE (u:User {id: creator_id})
    MERGE (u)-[:CREATED]->(i)
    
    WITH i, $assignee_id as assignee_id
    WHERE assignee_id IS NOT NULL
    MERGE (a:User {id: assignee_id})
    MERGE (a)-[:ASSIGNED_TO]->(i)
    
    RETURN i.id as id
    """
    
    with driver.session() as session:
        try:
            result = session.run(
                query,
                id=issue_data["id"],
                project_id=issue_data["project_id"],
                title=issue_data.get("title", ""),
                description=issue_data.get("description", ""),
                state=issue_data.get("state", ""),
                created_at=issue_data.get("created_at"),
                updated_at=issue_data.get("updated_at"),
                vector_id=issue_data.get("vector_id", f"issue_{issue_data['id']}"),
                creator_id=issue_data.get("creator_id"),
                assignee_id=issue_data.get("assignee_id"),
            )
            record = result.single()
            logger.info(f"Created/updated Issue node with ID: {record['id']}")
            return True
        except Exception as e:
            logger.error(f"Error creating Issue node: {e}")
            return False


def create_comment_node(comment_data: Dict[str, Any]) -> bool:
    """Create or update a Comment node in Neo4j with relationships."""
    driver = get_neo4j_driver()
    
    query = """
    MATCH (i:Issue {id: $issue_id})
    
    MERGE (c:Comment {id: $id})
    SET 
        c.content = $content,
        c.created_at = datetime($created_at),
        c.updated_at = datetime($updated_at),
        c.vector_id = $vector_id
    
    MERGE (c)-[:COMMENTS_ON]->(i)
    
    WITH c, $author_id as author_id
    WHERE author_id IS NOT NULL
    MERGE (u:User {id: author_id})
    MERGE (u)-[:AUTHORED]->(c)
    
    RETURN c.id as id
    """
    
    with driver.session() as session:
        try:
            result = session.run(
                query,
                id=comment_data["id"],
                issue_id=comment_data["issue_id"],
                content=comment_data.get("content", ""),
                created_at=comment_data.get("created_at"),
                updated_at=comment_data.get("updated_at"),
                vector_id=comment_data.get("vector_id", f"comment_{comment_data['id']}"),
                author_id=comment_data.get("author_id"),
            )
            record = result.single()
            logger.info(f"Created/updated Comment node with ID: {record['id']}")
            return True
        except Exception as e:
            logger.error(f"Error creating Comment node: {e}")
            return False


def create_state_change_node(state_change_data: Dict[str, Any]) -> bool:
    """Create a StateChange node and relationships in Neo4j."""
    driver = get_neo4j_driver()
    
    query = """
    MATCH (i:Issue {id: $issue_id})
    
    MERGE (from:State {name: $from_state})
    MERGE (to:State {name: $to_state})
    
    CREATE (sc:StateChange {
        id: $id,
        timestamp: datetime($timestamp)
    })
    
    CREATE (sc)-[:FROM]->(from)
    CREATE (sc)-[:TO]->(to)
    CREATE (sc)-[:CHANGES]->(i)
    
    WITH sc, $user_id as user_id
    WHERE user_id IS NOT NULL
    MERGE (u:User {id: user_id})
    CREATE (u)-[:PERFORMED]->(sc)
    
    RETURN sc.id as id
    """
    
    with driver.session() as session:
        try:
            result = session.run(
                query,
                id=state_change_data["id"],
                issue_id=state_change_data["issue_id"],
                from_state=state_change_data["from_state"],
                to_state=state_change_data["to_state"],
                timestamp=state_change_data["timestamp"],
                user_id=state_change_data.get("user_id"),
            )
            record = result.single()
            logger.info(f"Created StateChange node with ID: {record['id']}")
            return True
        except Exception as e:
            logger.error(f"Error creating StateChange node: {e}")
            return False


def query_issue_graph(issue_id: str) -> Dict[str, Any]:
    """Query the graph for an issue and its connections."""
    driver = get_neo4j_driver()
    
    query = """
    MATCH (i:Issue {id: $issue_id})
    OPTIONAL MATCH (i)-[:BELONGS_TO]->(p:Project)
    OPTIONAL MATCH (creator:User)-[:CREATED]->(i)
    OPTIONAL MATCH (assignee:User)-[:ASSIGNED_TO]->(i)
    OPTIONAL MATCH (c:Comment)-[:COMMENTS_ON]->(i)
    OPTIONAL MATCH (sc:StateChange)-[:CHANGES]->(i)
    OPTIONAL MATCH (sc)-[:FROM]->(from_state:State)
    OPTIONAL MATCH (sc)-[:TO]->(to_state:State)
    OPTIONAL MATCH (u:User)-[:PERFORMED]->(sc)
    
    RETURN 
        i,
        p,
        creator,
        assignee,
        collect(DISTINCT c) as comments,
        collect(DISTINCT {
            state_change: sc,
            from_state: from_state,
            to_state: to_state,
            performed_by: u
        }) as state_changes
    """
    
    with driver.session() as session:
        try:
            result = session.run(query, issue_id=issue_id)
            record = result.single()
            
            if not record:
                return {}
            
            # Format the result
            issue = dict(record["i"])
            issue["project"] = dict(record["p"]) if record["p"] else None
            issue["creator"] = dict(record["creator"]) if record["creator"] else None
            issue["assignee"] = dict(record["assignee"]) if record["assignee"] else None
            issue["comments"] = [dict(c) for c in record["comments"]]
            
            state_changes = []
            for sc in record["state_changes"]:
                if sc["state_change"]:
                    state_change = dict(sc["state_change"])
                    state_change["from_state"] = dict(sc["from_state"]) if sc["from_state"] else None
                    state_change["to_state"] = dict(sc["to_state"]) if sc["to_state"] else None
                    state_change["performed_by"] = dict(sc["performed_by"]) if sc["performed_by"] else None
                    state_changes.append(state_change)
            
            issue["state_changes"] = state_changes
            
            return issue
        except Exception as e:
            logger.error(f"Error querying issue graph: {e}")
            return {} 