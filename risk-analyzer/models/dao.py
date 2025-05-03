from typing import List, Dict, Any, Optional
from datetime import datetime

from risk_analyzer.services.neo4j import get_neo4j_driver
from risk_analyzer.core.exceptions import DatabaseError


class ProjectDAO:
    """Data Access Object for Project operations."""

    @staticmethod
    def get_project(project_id: str) -> Dict[str, Any]:
        """Get project details by ID."""
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            try:
                result = session.run("""
                    MATCH (p:Project {id: $project_id})
                    RETURN p.id as id, p.name as name, 
                           p.description as description,
                           p.created_at as created_at
                """, project_id=project_id)
                
                record = result.single()
                if not record:
                    return {}
                
                return dict(record)
            except Exception as e:
                raise DatabaseError(f"Error getting project: {e}")
    
    @staticmethod
    def get_project_metrics(project_id: str) -> Dict[str, Any]:
        """Get project metrics."""
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            try:
                result = session.run("""
                    MATCH (p:Project {id: $project_id})
                    MATCH (i:Issue)-[:BELONGS_TO]->(p)
                    
                    OPTIONAL MATCH (i)-[:HAS_STATE]->(s:State)
                    
                    OPTIONAL MATCH (sc:StateChange)-[:CHANGES]->(i)
                    
                    OPTIONAL MATCH (u:User)-[:ASSIGNED_TO]->(i)
                    
                    RETURN
                        count(DISTINCT i) AS total_issues,
                        collect(DISTINCT s.name) AS states,
                        count(DISTINCT sc) AS state_changes,
                        count(DISTINCT u) AS assigned_users
                """, project_id=project_id)
                
                record = result.single()
                if not record:
                    return {}
                
                return {
                    "total_issues": record["total_issues"],
                    "states": [s for s in record["states"] if s],
                    "state_changes": record["state_changes"],
                    "assigned_users": record["assigned_users"]
                }
            except Exception as e:
                raise DatabaseError(f"Error getting project metrics: {e}")


class IssueDAO:
    """Data Access Object for Issue operations."""

    @staticmethod
    def get_issue(issue_id: str) -> Dict[str, Any]:
        """Get issue details by ID."""
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            try:
                result = session.run("""
                    MATCH (i:Issue {id: $issue_id})
                    OPTIONAL MATCH (i)-[:BELONGS_TO]->(p:Project)
                    OPTIONAL MATCH (creator:User)-[:CREATED]->(i)
                    OPTIONAL MATCH (assignee:User)-[:ASSIGNED_TO]->(i)
                    
                    RETURN
                        i.id as id, 
                        i.title as title,
                        i.description as description,
                        i.state as state,
                        i.created_at as created_at,
                        i.updated_at as updated_at,
                        p.id as project_id,
                        p.name as project_name,
                        creator.id as creator_id,
                        assignee.id as assignee_id
                """, issue_id=issue_id)
                
                record = result.single()
                if not record:
                    return {}
                
                return dict(record)
            except Exception as e:
                raise DatabaseError(f"Error getting issue: {e}")
    
    @staticmethod
    def get_issue_metrics(issue_id: str) -> Dict[str, Any]:
        """Get issue metrics."""
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            try:
                result = session.run("""
                    MATCH (i:Issue {id: $issue_id})
                    
                    OPTIONAL MATCH (sc:StateChange)-[:CHANGES]->(i)
                    
                    OPTIONAL MATCH (c:Comment)-[:COMMENTS_ON]->(i)
                    
                    RETURN
                        count(DISTINCT sc) AS state_changes,
                        count(DISTINCT c) AS comment_count
                """, issue_id=issue_id)
                
                record = result.single()
                if not record:
                    return {}
                
                return {
                    "state_changes": record["state_changes"],
                    "comment_count": record["comment_count"]
                }
            except Exception as e:
                raise DatabaseError(f"Error getting issue metrics: {e}")


class TeamDAO:
    """Data Access Object for Team operations."""

    @staticmethod
    def get_team_members(project_id: str, period_days: int = 30) -> List[Dict[str, Any]]:
        """Get team member data for a project."""
        driver = get_neo4j_driver()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - datetime.timedelta(days=period_days)
        
        with driver.session() as session:
            try:
                result = session.run("""
                    MATCH (p:Project {id: $project_id})
                    MATCH (i:Issue)-[:BELONGS_TO]->(p)
                    MATCH (u:User)-[:ASSIGNED_TO]->(i)
                    
                    OPTIONAL MATCH (performer:User)-[:PERFORMED]->(sc:StateChange)-[:CHANGES]->(i)
                    WHERE sc.timestamp >= datetime($start_date) AND sc.timestamp <= datetime($end_date)
                    
                    OPTIONAL MATCH (author:User)-[:AUTHORED]->(c:Comment)-[:COMMENTS_ON]->(i)
                    WHERE c.created_at >= datetime($start_date) AND c.created_at <= datetime($end_date)
                    
                    RETURN
                        u.id AS user_id,
                        count(DISTINCT i) AS assigned_issues,
                        count(DISTINCT sc) AS state_changes_performed,
                        count(DISTINCT c) AS comments_authored
                """, 
                    project_id=project_id,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat()
                )
                
                team_members = []
                for record in result:
                    team_members.append({
                        "user_id": record["user_id"],
                        "assigned_issues": record["assigned_issues"],
                        "state_changes_performed": record["state_changes_performed"],
                        "comments_authored": record["comments_authored"],
                    })
                
                return team_members
            except Exception as e:
                raise DatabaseError(f"Error getting team members: {e}")


class WorkflowDAO:
    """Data Access Object for Workflow operations."""

    @staticmethod
    def get_state_transitions(project_id: str) -> List[Dict[str, Any]]:
        """Get state transition data for a project."""
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            try:
                result = session.run("""
                    MATCH (p:Project {id: $project_id})
                    MATCH (i:Issue)-[:BELONGS_TO]->(p)
                    MATCH (sc:StateChange)-[:CHANGES]->(i)
                    MATCH (sc)-[:FROM]->(from_state:State)
                    MATCH (sc)-[:TO]->(to_state:State)
                    
                    RETURN
                        from_state.name AS from_state,
                        to_state.name AS to_state,
                        count(*) AS transition_count
                    ORDER BY transition_count DESC
                """, project_id=project_id)
                
                transitions = []
                for record in result:
                    transitions.append({
                        "from_state": record["from_state"],
                        "to_state": record["to_state"],
                        "count": record["transition_count"],
                    })
                
                return transitions
            except Exception as e:
                raise DatabaseError(f"Error getting state transitions: {e}")
    
    @staticmethod
    def get_time_in_states(project_id: str) -> List[Dict[str, Any]]:
        """Get time spent in states for a project."""
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            try:
                result = session.run("""
                    MATCH (p:Project {id: $project_id})
                    MATCH (i:Issue)-[:BELONGS_TO]->(p)
                    MATCH (sc:StateChange)-[:CHANGES]->(i)-[:HAS_STATE]->(current_state:State)
                    MATCH (sc)-[:TO]->(to_state:State)
                    WHERE to_state.name = current_state.name
                    
                    WITH i, sc, to_state, current_state
                    ORDER BY sc.timestamp DESC
                    WITH i, collect(sc)[0] AS latest_transition, to_state, current_state
                    
                    RETURN
                        current_state.name AS state_name,
                        count(i) AS issue_count,
                        avg(duration.between(latest_transition.timestamp, datetime())) AS avg_time_in_state
                    ORDER BY avg_time_in_state DESC
                """, project_id=project_id)
                
                time_in_states = []
                for record in result:
                    time_in_states.append({
                        "state": record["state_name"],
                        "issue_count": record["issue_count"],
                        "avg_days_in_state": record["avg_time_in_state"].days if record["avg_time_in_state"] else 0,
                    })
                
                return time_in_states
            except Exception as e:
                raise DatabaseError(f"Error getting time in states: {e}") 