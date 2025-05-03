from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException, Path
from datetime import datetime, timedelta

from risk_analyzer.services.neo4j import get_neo4j_driver

router = APIRouter()

@router.get("/team-dynamics/{project_id}")
async def get_team_dynamics(
    project_id: str = Path(..., description="Project ID"),
    period_days: int = Query(30, description="Analysis period in days"),
) -> Dict[str, Any]:
    """
    Analyze team dynamics for a project.
    
    This endpoint identifies:
    - Key contributors
    - Collaboration patterns
    - Work distribution
    - Potential bottlenecks
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Query for team dynamics
            result = session.run("""
                MATCH (p:Project {id: $project_id})
                MATCH (i:Issue)-[:BELONGS_TO]->(p)
                MATCH (u:User)-[:ASSIGNED_TO]->(i)
                
                // Get state changes performed by users
                OPTIONAL MATCH (performer:User)-[:PERFORMED]->(sc:StateChange)-[:CHANGES]->(i)
                WHERE sc.timestamp >= datetime($start_date) AND sc.timestamp <= datetime($end_date)
                
                // Get comments authored by users
                OPTIONAL MATCH (author:User)-[:AUTHORED]->(c:Comment)-[:COMMENTS_ON]->(i)
                WHERE c.created_at >= datetime($start_date) AND c.created_at <= datetime($end_date)
                
                RETURN
                    u.id AS user_id,
                    count(DISTINCT i) AS assigned_issues,
                    count(DISTINCT sc) AS state_changes_performed,
                    count(DISTINCT c) AS comments_authored,
                    // Calculate basic workload metrics
                    CASE 
                        WHEN count(DISTINCT i) > 20 THEN 'HIGH'
                        WHEN count(DISTINCT i) > 10 THEN 'MEDIUM'
                        ELSE 'LOW'
                    END AS workload
            """, 
                project_id=project_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )
            
            team_members = []
            total_assigned_issues = 0
            total_state_changes = 0
            total_comments = 0
            
            for record in result:
                team_members.append({
                    "user_id": record["user_id"],
                    "assigned_issues": record["assigned_issues"],
                    "state_changes_performed": record["state_changes_performed"],
                    "comments_authored": record["comments_authored"],
                    "workload": record["workload"],
                })
                total_assigned_issues += record["assigned_issues"]
                total_state_changes += record["state_changes_performed"]
                total_comments += record["comments_authored"]
            
            # Calculate bottleneck score
            high_workload_count = sum(1 for member in team_members if member["workload"] == "HIGH")
            bottleneck_score = high_workload_count / len(team_members) if team_members else 0
            bottleneck_level = "HIGH" if bottleneck_score > 0.3 else "MEDIUM" if bottleneck_score > 0.1 else "LOW"
            
            # Calculate work distribution
            if team_members:
                avg_issues_per_member = total_assigned_issues / len(team_members)
                max_issues = max(member["assigned_issues"] for member in team_members) if team_members else 0
                distribution_score = max_issues / avg_issues_per_member if avg_issues_per_member > 0 else 0
                distribution_level = "POOR" if distribution_score > 2 else "FAIR" if distribution_score > 1.5 else "GOOD"
            else:
                distribution_level = "UNKNOWN"
            
            return {
                "project_id": project_id,
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": period_days,
                },
                "team_size": len(team_members),
                "team_members": team_members,
                "metrics": {
                    "total_assigned_issues": total_assigned_issues,
                    "total_state_changes": total_state_changes,
                    "total_comments": total_comments,
                },
                "analysis": {
                    "bottleneck_level": bottleneck_level,
                    "work_distribution": distribution_level,
                    "collaboration_score": "MEDIUM",  # Would be calculated based on comment interactions
                },
                "recommendations": [
                    "Redistribute work more evenly" if distribution_level == "POOR" else None,
                    "Consider increasing team capacity" if bottleneck_level == "HIGH" else None,
                    "Review team collaboration patterns" if total_comments < total_assigned_issues else None,
                ],
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing team dynamics: {str(e)}")


@router.get("/workflow-optimization/{project_id}")
async def get_workflow_optimization(
    project_id: str = Path(..., description="Project ID"),
) -> Dict[str, Any]:
    """
    Analyze workflow patterns and suggest optimizations.
    
    This endpoint identifies:
    - State transition patterns
    - Time spent in each state
    - Bottlenecks in the workflow
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Query for workflow metrics
            result = session.run("""
                MATCH (p:Project {id: $project_id})
                MATCH (i:Issue)-[:BELONGS_TO]->(p)
                MATCH (sc1:StateChange)-[:CHANGES]->(i)
                MATCH (sc1)-[:FROM]->(from_state:State)
                MATCH (sc1)-[:TO]->(to_state:State)
                
                // Count transitions between states
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
            
            # Get time in each state
            result = session.run("""
                MATCH (p:Project {id: $project_id})
                MATCH (i:Issue)-[:BELONGS_TO]->(p)
                MATCH (sc1:StateChange)-[:CHANGES]->(i)-[:HAS_STATE]->(current_state:State)
                MATCH (sc1)-[:TO]->(to_state:State)
                WHERE to_state.name = current_state.name
                
                // Get the most recent transition to the current state
                WITH i, sc1, to_state, current_state
                ORDER BY sc1.timestamp DESC
                WITH i, collect(sc1)[0] AS latest_transition, to_state, current_state
                
                // Calculate time in current state
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
            
            # Identify potential bottlenecks
            bottlenecks = []
            for state in time_in_states:
                if state["avg_days_in_state"] > 7 and state["issue_count"] > 5:
                    bottlenecks.append({
                        "state": state["state"],
                        "avg_days_in_state": state["avg_days_in_state"],
                        "issue_count": state["issue_count"],
                    })
            
            return {
                "project_id": project_id,
                "workflow_metrics": {
                    "state_transitions": transitions,
                    "time_in_states": time_in_states,
                },
                "bottlenecks": bottlenecks,
                "recommendations": [
                    f"Review process for state '{b['state']}' which has {b['issue_count']} issues stuck for an average of {b['avg_days_in_state']} days"
                    for b in bottlenecks
                ],
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing workflow: {str(e)}") 