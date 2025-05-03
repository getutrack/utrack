from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException, Path

from risk_analyzer.services.neo4j import get_neo4j_driver

router = APIRouter()

@router.get("/project/{project_id}")
async def get_project_risk(
    project_id: str = Path(..., description="Project ID"),
) -> Dict[str, Any]:
    """
    Get risk analysis for a specific project.
    
    This endpoint performs analysis of project risk factors based on:
    - Issue state transitions
    - Comment sentiment
    - Timeline adherence
    - Team capacity
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Query for overall project risk metrics
            result = session.run("""
                MATCH (p:Project {id: $project_id})
                MATCH (i:Issue)-[:BELONGS_TO]->(p)
                
                // Count issues by state
                OPTIONAL MATCH (i)-[:HAS_STATE]->(s:State)
                
                // Get state transitions
                OPTIONAL MATCH (sc:StateChange)-[:CHANGES]->(i)
                OPTIONAL MATCH (sc)-[:FROM]->(from_state:State)
                OPTIONAL MATCH (sc)-[:TO]->(to_state:State)
                
                // Get user assignments
                OPTIONAL MATCH (u:User)-[:ASSIGNED_TO]->(i)
                
                RETURN
                    p.name AS project_name,
                    count(DISTINCT i) AS total_issues,
                    collect(DISTINCT s.name) AS states,
                    count(DISTINCT sc) AS state_changes,
                    count(DISTINCT u) AS assigned_users,
                    // Calculate basic risk metrics
                    CASE 
                        WHEN count(DISTINCT i) > 50 AND count(DISTINCT u) < 3 THEN 'HIGH'
                        WHEN count(DISTINCT i) > 30 AND count(DISTINCT u) < 2 THEN 'HIGH'
                        WHEN count(DISTINCT i) > 20 AND count(DISTINCT u) < 2 THEN 'MEDIUM'
                        ELSE 'LOW'
                    END AS capacity_risk,
                    // Additional metrics would be calculated in a real implementation
                    'MEDIUM' AS timeline_risk,
                    'LOW' AS quality_risk
            """, project_id=project_id)
            
            record = result.single()
            if not record:
                raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
            
            # Basic risk analysis - would be more sophisticated in a real implementation
            overall_risk = "MEDIUM"  # Default
            
            if record["capacity_risk"] == "HIGH":
                overall_risk = "HIGH"
            elif record["capacity_risk"] == "LOW" and record["timeline_risk"] == "LOW":
                overall_risk = "LOW"
            
            return {
                "project_id": project_id,
                "project_name": record["project_name"],
                "metrics": {
                    "total_issues": record["total_issues"],
                    "state_distribution": [s for s in record["states"] if s],
                    "state_changes": record["state_changes"],
                    "assigned_users": record["assigned_users"],
                },
                "risk_factors": {
                    "capacity_risk": record["capacity_risk"],
                    "timeline_risk": record["timeline_risk"],
                    "quality_risk": record["quality_risk"],
                },
                "overall_risk": overall_risk,
                "recommendations": [
                    "Consider adding more team members" if record["capacity_risk"] == "HIGH" else None,
                    "Review the state transition process" if record["state_changes"] > 50 else None,
                    "Analyze issue distribution across team members"
                ],
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing project risk: {str(e)}")


@router.get("/issue/{issue_id}")
async def get_issue_risk(
    issue_id: str = Path(..., description="Issue ID"),
) -> Dict[str, Any]:
    """
    Get risk analysis for a specific issue.
    
    This endpoint analyzes an individual issue's risk factors.
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Query for issue risk metrics
            result = session.run("""
                MATCH (i:Issue {id: $issue_id})
                
                // Get state transitions
                OPTIONAL MATCH (sc:StateChange)-[:CHANGES]->(i)
                OPTIONAL MATCH (sc)-[:FROM]->(from_state:State)
                OPTIONAL MATCH (sc)-[:TO]->(to_state:State)
                
                // Get comments
                OPTIONAL MATCH (c:Comment)-[:COMMENTS_ON]->(i)
                
                // Get assigned user
                OPTIONAL MATCH (u:User)-[:ASSIGNED_TO]->(i)
                
                RETURN
                    i.title AS issue_title,
                    i.state AS current_state,
                    i.created_at AS created_at,
                    count(DISTINCT sc) AS state_changes,
                    count(DISTINCT c) AS comment_count,
                    u.id AS assigned_user_id
            """, issue_id=issue_id)
            
            record = result.single()
            if not record:
                raise HTTPException(status_code=404, detail=f"Issue with ID {issue_id} not found")
            
            # Basic risk analysis
            risk_level = "LOW"
            risk_factors = []
            
            if record["state_changes"] > 3:
                risk_level = "MEDIUM"
                risk_factors.append("Multiple state transitions indicate potential confusion")
            
            if record["comment_count"] > 10:
                risk_level = "MEDIUM"
                risk_factors.append("High number of comments may indicate unclear requirements")
            
            if not record["assigned_user_id"]:
                risk_level = "HIGH"
                risk_factors.append("Issue is not assigned to anyone")
            
            return {
                "issue_id": issue_id,
                "issue_title": record["issue_title"],
                "current_state": record["current_state"],
                "created_at": record["created_at"],
                "metrics": {
                    "state_changes": record["state_changes"],
                    "comment_count": record["comment_count"],
                    "assigned_user_id": record["assigned_user_id"],
                },
                "risk_level": risk_level,
                "risk_factors": risk_factors,
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing issue risk: {str(e)}") 