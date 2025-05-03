from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# Vector/Embedding models
class EmbeddingRequest(BaseModel):
    text: str
    preprocess: bool = True


class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    preprocess: bool = True


class EmbeddingResponse(BaseModel):
    text: str
    vector: List[float]
    dimension: int


class BatchEmbeddingResponse(BaseModel):
    count: int
    vectors: List[EmbeddingResponse]


# Search models
class SearchFilter(BaseModel):
    project_id: Optional[str] = None
    issue_types: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class SearchResult(BaseModel):
    id: str
    type: str
    score: float
    vector_data: Dict[str, Any]
    graph_data: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    project_id: Optional[str] = None
    results: List[SearchResult]


# Risk analysis models
class RiskMetrics(BaseModel):
    total_issues: int
    state_distribution: List[str]
    state_changes: int
    assigned_users: int


class RiskFactors(BaseModel):
    capacity_risk: str
    timeline_risk: str
    quality_risk: str


class ProjectRiskResponse(BaseModel):
    project_id: str
    project_name: str
    metrics: RiskMetrics
    risk_factors: RiskFactors
    overall_risk: str
    recommendations: List[Optional[str]]


class IssueRiskResponse(BaseModel):
    issue_id: str
    issue_title: str
    current_state: str
    created_at: datetime
    metrics: Dict[str, Any]
    risk_level: str
    risk_factors: List[str]


# Team analysis models
class TeamMember(BaseModel):
    user_id: str
    assigned_issues: int
    state_changes_performed: int
    comments_authored: int
    workload: str


class AnalysisPeriod(BaseModel):
    start_date: str
    end_date: str
    days: int


class TeamMetrics(BaseModel):
    total_assigned_issues: int
    total_state_changes: int
    total_comments: int


class TeamAnalysis(BaseModel):
    bottleneck_level: str
    work_distribution: str
    collaboration_score: str


class TeamDynamicsResponse(BaseModel):
    project_id: str
    analysis_period: AnalysisPeriod
    team_size: int
    team_members: List[TeamMember]
    metrics: TeamMetrics
    analysis: TeamAnalysis
    recommendations: List[Optional[str]]


# Workflow analysis models
class StateTransition(BaseModel):
    from_state: str
    to_state: str
    count: int


class StateTime(BaseModel):
    state: str
    issue_count: int
    avg_days_in_state: float


class Bottleneck(BaseModel):
    state: str
    avg_days_in_state: float
    issue_count: int


class WorkflowMetrics(BaseModel):
    state_transitions: List[StateTransition]
    time_in_states: List[StateTime]


class WorkflowResponse(BaseModel):
    project_id: str
    workflow_metrics: WorkflowMetrics
    bottlenecks: List[Bottleneck]
    recommendations: List[str] 