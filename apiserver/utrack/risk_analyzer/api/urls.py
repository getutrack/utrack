from django.urls import path

from apiserver.utrack.risk_analyzer.api.endpoints import (
    hybrid_search,
    risk,
    analysis,
    embedding
)

urlpatterns = [
    # Hybrid search endpoints
    path('search/hybrid', hybrid_search.hybrid_search, name='hybrid_search'),
    path('search/semantic', hybrid_search.semantic_search, name='semantic_search'),
    path('search/graph/<str:issue_id>', hybrid_search.graph_search, name='graph_search'),
    
    # Risk analysis endpoints
    path('risk/project/<str:project_id>', risk.get_project_risk, name='project_risk'),
    path('risk/issue/<str:issue_id>', risk.get_issue_risk, name='issue_risk'),
    
    # Team and workflow analysis endpoints
    path('analysis/team/<str:project_id>', analysis.get_team_dynamics, name='team_dynamics'),
    path('analysis/workflow/<str:project_id>', analysis.get_workflow_optimization, name='workflow_optimization'),
    
    # Embedding endpoints
    path('embedding/generate', embedding.create_embedding, name='create_embedding'),
    path('embedding/batch', embedding.batch_create_embeddings, name='batch_create_embeddings'),
] 