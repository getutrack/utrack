from django.urls import path


from utrack.app.views import UnsplashEndpoint
from utrack.app.views import GPTIntegrationEndpoint, WorkspaceGPTIntegrationEndpoint


urlpatterns = [
    path(
        "unsplash/",
        UnsplashEndpoint.as_view(),
        name="unsplash",
    ),
    path(
        "workspaces/<str:slug>/projects/<uuid:project_id>/ai-assistant/",
        GPTIntegrationEndpoint.as_view(),
        name="importer",
    ),
   path(
        "workspaces/<str:slug>/ai-assistant/",
        WorkspaceGPTIntegrationEndpoint.as_view(),
        name="importer",
    ),
]
