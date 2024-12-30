# Third party modules
from rest_framework import status
from rest_framework.response import Response

# Module imports
from utrack.app.serializers import StateSerializer
from utrack.app.views.base import BaseAPIView
from utrack.db.models import State
from utrack.app.permissions import WorkspaceEntityPermission
from utrack.utils.cache import cache_response

class WorkspaceStatesEndpoint(BaseAPIView):
    permission_classes = [
        WorkspaceEntityPermission,
    ]

    @cache_response(60 * 60 * 2)
    def get(self, request, slug):
        states = State.objects.filter(
            workspace__slug=slug,
            project__project_projectmember__member=request.user,
            project__project_projectmember__is_active=True,
            project__archived_at__isnull=True,
            is_triage=False,
        )
        serializer = StateSerializer(states, many=True).data
        return Response(serializer, status=status.HTTP_200_OK)
