# Module imports
from utrack.license.models import Instance
from utrack.app.serializers import BaseSerializer
from utrack.app.serializers import UserAdminLiteSerializer


class InstanceSerializer(BaseSerializer):
    primary_owner_details = UserAdminLiteSerializer(
        source="primary_owner", read_only=True
    )

    class Meta:
        model = Instance
        exclude = [
            "license_key",
            "user_count"
        ]
        read_only_fields = [
            "id",
            "email",
            "last_checked_at",
            "is_setup_done",
        ]
