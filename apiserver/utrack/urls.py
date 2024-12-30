"""utrack URL Configuration

"""

from django.conf import settings
from django.urls import include, path, re_path
from django.views.generic import TemplateView

handler404 = "utrack.app.views.error_404.custom_404_view"

urlpatterns = [
    path("", TemplateView.as_view(template_name="index.html")),
    path("api/", include("utrack.app.urls")),
    path("api/public/", include("utrack.space.urls")),
    path("api/instances/", include("utrack.license.urls")),
    path("api/v1/", include("utrack.api.urls")),
    path("auth/", include("utrack.authentication.urls")),
    path("", include("utrack.web.urls")),
]


if settings.DEBUG:
    try:
        import debug_toolbar

        urlpatterns = [
            re_path(r"^__debug__/", include(debug_toolbar.urls)),
        ] + urlpatterns
    except ImportError:
        pass
