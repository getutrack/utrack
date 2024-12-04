"""youtrack URL Configuration

"""

from django.conf import settings
from django.urls import include, path, re_path
from django.views.generic import TemplateView

handler404 = "youtrack.app.views.error_404.custom_404_view"

urlpatterns = [
    path("", TemplateView.as_view(template_name="index.html")),
    path("api/", include("youtrack.app.urls")),
    path("api/public/", include("youtrack.space.urls")),
    path("api/instances/", include("youtrack.license.urls")),
    path("api/v1/", include("youtrack.api.urls")),
    path("auth/", include("youtrack.authentication.urls")),
    path("", include("youtrack.web.urls")),
]


if settings.DEBUG:
    try:
        import debug_toolbar

        urlpatterns = [
            re_path(r"^__debug__/", include(debug_toolbar.urls)),
        ] + urlpatterns
    except ImportError:
        pass
