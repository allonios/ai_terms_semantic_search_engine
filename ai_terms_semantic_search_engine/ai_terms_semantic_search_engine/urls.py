from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("search/", include("search_engine.urls")),
    path("ontology-unifier/", include("ontology_unifier.urls")),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.UNIFIED_ONTOLOGIES_URL,
        document_root=settings.UNIFIED_ONTOLOGIES_ROOT,
    )
