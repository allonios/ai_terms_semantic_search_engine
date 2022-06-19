from django.urls import path
from ontology_unifier.views import OntologyUnifierFormAPIView, OntologyUnifierFormView

urlpatterns = [
    path("", OntologyUnifierFormView.as_view()),
    path("api/", OntologyUnifierFormAPIView.as_view()),
]
