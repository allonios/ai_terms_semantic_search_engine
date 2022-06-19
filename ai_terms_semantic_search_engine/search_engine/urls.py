from django.urls import path
from search_engine.views import SearchFormAPIView, SearchFormView

urlpatterns = [
    path("", SearchFormView.as_view()),
    path("api/", SearchFormAPIView.as_view()),
]
