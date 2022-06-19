from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic import FormView
from search_engine.forms import SearchForm
from search_engine.mixins import SearchMixin


class SearchFormView(FormView, SearchMixin):
    template_name = "search.html"
    form_class = SearchForm

    def form_valid(self, form):
        return render(
            request=self.request,
            template_name="search_results.html",
            context={
                "results": self.get_search_results(
                    form.clean()["search_query"]
                )
            },
        )


class SearchFormAPIView(FormView, SearchMixin):
    template_name = "search.html"
    form_class = SearchForm

    def form_valid(self, form):
        return JsonResponse(
            data={
                "results": self.get_search_results(
                    form.clean()["search_query"]
                )
            },
        )
