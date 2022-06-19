from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic import FormView
from ontology_unifier.forms import OntologyUnifierForm
from ontology_unifier.mixins import UnifyOntologyMixin


class OntologyUnifierFormView(FormView, UnifyOntologyMixin):
    template_name = "ontology_unifier_form.html"
    form_class = OntologyUnifierForm

    def form_valid(self, form):
        filepath = self.save_file_to_storage(form.clean()["file"])
        return render(
            request=self.request,
            template_name="ontology_unifier_form.html",
            context={"result": self.unify_file(filepath)},
        )


class OntologyUnifierFormAPIView(FormView, UnifyOntologyMixin):
    template_name = "ontology_unifier_form.html"
    form_class = OntologyUnifierForm

    def form_valid(self, form):
        filepath = self.save_file_to_storage(form.clean()["file"])
        return JsonResponse(
            data={"result": self.unify_file(filepath)},
        )
