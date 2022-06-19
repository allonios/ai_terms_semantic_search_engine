from django import forms


class OntologyUnifierForm(forms.Form):
    file = forms.FileField()
