from django import forms


class SearchForm(forms.Form):
    search_query = forms.CharField(
        label="Search Query", max_length=100, required=True
    )
