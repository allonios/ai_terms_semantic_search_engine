from typing import List

from utils.utils import load_ontologies


class QueryState:
    def __init__(
        self,
        nl_query: str,
        ontologies_base_dir: str,
        nl_query_tokens: List[str] = (),
        sparql_query: str = None,
    ):
        self.nl_query = nl_query
        self.nl_query_tokens = nl_query_tokens
        self.sparql_query = sparql_query

        self.ontologies = load_ontologies(ontologies_base_dir)
        self.subjects = {}
        self.classes = {}
        self.predicates = {}

        self.terms_combinations = []

        self.in_query_subjects = []
        self.in_query_classes = []
        self.in_query_predicates = []

        self.query_graph = []
        self.search_result = []
