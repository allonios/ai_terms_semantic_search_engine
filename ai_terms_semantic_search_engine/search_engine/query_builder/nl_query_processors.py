# TODO 4- Tokenize the NL query.
# TODO 5- Extract terms combinations with a moving window.
# TODO 6- Search for the closest Term in Subjects or Predicates to a term combination.
# TODO 7- Build the SparQL query based on th extracted Subjects and Predicates.
from itertools import chain
from math import inf
from typing import Dict, List

import en_core_web_md
import nltk
import numpy as np
from pipline.processors import BaseProcessor
from utils.sparql_query_utils import find_subjects
from utils.text_utils import calc_similarity


class NLQueryTokenizer(BaseProcessor):
    PROCESSOR_NAME = "Natural Language Query Processor"

    def process_state(self, input_state=None):
        self.state = super().process_state(input_state)

        nlp = en_core_web_md.load()

        self.state.nl_query_tokens = nltk.word_tokenize(self.state.nl_query)
        self.state.nl_query_doc = nlp(self.state.nl_query)

        return self.state


class TermsCombinationsProcessor(BaseProcessor):
    PROCESSOR_NAME = "Terms Combinations Processor"

    def __init__(self, window_size: int = 3, init_state=None) -> None:
        super().__init__(init_state)
        self.window_size = window_size

    def process_state(self, input_state=None):
        self.state = super().process_state(input_state)

        for index, _token in enumerate(self.state.nl_query_tokens):
            for window_size in range(1, self.window_size + 1):
                self.state.terms_combinations.append(
                    self.state.nl_query_doc[index : index + window_size]
                )

        return self.state


class ExtractClosestTermsProcessor(BaseProcessor):
    def __init__(
        self,
        get_attr_name: str,
        set_attr_name: str,
        similarity_threshold: float = 0.8,
        init_state=None,
        processor_name: str = "Extract Closest Terms Processor",
        default: str = None,
    ) -> None:
        super().__init__(init_state)
        self.get_attr_name = get_attr_name
        self.set_attr_name = set_attr_name
        self.similarity_threshold = similarity_threshold
        self.PROCESSOR_NAME = processor_name
        self.default = default

    def __get_max_similarity(self, terms: List) -> Dict:
        max_sim = -inf
        min_sim_len = -inf
        max_sim_index = None
        for index, term in enumerate(terms):
            if term["similarity"] > max_sim:
                max_sim = term["similarity"]
                min_sim_len = len(term["term"])
                max_sim_index = index
            if term["similarity"] == max_sim:
                if len(term["term"]) < min_sim_len:
                    max_sim = term["similarity"]
                    min_sim_len = len(term["term"])
                    max_sim_index = index

        return terms[max_sim_index]

    def __clean_terms(self, terms: List):
        grouped_elements = {}

        for key in map(lambda term: term["element"].name, terms):
            for term in terms:
                if term["element"].name == key:
                    if grouped_elements.get(key, False):
                        grouped_elements[key].append(
                            {
                                "term": term["term"],
                                "element": term["element"],
                                "similarity": term["similarity"],
                            }
                        )
                    else:
                        grouped_elements[key] = [
                            {
                                "term": term["term"],
                                "element": term["element"],
                                "similarity": term["similarity"],
                            }
                        ]

        cleaned_terms = []

        for element in grouped_elements:
            cleaned_terms.append(
                self.__get_max_similarity(grouped_elements[element])
            )

        return cleaned_terms

    def process_state(self, input_state=None):
        self.state = super().process_state(input_state)

        terms = []
        for term in self.state.terms_combinations:
            similarities = []
            for element in getattr(self.state, self.get_attr_name):
                similarity = (
                    calc_similarity(term.text, element.name)
                    + calc_similarity(element.name, term.text)
                ) / 2

                similarities.append(
                    {
                        "term": term,
                        "element": element,
                        "similarity": similarity,
                    }
                )

            max_sim_index = np.argmax(
                list(map(lambda sim: sim["similarity"], similarities))
            )

            highest_sim = similarities[max_sim_index]

            if highest_sim["similarity"] > self.similarity_threshold:
                terms.append(similarities[max_sim_index])
            similarities.clear()

        if self.default and not terms:
            best_predicate = None
            max_sim = -inf
            for predicate in self.state.predicates:
                sim = (
                    calc_similarity(self.default, predicate.name)
                    + calc_similarity(predicate.name, self.default)
                ) / 2

                if sim > max_sim:
                    max_sim = sim
                    best_predicate = predicate
            setattr(
                self.state,
                self.set_attr_name,
                [
                    {
                        "term": "",
                        "element": best_predicate,
                        "similarity": max_sim,
                    }
                ],
            )

            return self.state

        terms = self.__clean_terms(terms)

        setattr(self.state, self.set_attr_name, terms)

        return self.state


class SparQLQueryBuilder(BaseProcessor):
    PROCESSOR_NAME = "SparQL Query Builder"

    def process_state(self, input_state=None):
        self.state = super().process_state(input_state)

        subjects = set(
            map(
                lambda subject: subject["element"],
                self.state.in_query_subjects,
            )
        )

        predicates = set(
            map(
                lambda predicate: predicate["element"],
                self.state.in_query_predicates,
            )
        )

        for subject in subjects:
            self.state.query_graph.extend(
                find_subjects(subject, subjects, predicates)
            )

        self.state.search_result = list(chain(*self.state.query_graph))
