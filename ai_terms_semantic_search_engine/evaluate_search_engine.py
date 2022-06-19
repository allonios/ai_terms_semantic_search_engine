from itertools import chain
from typing import Any, Dict, List, Union

from search_engine.pipline.handler import BaseHandler
from search_engine.query_builder.nl_query_processors import (
    ExtractClosestTermsProcessor,
    NLQueryTokenizer,
    SparQLQueryBuilder,
    TermsCombinationsProcessor,
)
from search_engine.query_builder.ontology_processors import IterableExtractorProcessor
from search_engine.query_builder.states import QueryState


class SearchEngineEvaluator:
    def __init__(self, tests: List[Dict], search_engine: BaseHandler) -> None:
        self.tests = tests
        self.search_engine = search_engine

    def run_search_engine(self):
        results = []
        for question in self.tests:
            self.search_engine.run_processors(
                QueryState(
                    nl_query=question["question"],
                    ontologies_base_dir="../ignored/data/autoencoder/",
                )
            )
            result = self.search_engine.current_state.search_result
            results.append(
                {
                    "question": question["question"],
                    "result": list(chain(*result)),
                }
            )
        return results

    def get_test_case_by_question(self, question: str) -> Union[Dict, int]:
        result = list(
            filter(lambda test: test["question"] == question, self.tests)
        )
        if result:
            return result[0]
        else:
            return -1

    def search_for_result(self, question: str, result: Any) -> bool:
        test_case = self.get_test_case_by_question(question)

        for expected_result in test_case["result"]:
            if isinstance(expected_result, str):
                if expected_result in result:
                    return True
            else:
                if result == expected_result:
                    return True

        return False

    def evaluate(self) -> Dict:
        search_results = self.run_search_engine()
        results_existence = {}
        for question in search_results:
            results_existence[question["question"]] = []
            for result in question["result"]:
                results_existence[question["question"]].append(
                    self.search_for_result(question["question"], result)
                )

        correct_predictions = len(
            list(
                filter(
                    lambda result: any(results_existence[result]),
                    results_existence,
                )
            )
        )
        accuracy = correct_predictions / len(self.tests)

        return {"accuracy": accuracy}


search_engine = BaseHandler(
    processors=[
        IterableExtractorProcessor(
            get_attr_name="individuals",
            set_attr_name="subjects",
            processor_name="Subjects Extractor Processor",
        ),
        IterableExtractorProcessor(
            get_attr_name="properties",
            set_attr_name="predicates",
            processor_name="Predicates Extractor Processor",
        ),
        NLQueryTokenizer(),
        TermsCombinationsProcessor(window_size=4),
        ExtractClosestTermsProcessor(
            "subjects",
            "in_query_subjects",
            processor_name="Extract Closest Terms From " "Subjects Processor",
            similarity_threshold=0.7,
        ),
        ExtractClosestTermsProcessor(
            "predicates",
            "in_query_predicates",
            processor_name="Extract Closest Terms From "
            "Predicates Processor",
            similarity_threshold=0.7,
            default="description",
        ),
        SparQLQueryBuilder(),
    ],
)

evaluator = SearchEngineEvaluator(
    tests=[
        {
            "case_description": "getting a straight forward attribute.",
            "question": "is autoencoder supervised?",
            "result": [False],
        },
        {
            "case_description": "getting a complex attribute "
            "(it will fail due to file structure).",
            "question": "what is autoencoder loss function formula?",
            "result": ["formula num 1 for testing sake"],
        },
        {
            "case_description": "looking for implicit description.",
            "question": "what is cnn autoencoder?",
            "result": [
                "Convolutional Autoencoders\nconvolutional neural networks "
                "are far better suited than dense networks to work with "
                "images. So if you want to build an autoencoder for images"
            ],
        },
        {
            "case_description": "looking for implicit description.",
            "question": "tell me about the image datasets?",
            "result": ["1-image_net : https://www.image-net.org/"],
        },
        {
            "case_description": "playing with abbreviations.",
            "question": "what is recurrent neural networks autoencoder?",
            "result": [
                "If you want to build an autoencoder for sequences, "
                "such as time series or text (e.g., for unsupervised learning "
                "or dimensionality reduction), then recurrent neural networks "
                "may be better suited than dense networks. Building a "
                "recurrent autoencoder is straightforward: the encoder is "
                "typically a sequence-to-vector RNN which compresses the "
                "input sequence down to a single vector. The decoder is a "
                "vector-to-sequence RNN that does the reverse."
            ],
        },
        {
            "case_description": "looking for implicit description.",
            "question": "what is rnn autoencoder?",
            "result": [
                "If you want to build an autoencoder for sequences, "
                "such as time series or text (e.g., for unsupervised learning "
                "or dimensionality reduction), then recurrent neural networks "
                "may be better suited than dense networks. Building a "
                "recurrent autoencoder is straightforward: the encoder is "
                "typically a sequence-to-vector RNN which compresses the "
                "input sequence down to a single vector. The decoder is a "
                "vector-to-sequence RNN that does the reverse."
            ],
        },
        {
            "case_description": "getting a straight forward attribute.",
            "question": "what is autoencoder optimizer?",
            "result": [""],
        },
        {
            "case_description": "getting a straight forward attribute.",
            "question": "what does autoencoder learns?",
            "result": ["the weights are knowlage abstraction from dataset"],
        },
    ],
    search_engine=search_engine,
)

metrics = evaluator.evaluate()

print(metrics)
