# TODO 4- Tokenize the NL query.
# TODO 5- Extract terms combinations with a moving window.
# TODO 6- Search for the closest Term in Subjects or Predicates to a term combination.
# TODO 7- Build the SparQL query based on th extracted Subjects and Predicates.
# import en_core_web_lg
import nltk

from pipline.processors import BaseProcessor
from utils.text_utils import calc_similarity


class NLQueryTokenizer(BaseProcessor):
    PROCESSOR_NAME = "Natural Language Query Processor"

    def process_state(self, input_state=None):
        self.state = super().process_state(input_state)

        # nlp = en_core_web_lg.load()

        self.state.nl_query_tokens = nltk.word_tokenize(self.state.nl_query)
        # self.state.nl_query_doc = nlp(self.state.nl_query)

        return self.state


class TermsCombinationsProcessor(BaseProcessor):
    PROCESSOR_NAME = "Terms Combinations Processor"

    def __init__(self, window_size: int = 3, init_state=None) -> None:
        super().__init__(init_state)
        self.window_size = window_size

    def process_state(self, input_state=None):
        self.state = super().process_state(input_state)

        for index, token in enumerate(self.state.nl_query_tokens):
            for window_size in range(1, self.window_size + 1):
                self.state.terms_combinations.append(
                    " ".join(
                        self.state.nl_query_tokens[index:index + window_size]
                    )
                )

        return self.state


class ExtractClosestTermsProcessor(BaseProcessor):
    def __init__(
            self,
            get_attr_name: str,
            set_attr_name: str,
            similarity_threshold: float = 0.8,
            init_state=None,
            processor_name: str = "Extract Closest Terms Processor"
    ) -> None:
        super().__init__(init_state)
        self.get_attr_name = get_attr_name
        self.set_attr_name = set_attr_name
        self.similarity_threshold = similarity_threshold
        self.PROCESSOR_NAME = processor_name

    def process_state(self, input_state=None):
        self.state = super().process_state(input_state)

        for term in self.state.terms_combinations:
            for subject in self.state.subjects:
                similarity = (
                    (
                        calc_similarity(term, subject)
                        + calc_similarity(subject, term)
                    )
                    / 2
                )

                print(
                    "Term:", term,
                    "Subject:", subject,
                    "Similarity:", similarity
                )
            print("---------------------------------------------------------")






















