# TODO 1- extract all Subjects.
# TODO 2- extract all Classes.
# TODO 3- extract all Predicates.

from pipline.processors import BaseProcessor
from itertools import chain


class IterableExtractorProcessor(BaseProcessor):
    def __init__(
            self,
            get_attr_name: str,
            set_attr_name: str,
            init_state=None,
            processor_name: str = "Iterable Extractor Processor"
    ) -> None:
        super().__init__(init_state)
        self.get_attr_name = get_attr_name
        self.set_attr_name = set_attr_name
        self.PROCESSOR_NAME = processor_name

    def process_state(self, input_state=None):
        self.state = super().process_state(input_state)

        setattr(
            self.state,
            self.set_attr_name,
            set(
                chain(
                    *map(
                        lambda individuals: individuals,
                        map(
                            lambda onto: list(
                                getattr(onto, self.get_attr_name)()),
                            self.state.ontologies
                        )
                    )
                )
            )
        )

        return self.state
