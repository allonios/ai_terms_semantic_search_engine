from pipline.handler import BaseHandler
from query_builder.nl_query_processors import (
    ExtractClosestTermsProcessor,
    NLQueryTokenizer,
    SparQLQueryBuilder,
    TermsCombinationsProcessor,
)
from query_builder.ontology_processors import IterableExtractorProcessor
from query_builder.states import QueryState

init_state = QueryState(
    # nl_query="is en_autoencoder isSupervised?",
    # nl_query="what is autoencoder loss function formula?",
    nl_query="what is cnn autoencoder output?",
    ontologies_base_dir="../ignored/data/autoencoder/",
)


handler = BaseHandler(
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
        TermsCombinationsProcessor(),
        ExtractClosestTermsProcessor(
            "subjects",
            "in_query_subjects",
            processor_name="Extract Closest Terms From Subjects Processor",
            similarity_threshold=0.7,
        ),
        ExtractClosestTermsProcessor(
            "predicates",
            "in_query_predicates",
            processor_name="Extract Closest Terms From Predicates Processor",
            similarity_threshold=0.7,
            default="description",
        ),
        SparQLQueryBuilder(),
    ],
    initial_state=init_state,
)

handler.run_processors()

result = handler.current_state
