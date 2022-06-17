from pipline.handler import BaseHandler
from query_builder.nl_query_processors import (
    NLQueryTokenizer,
    TermsCombinationsProcessor,
    ExtractClosestTermsProcessor
)
from query_builder.ontology_processors import IterableExtractorProcessor
from query_builder.states import QueryState

init_state = QueryState(
    # nl_query="is en_autoencoder isSupervised?",
    nl_query="what is autoencoder loss function formula?",
    ontologies_base_dir="../ignored/data/autoencoder/"
)


handler = BaseHandler(
    processors=[
        IterableExtractorProcessor(
            get_attr_name="individuals",
            set_attr_name="subjects",
            processor_name="Subjects Extractor Processor"
        ),
        IterableExtractorProcessor(
            get_attr_name="classes",
            set_attr_name="classes",
            processor_name="Classes Extractor Processor"
        ),
        IterableExtractorProcessor(
            get_attr_name="properties",
            set_attr_name="predicates",
            processor_name="Predicates Extractor Processor"
        ),
        NLQueryTokenizer(),
        TermsCombinationsProcessor(),
        ExtractClosestTermsProcessor("", ""),
    ],
    initial_state=init_state
)

handler.run_processors()

result = handler.current_state

# nl = NLQueryTokenizer(
#     init_state=QueryState(
#         nl_query="what is en_autoencoder loss function formula?",
#         ontologies_base_dir="../ignored/data/autoencoder/"
#     )
# )
# res = nl.process_state()
# tokens = list(res.nl_query_doc)
# what = tokens[0]
# is_ = tokens[1]
# en_autoencoder = tokens[2]
# loss = tokens[3]
# function = tokens[4]
# formula = tokens[5]
# question_mark = tokens[6]

