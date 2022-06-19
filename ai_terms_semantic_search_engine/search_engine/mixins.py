from typing import List

from django.conf import settings
from search_engine.query_builder.states import QueryState


class SearchMixin:
    def get_search_results(self, query: str) -> List[str]:
        settings.SEARCH_ENGINE.run_processors(
            QueryState(
                nl_query=query, ontologies_base_dir=settings.ONTOLOGIES_PATH
            )
        )
        return settings.SEARCH_ENGINE.current_state.search_result
