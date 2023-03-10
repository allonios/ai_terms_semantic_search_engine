from typing import Any, List

from search_engine.pipline.processors import BaseProcessor
from search_engine.utils.utils import exec_timer


class BaseHandler:
    def __init__(
        self,
        processors: List[BaseProcessor],
        initial_state: Any = None,
        verbose: bool = True,
        run_timer: bool = False,
    ):
        self.processors = processors
        self.current_state = initial_state

        self.verbose = verbose
        self.run_timer = run_timer

    def run_processors(self, init_state=None) -> None:
        if init_state:
            self.current_state = init_state

        for processor in self.processors:
            if self.verbose:
                print("Running:", processor.PROCESSOR_NAME)
            if self.run_timer:
                named_timer_func = exec_timer(processor.PROCESSOR_NAME)
                processor_func = named_timer_func(processor)
                self.current_state = processor_func(self.current_state)
            else:
                self.current_state = processor(self.current_state)
