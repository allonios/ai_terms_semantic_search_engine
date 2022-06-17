from abc import ABCMeta


class BaseProcessor(metaclass=ABCMeta):
    PROCESSOR_NAME = "Base Processor"

    def __init__(self, init_state=None) -> None:
        self.state = init_state

    def __call__(self, input_state):
        self.process_state(input_state)
        return self.state

    def process_state(self, input_state=None):
        if input_state:
            self.state = input_state
        return self.state
