from abc import ABC, abstractmethod
from typing import List, Any
from report import Reporter, EmptyReporter

Input = Any
Output = Any

class EncoderDecoder(ABC):
    @abstractmethod
    def encode(self, data: Input, reporter: Reporter) -> Output:
        pass

    @abstractmethod
    def decode(self, data: Input) -> Output:
        pass

class Pipeline:
    def __init__(self, processes: List[EncoderDecoder], reporter: Reporter | None):
        self.processes = processes

        self.reporter = reporter
        if reporter is None:
            self.reporter = EmptyReporter()

    def run(self, input: Input):
        decoders = []
        for process in self.processes:
            decoders.insert(0, process)
            input = process.encode(input, self.reporter)

        for process in decoders:
            input = process.decode(input, self.reporter)

        if self.reporter is not None:
            self.reporter.show()

        return input

