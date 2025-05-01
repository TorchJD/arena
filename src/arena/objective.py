from abc import ABC, abstractmethod

from torchjd.aggregation import Aggregator


class Objective(ABC):
    @abstractmethod
    def __call__(self, A: Aggregator) -> float:
        """Returns the value of the objective obtained for the provided aggregator."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
