from abc import ABC, abstractmethod
from typing import Callable


class Objective(ABC):
    @abstractmethod
    def __call__(self, func: Callable) -> float:
        """Returns the value of the objective obtained for the provided function."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
