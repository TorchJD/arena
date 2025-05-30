import importlib
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import torch  # noqa
from torchjd.aggregation import Aggregator


class Interface(ABC):
    @abstractmethod
    def __call__(self, representation: str) -> Callable:
        """Returns the callable corresponding to a given representation."""


class CurryingInterface(Interface):
    def __call__(self, representation: str) -> Callable:
        """
        Interface able to import and curry a function based on its representation.

        Example:
            >>> CurryingInterface()('torchjd.aggregation._dual_cone_utils.project_weights{"solver":"quadprog"}')
            # Output: partial(torchjd.aggregation._dual_cone_utils.project_weights, solver="quadprog")
        """

        module_name = ".".join(representation[: representation.find("{")].split(".")[:-1])
        function_name = representation[: representation.find("{")].split(".")[-1]
        params_string = representation[representation.find("{") :]

        kwargs = eval(params_string)

        _import_from_module(module_name, function_name)
        fn = globals()[function_name]

        return partial(fn, **kwargs)


class AggregatorInterface(Interface):
    def __call__(self, representation: str) -> Aggregator:
        """
        Interface able to import and instantiate an aggregator based on its representation.

        Example:
            >>> AggregatorInterface()('UPGrad(reg_eps=0.01)')
            # Output: UPGrad(reg_eps=0.1)
        """

        class_name = representation[: representation.find("(")]
        _import_from_module("torchjd.aggregation", class_name)

        return eval(representation)


class FnInterface(Interface):
    def __call__(self, representation: str) -> Aggregator:
        """
        Interface able to import a function based on its representation.

        Example:
            >>> FnInterface()('mtl_backward')
            # Output: <function mtl_backward at 0x77745b8e6de0>
        """

        _import_from_module("torchjd", representation)
        return eval(representation)


def _import_from_module(module_name: str, object_name: str):
    module = importlib.import_module(module_name)
    cls = getattr(module, object_name)
    globals()[object_name] = cls


INTERFACES = {
    "agg": AggregatorInterface(),
    "curry": CurryingInterface(),
    "fn": FnInterface(),
}
