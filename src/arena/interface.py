from abc import ABC, abstractmethod
from typing import Callable

from torchjd.aggregation import Aggregator
import importlib
from functools import partial
import ast


class Interface(ABC):
    @abstractmethod
    def __call__(self, representation: str) -> Callable:
        """Returns the callable corresponding to a given representation."""


class CurryingInterface(Interface):
    def __call__(self, representation: str) -> Callable:
        """
        Interface able to import and curry a function based on its representation.

        Example:
            >>> CurryingInterface()('torchjd.aggregation._dual_cone_utils.project_weights(solver="quadprog")')
            # Output: partial(torchjd.aggregation._dual_cone_utils.project_weights, solver="quadprog")
        """

        module_name = ".".join(representation[:representation.find("(")].split(".")[:-1])
        function_name = representation[:representation.find("(")].split(".")[-1]
        params_string = representation[representation.find("(")+1:-1]
        kwargs = ast.literal_eval(params_string)
        # kwargs = {}
        # for param_string in params_string.split(","):
        #     param_string = param_string.strip()
        #     key, value_str = param_string.split("=")
        #     key = key.strip()
        #     value_str = value_str.strip()
        #     kwargs[key] = eval(value_str)

        print(kwargs)

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

        class_name = representation[:representation.find("(")]
        _import_from_module("torchjd.aggregation", class_name)

        return eval(representation)


def _import_from_module(module_name: str, object_name: str):
    module = importlib.import_module(module_name)
    cls = getattr(module, object_name)
    globals()[object_name] = cls


def parse_string_to_dict(input_string):
    """
    Converts a string like '"a"=5.0, "b"="test"' into a Python dictionary.

    Args:
        input_string: The string to parse.

    Returns:
        A dictionary
    """
    result_dict = {}
    # Split the string by commas to get key-value pairs
    pairs = input_string.split(',')

    for pair in pairs:
        # Split each pair by the equals sign
        if '=' in pair:
            key_str, value_str = pair.split('=', 1)

            # Clean up the key (remove quotes and leading/trailing spaces)
            key = key_str.strip().strip('"')

            # Clean up the value and attempt to convert its type
            value = value_str.strip()

            # Try to evaluate the value to handle numbers and strings correctly
            try:
                # Safely evaluate the string to a Python literal
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If evaluation fails, keep it as a string after cleaning quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

            result_dict[key] = value

    return result_dict


fn = CurryingInterface()('torchjd.aggregation._dual_cone_utils.project_weights(solver="quadprog", U=torch.tensor([1., 2.]))')
# CurryingInterface()('torchjd.aggregation._dual_cone_utils.project_weights(solver="quadprog", mabite=3)')
# CurryingInterface()('torchjd.aggregation._dual_cone_utils.project_weights(solver="quadprog", test=None)')
