import importlib
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import torch  # noqa
import torchjd
from torch import Tensor
from torch.nn import Module
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


class ComputeGramianWithAutojacInterface(Interface):
    def __call__(self, representation: str) -> Callable:
        """
        Interface able to make a function to compute the gramian from autojac transforms.

        Example:
            >>> ComputeGramianWithAutojacInterface()('mtl_backward')
            # Output: <function mtl_backward at 0x77745b8e6de0>
        """

        def get_model_gramian_via_autojac(f: Callable, x: Tensor) -> Tensor:
            from torchjd._autojac._transform import Diagonalize, Init, Jac, OrderedSet
            from torchjd._autojac._transform._aggregate import _Matrixify

            output = f(x)
            params = OrderedSet([x])
            outputs = OrderedSet([output])
            init = Init(outputs)
            diag = Diagonalize(outputs)
            jac = Jac(outputs, params, chunk_size=None)
            mat = _Matrixify()

            jacobian_matrices = (mat << jac << diag << init)({})
            gramian = torch.sum(torch.stack([J @ J.T for J in jacobian_matrices.values()]), dim=0)

            return gramian

        return get_model_gramian_via_autojac


class ForwardBackwardAutojacInterface(Interface):
    def __call__(self, _: str):
        from torchjd.aggregation._aggregator_bases import GramianWeightedAggregator
        def forward_backward(model: Module, input: Tensor, aggregator: GramianWeightedAggregator) -> None:
            output = model(input)
            torchjd.backward(output, aggregator)

        return forward_backward


class ForwardBackwardAutogramInterface(Interface):
    def __call__(self, _: str):
        from torchjd.aggregation._aggregator_bases import GramianWeightedAggregator
        from torchjd.autogram._vgp import vgp_from_module, get_gramian
        def forward_backward(model: Module, input: Tensor, aggregator: GramianWeightedAggregator) -> None:
            output, vgp_fn = vgp_from_module(model, input)
            gramian = get_gramian(vgp_fn, output.shape[0])
            weights = aggregator.weighting.weighting(gramian)
            output.backward(weights)


        return forward_backward


def _import_from_module(module_name: str, object_name: str):
    module = importlib.import_module(module_name)
    cls = getattr(module, object_name)
    globals()[object_name] = cls


INTERFACES = {
    "agg": AggregatorInterface(),
    "curry": CurryingInterface(),
    "fn": FnInterface(),
    "compute_gramian_with_autojac": ComputeGramianWithAutojacInterface(),
}
