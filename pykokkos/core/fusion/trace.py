from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pykokkos.core.type_inference import UpdatedDecorator, UpdatedTypes 
from pykokkos.interface import ExecutionPolicy


class Future:
    def __init__(self) -> None:
        self.value = None

    def assign_value(self, value) -> None:
        self.value = value

    def __add__(self, other):
        assert self.value is not None
        return self.value + other

    def __sub__(self, other):
        assert self.value is not None
        return self.value - other

    def __mul__(self, other):
        assert self.value is not None
        return self.value * other

    def __str__(self):
        assert self.value is not None
        return str(self.value)

    def __repr__(self) -> str:
        assert self.value is not None
        return str(self.value)


@dataclass
class TracerOperation:
    """
    A single operation in a trace
    """

    future: Optional[Future]
    name: Optional[str]
    policy: ExecutionPolicy
    workunit: Callable[..., None]
    operation: str
    decorator: UpdatedDecorator
    types: UpdatedTypes
    args: Dict[str, Any]


class Tracer:
    """
    Holds traces of operations
    """

    def __init__(self) -> None:
        self.operations: List[TracerOperation] = []

    def log_operation(
        self,
        future: Optional[Future],
        name: Optional[str],
        policy: ExecutionPolicy,
        workunit: Callable[..., None],
        operation: str,
        updated_decorator: UpdatedDecorator,
        updated_types: Optional[UpdatedTypes],
        **kwargs
    ) -> None:
        """
        Log the workunit and its arguments in the trace

        :param name: the name of the kernel
        :param policy: the execution policy of the operation
        :param workunit: the workunit function object
        :param kwargs: the keyword arguments passed to the workunit
        :param updated_decorator: Object with decorator specifier information
        :param updated_types: UpdatedTypes object with type inference information
        :param operation: the name of the operation "for", "reduce", or "scan"
        :param initial_value: the initial value of the accumulator
        """

        self.operations.append(
            TracerOperation(future, name, policy, workunit, operation, updated_decorator, updated_types, dict(kwargs)))
