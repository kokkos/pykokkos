import copy
from dataclasses import dataclass
from types import NoneType
from typing import Any, Callable, Dict, List, Optional, Set, Union

from pykokkos.interface import ExecutionPolicy, ViewType

from .future import Future


@dataclass
class DataDependency:
    """
    Represents data + version
    """

    name: Optional[str] # for debugging purposes
    data_id: int
    version: int

    def __hash__(self) -> int:
        return hash((self.data_id, self.version))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataDependency):
            return False

        return self.data_id == other.data_id and self.version == other.version


@dataclass
class TracerOperation:
    """
    A single operation in a trace
    """

    op_id: int
    future: Optional[Future]
    name: Optional[str]
    policy: ExecutionPolicy
    workunit: Callable[..., None]
    operation: str
    args: Dict[str, Any]
    dependencies: Set[DataDependency]

    def __hash__(self) -> int:
        return self.op_id

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TracerOperation):
            return False

        return self.op_id == other.op_id

class Tracer:
    """
    Holds traces of operations
    """

    def __init__(self) -> None:
        self.op_id: int = 0

        # This functions as an ordered set
        self.operations: Dict[TracerOperation, NoneType] = {}

        # Map from each data object id (future or array) to the current version
        self.data_version: Dict[int, int] = {}

        # Map from data version to tracer operation
        self.data_operation: Dict[DataDependency, TracerOperation] = {}

    def log_operation(
        self,
        future: Optional[Future],
        name: Optional[str],
        policy: ExecutionPolicy,
        workunit: Callable[..., None],
        operation: str,
        **kwargs
    ) -> None:
        """
        Log the workunit and its arguments in the trace

        :param name: the name of the kernel
        :param policy: the execution policy of the operation
        :param workunit: the workunit function object
        :param kwargs: the keyword arguments passed to the workunit
        :param operation: the name of the operation "for", "reduce", or "scan"
        :param initial_value: the initial value of the accumulator
        """

        dependencies: Set[DataDependency] = set()

        # For now assume each dependency is RW, later on parse the AST
        for arg, value in kwargs.items():
            if isinstance(value, (Future, ViewType)):
                version: int = self.data_version.get(id(value), 0)
                dependency = DataDependency(arg, id(value), version)

                dependencies.add(dependency)
                self.data_version[id(value)] = version + 1

        tracer_op = TracerOperation(self.op_id, future, name, policy, workunit, operation, dict(kwargs), dependencies)
        self.op_id += 1

        for dependency in dependencies:
            new_dep = copy.deepcopy(dependency)
            new_dep.version += 1
            self.data_operation[new_dep] = tracer_op

        if operation in {"reduce", "scan"}:
            assert future is not None
            self.data_operation[DataDependency(None, id(future), 0)] = tracer_op

        # Add to the ordered set of operations
        self.operations[tracer_op] = None

    def get_operations(self, data: Union[Future, ViewType]) -> List[TracerOperation]:
        """
        Get all the operations needed to update the data of a future
        or view and remove them from the trace

        :param future: the future corresponding to the value that needs to be updated
        :returns: the list of operations to be executed
        """

        version: int = self.data_version.get(id(data), 0)
        dependency = DataDependency(None, id(data), version)

        operation: TracerOperation = self.data_operation[dependency]
        if operation not in self.operations:
            # This means that the dependency was already updated
            return []

        operations: List[TracerOperation] = [operation]
        del self.operations[operation]

        # Ideally, we would not have to do this. By adding an
        # operation to this list, its dependencies should be
        # automatically updated when the operation is executed.
        # However, since the operations are not executed in Python, we
        # cannot trigger the flush. We could also potentially iterate
        # over kwargs prior to invoking a kernel and call
        # flush_value() for all futures and views. We should implement
        # both and benchmark them
        i: int = 0
        while i < len(operations):
            current_op = operations[i]

            for dep in current_op.dependencies:
                if dep not in self.data_operation:
                    assert dep.version == 0
                    continue

                dependency_op: TracerOperation = self.data_operation[dep]
                if dependency_op not in self.operations:
                    # This means that the dependency was already updated
                    continue

                operations.append(dependency_op)
                del self.operations[dependency_op]

            i += 1

        return operations
