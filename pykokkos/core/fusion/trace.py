import ast
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pykokkos.core.parsers import Parser, PyKokkosEntity
from pykokkos.interface import ExecutionPolicy, RangePolicy, ViewType

from .access_modes import AccessMode, get_view_access_modes
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

    op_id: Optional[int] # None for fused operations
    future: Optional[Future]
    name: Optional[str]
    policy: ExecutionPolicy
    workunit: Callable[..., None]
    operation: str
    parser: Parser
    entity_name: str
    args: Dict[str, Any]
    dependencies: Set[DataDependency]

    def __hash__(self) -> int:
        return self.op_id

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TracerOperation):
            return False

        return self.op_id == other.op_id

    def __repr__(self) -> str:
        if self.name is None:
            return self.workunit.__name__

        return self.name


class Tracer:
    """
    Holds traces of operations
    """

    def __init__(self) -> None:
        self.op_id: int = 0

        # This functions as an ordered set
        self.operations: Dict[TracerOperation, None] = {}

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
        parser: Parser,
        entity_name: str,
        **kwargs
    ) -> None:
        """
        Log the workunit and its arguments in the trace

        :param future: the future object corresponding to the output of reductions and scans
        :param name: the name of the kernel
        :param policy: the execution policy of the operation
        :param workunit: the workunit function object
        :param kwargs: the keyword arguments passed to the workunit
        :param operation: the name of the operation "for", "reduce", or "scan"
        :param parser: the parser containing the AST of the workunit
        :param entity_name: the name of the workunit entity
        """

        entity: PyKokkosEntity = parser.get_entity(entity_name)
        AST: ast.FunctionDef = entity.AST

        dependencies: Set[DataDependency]
        access_modes: Dict[str, AccessMode]
        dependencies, access_modes = self.get_data_dependencies(kwargs, AST)

        tracer_op = TracerOperation(self.op_id, future, name, policy, workunit, operation, parser, entity_name, dict(kwargs), dependencies)
        self.op_id += 1

        self.update_output_data_operations(kwargs, access_modes, tracer_op, future, operation)

        self.operations[tracer_op] = None

    def get_operations(self, data: Union[Future, ViewType]) -> List[TracerOperation]:
        """
        Get all the operations needed to update the data of a future
        or view and remove them from the trace

        :param data: the Future or View corresponding to the value that needs to be updated
        :returns: the list of operations to be executed
        """

        version: int = self.data_version.get(id(data), 0)
        dependency = DataDependency(None, id(data), version)

        if dependency not in self.data_operation:
            # The data does not depend on any prior operation
            return []

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
        # over kwargs prior to invoking a kernel and call flush_data()
        # for all futures and views. We should implement both and
        # benchmark them
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

        operations.sort(key=lambda op: op.op_id)

        return operations

    def fuse(self, operations: List[TracerOperation], strategy: str) -> List[TracerOperation]:
        """
        Apply the specified fusion strategy to the given list of operations

        :param operations: the TracerOperations to be fused
        :param strategy: the fusion strategy to follow ("trace", "naive")
        """

        if strategy == "trace":
            return operations

        if strategy == "naive":
            return self.fuse_naive(operations)

        raise RuntimeError(f"Unrecognized fusion strategy '{strategy}'")

    def fuse_naive(self, operations: List[TracerOperation]) -> List[TracerOperation]:
        """
        Fuse a list of operations naively: combine all consecutive
        parallel fors and leave reductions and scans alone

        :param operations: the TracerOperations to be fused
        :returns: the list of TracerOperations post fusion
        """

        fused_ops: List[TracerOperation] = []
        ops_to_fuse: List[TracerOperation] = []

        while len(operations) > 0:
            op: TracerOperation = operations.pop()

            if op.operation == "for":
                ops_to_fuse.append(op)
            elif op.operation == "reduce":
                if len(ops_to_fuse) == 0:
                    ops_to_fuse.append(op)
                else:
                    ops_to_fuse.reverse()
                    fused_ops.append(self.fuse_operations(ops_to_fuse))
                    ops_to_fuse.clear()

                    ops_to_fuse.append(op)
            else:
                ops_to_fuse.reverse()
                fused_ops.append(self.fuse_operations(ops_to_fuse))
                ops_to_fuse.clear()

                ops_to_fuse.append(op)

        # Fuse anything left over
        if len(ops_to_fuse) > 0:
            ops_to_fuse.reverse()
            fused_ops.append(self.fuse_operations(ops_to_fuse))

        fused_ops.reverse()

        return fused_ops

    def fuse_operations(self, operations: List[TracerOperation]) -> TracerOperation:
        """
        Fuse a list of TracerOperations into one

        :param operations: the TracerOperations to be fused
        :returns: the fused operation
        """

        if len(operations) == 1:
            return operations[0]

        names: List[str] = []
        policy: RangePolicy = operations[0].policy
        workunits: List[Callable[..., None]] = []

        # The last operation determines the type of the fused
        # operation since it can be a reduce
        operation: str = operations[-1].operation
        future: Optional[Future] = operations[-1].future

        parsers: List[Parser] = []
        args: Dict[str, Dict[str, Any]] = {}
        dependencies: Set[DataDependency] = set()

        for index, op in enumerate(operations):
            assert isinstance(op.policy, RangePolicy) and policy.begin == op.policy.begin and policy.end == op.policy.end

            names.append(op.name if op.name is not None else op.workunit.__name__)
            workunits.append(op.workunit)
            parsers.append(op.parser)
            args[f"args_{index}"] = op.args
            dependencies.update(op.dependencies)

        fused_name: str
        if len(names) < 5:
            fused_name = "_".join(names)
        else:
            # Avoid long names
            fused_name = "_".join(names[:5]) + hashlib.md5(("".join(names)).encode()).hexdigest()

        return TracerOperation(None, future, fused_name, policy, workunits, operation, parsers, fused_name, args, dependencies)

    def get_data_dependencies(self, kwargs: Dict[str, Any], AST: ast.FunctionDef) -> Tuple[Set[DataDependency], Dict[str, AccessMode]]:
        """
        Get the data dependencies of an operation from its input arguments

        :param kwargs: the keyword arguments passed to the workunit
        :param AST: the AST of the input workunit
        :returns: the set of data dependencies and the access modes of the views
        """

        dependencies: Set[DataDependency] = set()
        view_args: Set[str] = set()

        # First pass to get the Future dependencies and record all the views
        for arg, value in kwargs.items():
            if isinstance(value, Future):
                version: int = self.data_version.get(id(value), 0)
                dependency = DataDependency(arg, id(value), version)

                dependencies.add(dependency)

            if isinstance(value, ViewType):
                view_args.add(arg)

        access_modes: Dict[str, AccessMode] = get_view_access_modes(AST, view_args)

        # Second pass to check if the views are dependencies
        for arg, value in kwargs.items():
            if isinstance(value, ViewType) and access_modes[arg] in {AccessMode.Read, AccessMode.ReadWrite}:
                version: int = self.data_version.get(id(value), 0)
                dependency = DataDependency(arg, id(value), version)

                dependencies.add(dependency)

        return dependencies, access_modes

    def update_output_data_operations(
        self,
        kwargs: Dict[str, Any],
        access_modes: Dict[str, AccessMode],
        tracer_op: TracerOperation,
        future: Optional[Future],
        operation: str
    ) -> None:
        """
        Update the data versions and operations of all data being written to

        :param kwargs: the keyword arguments passed to the workunit
        :param access_modes: how the passed views are being accessed
        :param tracer_op: the current tracer operation being logged
        :param future: the future object corresponding to the output of reductions and scans
        :param operation: the name of the operation "for", "reduce", or "scan"
        """

        for arg, value in kwargs.items():
            if isinstance(value, ViewType) and access_modes[arg] in {AccessMode.Write, AccessMode.ReadWrite}:
                version: int = self.data_version.get(id(value), 0)
                self.data_version[id(value)] = version + 1
                dependency = DataDependency(arg, id(value), version + 1)

                self.data_operation[dependency] = tracer_op

        if operation in {"reduce", "scan"}:
            assert future is not None
            self.data_operation[DataDependency(None, id(future), 0)] = tracer_op
