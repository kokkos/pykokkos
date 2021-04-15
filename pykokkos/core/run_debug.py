import ast
import inspect
import itertools
from typing import Callable, Dict, Optional, Tuple, Union

from pykokkos.interface import (
    Acc, Decorator, ExecutionPolicy, ExecutionSpace,
    MDRangePolicy, TeamMember, TeamPolicy,
    TeamThreadRange, ThreadVectorRange
)
import pykokkos.kokkos_manager as km


def run_workload_debug(workload: object) -> None:
    """
    Run a workload in Python

    :param workload: the workload object
    """

    workload_source: str = inspect.getsource(type(workload))
    tree: ast.Module = ast.parse(workload_source)
    classdef: ast.ClassDef = tree.body[0]
    
    def get_annotated_functions(decorator: Decorator) -> Dict[str, ast.FunctionDef]:
        visitor = ast.NodeVisitor()
        functions: Dict[str, ast.FunctionDef] = {}

        def visit_FunctionDef(node):
            if node.decorator_list:
                try:
                    node_decorator: str = node.decorator_list[0].id
                except AttributeError:
                    node_decorator: str = node.decorator_list[0].attr

                if decorator.value == node_decorator:
                    functions[node.name] = node

        visitor.visit_FunctionDef = visit_FunctionDef
        for method in classdef.body:
            visitor.visit(method)

        return functions

    for name in get_annotated_functions(Decorator.KokkosMain):
        kokkos_main = getattr(workload, name)
        kokkos_main()

    for name in get_annotated_functions(Decorator.KokkosCallback):
        kokkos_callback = getattr(workload, name)
        kokkos_callback()

def call_workunit(
    operation: str,
    workunit: Callable[..., None],
    index: Union[int, Tuple[int, int], TeamMember],
    acc: Acc,
    **kwargs
) -> None:
    """
    Run a workunit for a single iteration

    :param operation: the name of the operation "for", "reduce", or "scan"
    :param workunit: the workunit function object
    :param index: the thread ID value of the current iteration
    :param acc: the accumulator variable (unused by "for")
    :param kwargs: the keyword arguments passed to the workunit
    """

    is_md: bool = isinstance(index, tuple)

    if operation == "for":
        if is_md:
            workunit(*index, **kwargs)
        else:
            workunit(index, **kwargs)

    elif operation == "reduce":
        if is_md:
            workunit(*index, acc, **kwargs)
        else:
            workunit(index, acc, **kwargs)
    elif operation == "scan":
        if is_md:
            workunit(*index, acc, True, **kwargs)
        else:
            workunit(index, acc, True, **kwargs)


def run_workunit_debug(
    policy: ExecutionPolicy,
    workunit: Callable[..., None],
    operation: str,
    initial_value = 0,
    **kwargs
) -> Optional[Union[float, int]]:
    """
    Run a workunit in Python

    :param operation: the name of the operation "for", "reduce", or "scan"
    :param policy: the execution policy of the operation
    :param workunit: the workunit function object
    :param initial_value: the initial value of the accumulator
    :param kwargs: the keyword arguments passed to the workunit
    :returns: the result of the operation (None for parallel_for)
    """

    acc = Acc(initial_value)
    if policy.space is ExecutionSpace.Default:
        policy.space = km.get_default_space()

    if isinstance(policy, TeamPolicy):
        for i in range(policy.league_size):
            call_workunit(operation, workunit, TeamMember(i, 0), acc, **kwargs)

    elif isinstance(policy, TeamThreadRange) or isinstance(policy, ThreadVectorRange):
        for i in range(policy.count):
            call_workunit(operation, workunit, TeamMember(i, 0), acc, **kwargs)

    else:
        if isinstance(policy, MDRangePolicy):
            if policy.rank > 1:
                for idx in itertools.product(*[range(*interval) for interval in zip(policy.begin, policy.end)]):
                    call_workunit(operation, workunit, idx, acc, **kwargs)
        else:
            for i in range(policy.begin, policy.end):
                call_workunit(operation, workunit, i, acc, **kwargs)

    return acc.val
