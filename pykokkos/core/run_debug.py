import ast
import inspect
import itertools
from typing import Callable, Dict, Optional, Tuple, Union

from pykokkos.interface import (
    Acc,
    Decorator,
    ExecutionPolicy,
    ExecutionSpace,
    MDRangePolicy,
    TeamMember,
    TeamPolicy,
    TeamThreadRange,
    ThreadVectorRange,
)
import pykokkos.kokkos_manager as km


def call_workunit(
    operation: str,
    workunit: Callable[..., None],
    index: Union[int, Tuple[int, int], TeamMember],
    acc: Acc,
    **kwargs,
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
    initial_value=0,
    **kwargs,
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
                for idx in itertools.product(
                    *[range(*interval) for interval in zip(policy.begin, policy.end)]
                ):
                    call_workunit(operation, workunit, idx, acc, **kwargs)
        else:
            for i in range(policy.begin, policy.end):
                call_workunit(operation, workunit, i, acc, **kwargs)

    return acc.val
