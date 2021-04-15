from enum import Enum
from typing import Final, List, Union

from .execution_space import ExecutionSpace
from .hierarchical import TeamMember


class ExecutionPolicy:
    """
    The parent class for all execution policies
    """

    space: ExecutionSpace


class RangePolicy(ExecutionPolicy):
    """
    Contains the execution space and begin and end threads for a parallel operation
    """

    def __init__(self, space: ExecutionSpace, begin: int, end: int):
        """
        RangePolicy constructor

        :param space: the execution space of the operation
        :param begin: the tid of the first thread
        :param end: the tid of the last thread
        """

        self.space: Final = space
        self.begin: Final = begin 
        self.end : Final = end


class Iterate(Enum):
    Left = "Kokkos::Iterate::Left"
    Right = "Kokkos::Iterate::Right"
    Default = "Kokkos::Iterate::Default"


class Rank:
    def __init__(self, n: int, iter_outer: Iterate, iter_inner: Iterate):
        self.n = n
        self.iter_outer = iter_outer
        self.iter_inner = iter_inner


class MDRangePolicy(ExecutionPolicy):
    def __init__(
        self, begin: List[int], end: List[int], tiling: List[int] = None,
        space: ExecutionSpace = ExecutionSpace.Default,
        iter_outer: Iterate = Iterate.Default,
        iter_inner: Iterate = Iterate.Default,
        rank: Rank = None
    ):

        self.space: Final = space
        self.begin: Final = begin 
        self.end : Final = end
        self.tiling = tiling

        if rank is not None:
            if rank.n != len(begin):
                raise ValueError(f"RangePolicy dimension mismatch: {rank.n} != {len(begin)}")

            iter_outer = rank.iter_outer
            iter_inner = rank.iter_inner

        self.iter_outer: Final = iter_outer
        self.iter_inner: Final = iter_inner

        if len(begin) != len(end):
            raise ValueError(f"RangePolicy dimension mismatch: {len(begin)} != {len(end)}")

        self.rank = len(begin)


class TeamPolicy(ExecutionPolicy):
    def __init__(
        self,
        league_size: int,
        team_size: Union[int, type],
        vector_length: int = 1,
        space: ExecutionSpace = ExecutionSpace.Default
    ):
        self.space: ExecutionSpace = space
        self.league_size: int = league_size
        self.team_size: int = team_size if isinstance(team_size, int) else -1
        self.vector_length: int = vector_length


class TeamThreadRange(ExecutionPolicy):
    def __init__(self, team_member: TeamMember, count: int):
        self.team_member = team_member
        self.count: Final = count
        self.space: ExecutionSpace = ExecutionSpace.Debug


class ThreadVectorRange(ExecutionPolicy):
    def __init__(self, team_member: TeamMember, count: int):
        self.team_member = team_member
        self.count: Final = count
        self.space: ExecutionSpace = ExecutionSpace.Debug
