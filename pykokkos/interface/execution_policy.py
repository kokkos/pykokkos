from enum import Enum
from typing import Final, List, Tuple, Union

import pykokkos.kokkos_manager as km

from .execution_space import ExecutionSpace, ExecutionSpaceInstance
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

    def __init__(self, *args):
        """
        RangePolicy constructor

        :param *args:
            :param space: (optional) an ExecutionSpace or
                ExecutionSpaceInstance object. If not specified, default
                is used.
            :param begin: the tid of the first thread
            :param end: the total number of threads
        """

        unpacked: Tuple = tuple(args)

        if len(unpacked) == 2:
            space = km.get_default_space()
            begin = unpacked[0]
            end = unpacked[1]

        elif len(unpacked) == 3:
            space = unpacked[0]
            begin = unpacked[1]
            end = unpacked[2]

        else:
            raise ValueError(f"Incorrect number of arguments {len(unpacked)}")

        if not isinstance(begin, int):
            raise TypeError(f"Invalid argument {begin}")

        if not isinstance(end, int):
            raise TypeError(f"Invalid argument {end}")

        if isinstance(space, ExecutionSpace):
            if space is ExecutionSpace.Default:
                space = km.get_default_space()

            if space is not ExecutionSpace.Debug:
                space = km.get_execution_space_instance(space)

        elif not isinstance(space, ExecutionSpaceInstance):
            raise TypeError(f"Invalid space argument {space}")

        self.space: ExecutionSpaceInstance = space
        self.begin: int = begin
        self.end: int = end


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
    def __init__(self, *args):
        """
        TeamPolicy constructor

        :param *args:
            :param space: (optional) a string or
                ExecutionSpaceInstance object. If not specified, default
                is used
            :param league_size: the total number of teams
            :param team_size: the number of threads per team
        """

        unpacked: Tuple = tuple(args)

        if len(unpacked) == 2:
            space = km.get_execution_space_instance(km.get_default_space())
            league_size = unpacked[0]
            team_size = unpacked[1]
            vector_length = -1

        elif len(unpacked) == 3:
            first = unpacked[0]
            second = unpacked[1]
            third = unpacked[2]

            if isinstance(first, ExecutionSpace) or isinstance(first, ExecutionSpaceInstance):
                space = first
                league_size = second
                team_size = third
                vector_length = -1
            else:
                space = km.get_execution_space_instance(km.get_default_space())
                league_size = first
                team_size = second
                vector_length = third

        elif len(unpacked) == 4:
            space = unpacked[0]
            league_size = unpacked[1]
            team_size = unpacked[2]
            vector_length = unpacked[3]

        else:
            raise ValueError(f"Incorrect number of arguments {len(unpacked)}")

        if not isinstance(league_size, int):
            raise TypeError(f"Invalid argument {league_size}")

        if not isinstance(team_size, int):
            team_size = -1

        if not isinstance(vector_length, int):
            vector_length = -1

        if isinstance(space, ExecutionSpace):
            if space is ExecutionSpace.Default:
                space = km.get_default_space()
            space = ExecutionSpaceInstance(space)

        elif not isinstance(space, ExecutionSpaceInstance):
            raise TypeError(f"Invalid space argument {space}")

        self.space: ExecutionSpaceInstance = space
        self.league_size: int = league_size
        self.team_size: int = team_size
        self.vector_length: int = vector_length

    def set_scratch_size(self, level: int, per_team_or_thread): # -> TeamPolicy:
        pass


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


class TeamThreadMDRange(ExecutionPolicy):
    def __init__(self, *args) -> None:
        pass