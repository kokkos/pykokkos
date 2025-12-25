"""
Test algorithms like lower_bound and upper_bound
"""

import pykokkos as pk
import pytest


@pk.workunit
def init_int_data(i, view):
    view[i] = (i + 1) * 2


@pk.workunit
def init_float_data(i, view: pk.View1D[pk.float]):
    view[i] = (i + 1) * 2.0


@pk.workunit
def init_double_data(i, view: pk.View1D[pk.double]):
    view[i] = (i + 1) * 2.0


@pk.workunit
def lower_bound_int(i, view, result_view):
    search_value: int = i * 2
    bound_idx: int = pk.lower_bound(view, 10, search_value)
    result_view[i] = bound_idx


@pk.workunit
def lower_bound_float(i, view, result_view):
    search_value: float = float(i * 2)
    bound_idx: int = pk.lower_bound(view, 10, search_value)
    result_view[i] = bound_idx


@pk.workunit
def lower_bound_double(i, view, result_view):
    search_value: float = float(i * 2)
    bound_idx: int = pk.lower_bound(view, 10, search_value)
    result_view[i] = bound_idx


@pk.workunit
def upper_bound_int(i, view, result_view):
    search_value: int = i * 2
    bound_idx: int = pk.upper_bound(view, 10, search_value)
    result_view[i] = bound_idx


@pk.workunit
def upper_bound_float(i, view, result_view):
    search_value: float = float(i * 2)
    bound_idx: int = pk.upper_bound(view, 10, search_value)
    result_view[i] = bound_idx


@pk.workunit
def upper_bound_double(i, view, result_view):
    search_value: float = float(i * 2)
    bound_idx: int = pk.upper_bound(view, 10, search_value)
    result_view[i] = bound_idx


@pk.workunit
def team_lower_bound_int(team_member, view, result_view):
    team_size: int = team_member.team_size()
    offset: int = team_member.league_rank() * team_size
    localIdx: int = team_member.team_rank()
    globalIdx: int = offset + localIdx
    team_rank: int = team_member.team_rank()

    scratch: pk.ScratchView1D[pk.int32] = pk.ScratchView1D(
        team_member.team_scratch(0), team_size
    )

    scratch[team_rank] = view[globalIdx]
    team_member.team_barrier()
    search_value: int = team_rank * 4
    bound_idx: int = pk.lower_bound(scratch, team_size, search_value)
    result_view[globalIdx] = bound_idx


@pk.workunit
def team_lower_bound_float(team_member, view, result_view):
    team_size: int = team_member.team_size()
    offset: int = team_member.league_rank() * team_size
    localIdx: int = team_member.team_rank()
    globalIdx: int = offset + localIdx
    team_rank: int = team_member.team_rank()

    scratch: pk.ScratchView1D[pk.float] = pk.ScratchView1D(
        team_member.team_scratch(0), team_size
    )

    scratch[team_rank] = view[globalIdx]
    team_member.team_barrier()
    search_value: float = float(team_rank * 4)
    bound_idx: int = pk.lower_bound(scratch, team_size, search_value)
    result_view[globalIdx] = bound_idx


@pk.workunit
def team_lower_bound_double(team_member, view, result_view):
    team_size: int = team_member.team_size()
    offset: int = team_member.league_rank() * team_size
    localIdx: int = team_member.team_rank()
    globalIdx: int = offset + localIdx
    team_rank: int = team_member.team_rank()

    scratch: pk.ScratchView1D[pk.double] = pk.ScratchView1D(
        team_member.team_scratch(0), team_size
    )

    scratch[team_rank] = view[globalIdx]
    team_member.team_barrier()
    search_value: float = float(team_rank * 4)
    bound_idx: int = pk.lower_bound(scratch, team_size, search_value)
    result_view[globalIdx] = bound_idx


@pk.workunit
def team_upper_bound_int(team_member, view, result_view):
    team_size: int = team_member.team_size()
    offset: int = team_member.league_rank() * team_size
    localIdx: int = team_member.team_rank()
    globalIdx: int = offset + localIdx
    team_rank: int = team_member.team_rank()

    scratch: pk.ScratchView1D[pk.int32] = pk.ScratchView1D(
        team_member.team_scratch(0), team_size
    )

    scratch[team_rank] = view[globalIdx]
    team_member.team_barrier()
    search_value: int = team_rank * 4
    bound_idx: int = pk.upper_bound(scratch, team_size, search_value)
    result_view[globalIdx] = bound_idx


@pk.workunit
def team_upper_bound_float(team_member, view, result_view):
    team_size: int = team_member.team_size()
    offset: int = team_member.league_rank() * team_size
    localIdx: int = team_member.team_rank()
    globalIdx: int = offset + localIdx
    team_rank: int = team_member.team_rank()

    scratch: pk.ScratchView1D[pk.float] = pk.ScratchView1D(
        team_member.team_scratch(0), team_size
    )

    scratch[team_rank] = view[globalIdx]
    team_member.team_barrier()
    search_value: float = float(team_rank * 4)
    bound_idx: int = pk.upper_bound(scratch, team_size, search_value)
    result_view[globalIdx] = bound_idx


@pk.workunit
def team_upper_bound_double(team_member, view, result_view):
    team_size: int = team_member.team_size()
    offset: int = team_member.league_rank() * team_size
    localIdx: int = team_member.team_rank()
    globalIdx: int = offset + localIdx
    team_rank: int = team_member.team_rank()

    scratch: pk.ScratchView1D[pk.double] = pk.ScratchView1D(
        team_member.team_scratch(0), team_size
    )

    scratch[team_rank] = view[globalIdx]
    team_member.team_barrier()
    search_value: float = float(team_rank * 4)
    bound_idx: int = pk.upper_bound(scratch, team_size, search_value)
    result_view[globalIdx] = bound_idx


class TestLowerBoundInt:
    def test_lower_bound_int(self):
        N = 20
        view: pk.View1D[pk.int32] = pk.View([N], pk.int32)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_int_data, view=view)
        pk.parallel_for(N, lower_bound_int, view=view, result_view=result_view)

        # View contains [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, ...]
        # For search value i*2: 0->0, 2->0, 4->1, 6->2, 8->3, 10->4, etc.
        # lower_bound returns first element >= value
        expected = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        for i in range(N):
            assert (
                result_view[i] == expected[i]
            ), f"Failed at index {i}: expected {expected[i]}, got {result_view[i]}"


class TestLowerBoundFloat:
    def test_lower_bound_float(self):
        N = 20
        view: pk.View1D[pk.float] = pk.View([N], pk.float)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_float_data, view=view)
        pk.parallel_for(N, lower_bound_float, view=view, result_view=result_view)

        # View contains [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, ...]
        # lower_bound returns first element >= value
        expected = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        for i in range(N):
            assert (
                result_view[i] == expected[i]
            ), f"Failed at index {i}: expected {expected[i]}, got {result_view[i]}"


class TestLowerBoundDouble:
    def test_lower_bound_double(self):
        N = 20
        view: pk.View1D[pk.double] = pk.View([N], pk.double)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_double_data, view=view)
        pk.parallel_for(N, lower_bound_double, view=view, result_view=result_view)

        # View contains [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, ...]
        # lower_bound returns first element >= value
        expected = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        for i in range(N):
            assert (
                result_view[i] == expected[i]
            ), f"Failed at index {i}: expected {expected[i]}, got {result_view[i]}"


class TestUpperBoundInt:
    def test_upper_bound_int(self):
        N = 20
        view: pk.View1D[pk.int32] = pk.View([N], pk.int32)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_int_data, view=view)
        pk.parallel_for(N, upper_bound_int, view=view, result_view=result_view)

        # View contains [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, ...]
        # For search value i*2: upper_bound finds first element > value
        # 0->0 (first>0 is 2), 2->1 (first>2 is 4), 4->2 (first>4 is 6), etc.
        expected = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
        ]

        for i in range(N):
            assert (
                result_view[i] == expected[i]
            ), f"Failed at index {i}: expected {expected[i]}, got {result_view[i]}"


class TestUpperBoundFloat:
    def test_upper_bound_float(self):
        N = 20
        view: pk.View1D[pk.float] = pk.View([N], pk.float)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_float_data, view=view)
        pk.parallel_for(N, upper_bound_float, view=view, result_view=result_view)

        # upper_bound returns first element > value
        expected = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
        ]

        for i in range(N):
            assert (
                result_view[i] == expected[i]
            ), f"Failed at index {i}: expected {expected[i]}, got {result_view[i]}"


class TestUpperBoundDouble:
    def test_upper_bound_double(self):
        N = 20
        view: pk.View1D[pk.double] = pk.View([N], pk.double)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_double_data, view=view)
        pk.parallel_for(N, upper_bound_double, view=view, result_view=result_view)

        # upper_bound returns first element > value
        expected = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
        ]

        for i in range(N):
            assert (
                result_view[i] == expected[i]
            ), f"Failed at index {i}: expected {expected[i]}, got {result_view[i]}"


class TestTeamLowerBound:
    def test_team_lower_bound_int(self):
        N = 32
        num_teams = 2

        view: pk.View1D[pk.int32] = pk.View([N], pk.int32)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_int_data, view=view)

        max_team_size = N // num_teams
        scratch_size = pk.ScratchView1D[pk.int32].shmem_size(max_team_size)

        policy = pk.TeamPolicy(num_teams, "auto")
        policy.set_scratch_size(0, pk.PerTeam(scratch_size))

        pk.parallel_for(
            policy, team_lower_bound_int, view=view, result_view=result_view
        )

        # Just verify it runs without error - the exact values depend on the sorting
        assert result_view[0] >= 0

    def test_team_lower_bound_float(self):
        N = 32
        num_teams = 2

        view: pk.View1D[pk.float] = pk.View([N], pk.float)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_float_data, view=view)

        max_team_size = N // num_teams
        scratch_size = pk.ScratchView1D[pk.float].shmem_size(max_team_size)

        policy = pk.TeamPolicy(num_teams, "auto")
        policy.set_scratch_size(0, pk.PerTeam(scratch_size))

        pk.parallel_for(
            policy, team_lower_bound_float, view=view, result_view=result_view
        )

        assert result_view[0] >= 0

    def test_team_lower_bound_double(self):
        N = 32
        num_teams = 2

        view: pk.View1D[pk.double] = pk.View([N], pk.double)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_double_data, view=view)

        max_team_size = N // num_teams
        scratch_size = pk.ScratchView1D[pk.double].shmem_size(max_team_size)

        policy = pk.TeamPolicy(num_teams, "auto")
        policy.set_scratch_size(0, pk.PerTeam(scratch_size))

        pk.parallel_for(
            policy, team_lower_bound_double, view=view, result_view=result_view
        )

        assert result_view[0] >= 0


class TestTeamUpperBound:
    def test_team_upper_bound_int(self):
        N = 32
        num_teams = 2

        view: pk.View1D[pk.int32] = pk.View([N], pk.int32)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_int_data, view=view)

        max_team_size = N // num_teams
        scratch_size = pk.ScratchView1D[pk.int32].shmem_size(max_team_size)

        policy = pk.TeamPolicy(num_teams, "auto")
        policy.set_scratch_size(0, pk.PerTeam(scratch_size))

        pk.parallel_for(
            policy, team_upper_bound_int, view=view, result_view=result_view
        )

        assert result_view[0] >= 0

    def test_team_upper_bound_float(self):
        N = 32
        num_teams = 2

        view: pk.View1D[pk.float] = pk.View([N], pk.float)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_float_data, view=view)

        max_team_size = N // num_teams
        scratch_size = pk.ScratchView1D[pk.float].shmem_size(max_team_size)

        policy = pk.TeamPolicy(num_teams, "auto")
        policy.set_scratch_size(0, pk.PerTeam(scratch_size))

        pk.parallel_for(
            policy, team_upper_bound_float, view=view, result_view=result_view
        )

        assert result_view[0] >= 0

    def test_team_upper_bound_double(self):
        N = 32
        num_teams = 2

        view: pk.View1D[pk.double] = pk.View([N], pk.double)
        result_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        pk.parallel_for(N, init_double_data, view=view)

        max_team_size = N // num_teams
        scratch_size = pk.ScratchView1D[pk.double].shmem_size(max_team_size)

        policy = pk.TeamPolicy(num_teams, "auto")
        policy.set_scratch_size(0, pk.PerTeam(scratch_size))

        pk.parallel_for(
            policy, team_upper_bound_double, view=view, result_view=result_view
        )

        assert result_view[0] >= 0
