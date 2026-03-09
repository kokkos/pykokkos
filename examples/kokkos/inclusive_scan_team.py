import numpy as np
import pykokkos as pk


@pk.workunit
def init_data(i: int, view):
    view[i] = i + 1


# Test inclusive_scan with scratch memory
@pk.workunit(scratch=[(int, lambda p: p.team_size)])
def team_scan(team_member: pk.TeamMember, view, team_size):
    offset: int = team_member.league_rank() * team_size
    localIdx: int = team_member.team_rank()
    globalIdx: int = offset + localIdx
    team_rank: int = team_member.team_rank()

    scratch: pk.ScratchView1D[int] = pk.ScratchView1D(
        team_member.team_scratch(0), team_size
    )

    scratch[team_rank] = view[globalIdx]
    team_member.team_barrier()

    pk.inclusive_scan(team_member, scratch)
    team_member.team_barrier()

    view[globalIdx] = scratch[team_rank]


def main():
    N = 64
    team_size = 8
    num_teams = (N + team_size - 1) // team_size

    view = np.zeros([N], dtype=np.int32)
    p_init = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, N)
    pk.parallel_for(p_init, init_data, view=view)

    print(f"Total elements: {N}, Team size: {team_size}, Number of teams: {num_teams}")

    team_policy = pk.TeamPolicy(pk.ExecutionSpace.OpenMP, num_teams, team_size)

    print("Running kernel...")
    pk.parallel_for(team_policy, team_scan, view=view, team_size=team_size)
    print(f"View, splitted by two groups of size = {team_size}")
    print(view)


if __name__ == "__main__":
    main()
