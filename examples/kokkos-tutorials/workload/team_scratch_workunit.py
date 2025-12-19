from typing import Tuple
from parse_args import parse_args
import pykokkos as pk

"""
Same example as `team_scratch_memory.py`, but works with workunits instead of
workloads
"""


@pk.workunit
def yAx(team_member, acc: pk.Acc[float], y, x, A, M, N):
    e: int = team_member.league_rank()
    s_x: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), M)

    def init_scratch(i: int):
        s_x[i] = x[e][i]

    if team_member.team_rank() == 0:
        pk.parallel_for(pk.ThreadVectorRange(team_member, M), init_scratch)

    team_member.team_barrier()

    def team_reduce(j: int, team_acc: pk.Acc[float]):
        def vector_reduce(i: int, vector_acc: pk.Acc[float]):
            vector_acc += A[e][j][i] * s_x[i]

        tempM: float = pk.parallel_reduce(
            pk.ThreadVectorRange(team_member, M), vector_reduce
        )

        team_acc += y[e][j] * tempM

    tempN: float = pk.parallel_reduce(pk.TeamThreadRange(team_member, N), team_reduce)

    def single_closure():
        nonlocal acc
        acc += tempN

    pk.single(pk.PerTeam(team_member), single_closure)


if __name__ == "__main__":
    values: Tuple[int, int, int, int, int, bool] = parse_args()
    N: int = values[0]
    M: int = values[1]
    E: int = values[3]
    nrepeat: int = values[4]
    fill: bool = values[-1]

    space = pk.ExecutionSpace.OpenMP
    pk.set_default_space(space)

    # Initialize data
    N: int = N
    M: int = M
    E: int = E
    nrepeat: int = nrepeat
    y = pk.View([E, N], pk.double, layout=pk.Layout.LayoutRight)
    x = pk.View([E, M], pk.double, layout=pk.Layout.LayoutRight)
    A = pk.View([E, N, M], pk.double, layout=pk.Layout.LayoutRight)

    # Fill input arrays
    y.fill(1)
    x.fill(1)
    A.fill(1)
    result: float = 0
    timer_result: float = 0

    # Run kernel
    acc = 0

    timer = pk.Timer()
    # For workunits, pass M and let the C++ code compute scratch size
    # Approximate scratch size: M * sizeof(double) = M * 8 bytes
    # scratch_size: int = pk.ScratchView1D[float].shmem_size(M)
    scratch_size: int = M * 8
    print(f"Before: {N} | {M} | {E}")

    for i in range(nrepeat):
        result = pk.parallel_reduce(
            "team_scratch_workunit",
            pk.TeamPolicy(E, "auto", 32).set_scratch_size(0, pk.PerTeam(scratch_size)),
            yAx, acc=acc, y=y, x=x, A=A, M=M, N=N
        )

    timer_result = timer.seconds()
    solution: float = N * M * E

    print(
        f"result: {result} | solution {solution} | result==soluton: {result==solution}"
    )
    print(f"Total size S = {N * M} N = {N} M = {M} E = {E}")
