from typing import Tuple

import pykokkos as pk

from parse_args import parse_args

@pk.workunit(
    y=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    x=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    A=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
def yAx(team_member: pk.TeamMember, acc: pk.Acc[float], N:int, M: int, y: pk.View2D[pk.double], x: pk.View2D[pk.double], A: pk.View3D[pk.double]):
    e: int = team_member.league_rank()

    def team_reduce(j: int, team_acc: pk.Acc[float]):
        def vector_reduce(i: int, vector_acc: pk.Acc[float]):
            vector_acc += A[e][j][i] * x[e][i]

        tempM: float = pk.parallel_reduce(
            pk.ThreadVectorRange(team_member, M), vector_reduce)

        team_acc += y[e][j] * tempM

    tempN: float = pk.parallel_reduce(
        pk.TeamThreadRange(team_member, N), team_reduce)

    def single_closure():
        nonlocal acc
        acc += tempN

    pk.single(pk.PerTeam(team_member), single_closure)


def run() -> None:
    values: Tuple[int, int, int, int, int, bool] = parse_args()
    N: int = values[0]
    M: int = values[1]
    E: int = values[3]
    fill: bool = values[-1]
    nrepeat: int = 1000
    print(f"Total size S = {N * M} N = {N} M = {M} E = {E}")

    y: pk.View2D = pk.View([E, N], pk.double, layout=pk.Layout.LayoutRight)
    x: pk.View2D = pk.View([E, M], pk.double, layout=pk.Layout.LayoutRight)
    A: pk.View3D = pk.View([E, N, M], pk.double, layout=pk.Layout.LayoutRight)

    if fill:
        y.fill(1)
        x.fill(1)
        A.fill(1)
    else:
        for e in range(E):
            for i in range(N):
                y[e][i] = 1

            for i in range(M):
                x[e][i] = 1

            for j in range(N):
                for i in range(M):
                    A[e][j][i] = 1

    p = pk.TeamPolicy(E, "auto", 32, pk.get_default_space())

    timer = pk.Timer()

    for i in range(nrepeat):
        result = pk.parallel_reduce(p, yAx, N=N, M=M, y=y, x=x, A=A)

    timer_result = timer.seconds()

    print(
        f"Computed result for {N} x {M} x {E} is {result}")
    solution: float = N * M * E

    if result != solution:
        pk.printf("Error: result (%lf) != solution (%lf)\n",
                  result, solution)

    print(f"N({N}) M({M}) E({E}) nrepeat({nrepeat}) problem(MB) time({timer_result}) bandwidth(GB/s)")

if __name__ == "__main__":
    run()
