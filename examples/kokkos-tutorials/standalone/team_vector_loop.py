from typing import Tuple

import pykokkos as pk

from parse_args import parse_args

@pk.workunit
def yAx(team_member, acc, rows, cols, y_view, x_view, A_view):
    e: int = team_member.league_rank()

    def team_reduce(j: int, team_acc: pk.Acc[float]):
        def vector_reduce(i: int, vector_acc: pk.Acc[float]):
            vector_acc += A_view[e][j][i] * x_view[e][i]

        tempM: float = pk.parallel_reduce(
            pk.ThreadVectorRange(team_member, cols), vector_reduce)

        team_acc += y_view[e][j] * tempM

    tempN: float = pk.parallel_reduce(
        pk.TeamThreadRange(team_member, rows), team_reduce)

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

    p = pk.TeamPolicy(E, pk.AUTO, 32)

    timer = pk.Timer()

    for i in range(nrepeat):
        result = pk.parallel_reduce(p, yAx, rows=N, cols=M, y_view=y, x_view=x, A_view=A)

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
