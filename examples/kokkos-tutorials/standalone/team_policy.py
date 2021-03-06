from typing import Tuple

import pykokkos as pk

from parse_args import parse_args

@pk.workunit(
    y=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    x=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    A=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
def yAx(team_member: pk.TeamMember, acc: pk.Acc[float], M: int, y: pk.View1D[pk.double], x: pk.View1D[pk.double], A: pk.View2D[pk.double]):
    j: int = team_member.league_rank()

    def inner_reduce(i: int, inner_acc: pk.Acc[float]):
        inner_acc += A[j][i] * x[i]

    temp2: float = pk.parallel_reduce(
        pk.TeamThreadRange(team_member, M), inner_reduce)

    if team_member.team_rank() == 0:
        acc += y[j] * temp2

def run() -> None:
    values: Tuple[int, int, int, int, int, bool] = parse_args()
    N: int = values[0]
    M: int = values[1]
    fill: bool = values[-1]
    nrepeat: int = 100
    print(f"Total size S = {N * M} N = {N} M = {M}")

    y: pk.View1D = pk.View([N], pk.double, layout=pk.Layout.LayoutRight)
    x: pk.View1D = pk.View([M], pk.double, layout=pk.Layout.LayoutRight)
    A: pk.View2D = pk.View([N, M], pk.double, layout=pk.Layout.LayoutRight)

    if fill:
        y.fill(1)
        x.fill(1)
        A.fill(1)
    else:
        for i in range(N):
            y[i] = 1

        for i in range(M):
            x[i] = 1

        for j in range(N):
            for i in range(M):
                A[j][i] = 1

    p = pk.TeamPolicy(N, "auto", space=pk.get_default_space())

    timer = pk.Timer()

    for i in range(nrepeat):
        result = pk.parallel_reduce(p, yAx, M=M, y=y, x=x, A=A)

    timer_result = timer.seconds()

    print(f"Computed result for {N} x {M} is {result}")
    solution: float = N * M

    if result != solution:
        pk.printf("Error: result (%lf) != solution (%lf)\n",
                  result, solution)

    print(f"N({N}) M({M}) nrepeat({nrepeat}) problem(MB) time({timer_result}) bandwidth(GB/s)")

if __name__ == "__main__":
    run()
