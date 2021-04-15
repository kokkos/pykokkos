from typing import Tuple

import pykokkos as pk

from parse_args import parse_args

@pk.functor(
    y=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    x=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    A=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
class Workload:
    def __init__(self, N: int, M: int, E: int, fill: bool):
        self.N: int = N
        self.M: int = M
        self.y: pk.View2D[pk.double] = pk.View([E, N], pk.double, layout=pk.Layout.LayoutRight)
        self.x: pk.View2D[pk.double] = pk.View([E, M], pk.double, layout=pk.Layout.LayoutRight)
        self.A: pk.View3D[pk.double] = pk.View([E, N, M], pk.double, layout=pk.Layout.LayoutRight)

        if fill:
            self.y.fill(1)
            self.x.fill(1)
            self.A.fill(1)
        else:
            for e in range(E):
                for i in range(N):
                    self.y[e][i] = 1

                for i in range(M):
                    self.x[e][i] = 1

                for j in range(N):
                    for i in range(M):
                        self.A[e][j][i] = 1

    @pk.workunit
    def yAx(self, team_member: pk.TeamMember, acc: pk.Acc[float]):
        e: int = team_member.league_rank()

        def team_reduce(j: int, team_acc: pk.Acc[float]):
            def vector_reduce(i: int, vector_acc: pk.Acc[float]):
                vector_acc += self.A[e][j][i] * self.x[e][i]

            tempM: float = pk.parallel_reduce(
                pk.ThreadVectorRange(team_member, self.M), vector_reduce)

            team_acc += self.y[e][j] * tempM

        tempN: float = pk.parallel_reduce(
            pk.TeamThreadRange(team_member, self.N), team_reduce)

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

    w = Workload(N, M, E, fill)
    p = pk.TeamPolicy(E, "auto", 32, pk.get_default_space())

    timer = pk.Timer()

    for i in range(nrepeat):
        result = pk.parallel_reduce(p, w.yAx)

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
