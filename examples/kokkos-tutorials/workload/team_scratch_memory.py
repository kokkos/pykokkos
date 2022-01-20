from typing import Tuple

import pykokkos as pk

from parse_args import parse_args


@pk.workload(
    y=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    x=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    A=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
class Workload:
    def __init__(self, N: int, M: int, E: int, nrepeat: int, fill: bool):
        self.N: int = N
        self.M: int = M
        self.E: int = E
        self.nrepeat: int = nrepeat
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

        self.result: float = 0
        self.timer_result: float = 0

    @pk.main
    def run(self):
        timer = pk.Timer()
        scratch_size: int = pk.ScratchView1D[float].shmem_size(M)

        for i in range(self.nrepeat):
            self.result = pk.parallel_reduce(
                "team_scratch_memory",
                pk.TeamPolicy(self.E, "auto", 32).set_scratch_size(0, pk.PerTeam(scratch_size)),
                self.yAx)

        self.timer_result = timer.seconds()

    @pk.callback
    def results(self):
        print(
            f"Computed result for {self.N} x {self.M} x {self.E} is {self.result}")
        solution: float = self.N * self.M * self.E

        if self.result != solution:
            pk.printf("Error: result (%lf) != solution (%lf)\n",
                      self.result, solution)

        print(f"N({self.N}) M({self.M}) E({self.E}) nrepeat({self.nrepeat}) problem(MB) time({self.timer_result}) bandwidth(GB/s)")

    @pk.workunit
    def yAx(self, team_member: pk.TeamMember, acc: pk.Acc[float]):
        e: int = team_member.league_rank()
        s_x: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), self.M)

        def init_scratch(i: int):
            s_x[i] = self.x[e][i]

        if team_member.team_rank() == 0:
            pk.parallel_for(pk.ThreadVectorRange(team_member, self.M), init_scratch)

        team_member.team_barrier()

        def team_reduce(j: int, team_acc: pk.Acc[float]):
            def vector_reduce(i: int, vector_acc: pk.Acc[float]):
                vector_acc += self.A[e][j][i] * s_x[i]

            tempM: float = pk.parallel_reduce(pk.ThreadVectorRange(
                team_member, self.M), vector_reduce)

            team_acc += self.y[e][j] * tempM

        tempN: float = pk.parallel_reduce(
            pk.TeamThreadRange(team_member, self.N), team_reduce)

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

    space: str = values[-2]
    if space == "":
        space = pk.ExecutionSpace.OpenMP
    else:
        space = pk.ExecutionSpace(space)

    pk.set_default_space(space)

    print(f"Total size S = {N * M} N = {N} M = {M} E = {E}")
    pk.execute(pk.get_default_space(), Workload(N, M, E, nrepeat, fill))
