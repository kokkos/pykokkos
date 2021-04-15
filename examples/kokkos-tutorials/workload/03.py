from typing import Tuple

import pykokkos as pk

from parse_args import parse_args


@pk.workload
class Workload:
    def __init__(self, N: int, M: int, nrepeat: int, fill: bool):
        self.N: int = N
        self.M: int = M
        self.nrepeat: int = nrepeat
        self.y: pk.View1D[pk.double] = pk.View([N], pk.double)
        self.x: pk.View1D[pk.double] = pk.View([M], pk.double)
        self.A: pk.View2D[pk.double] = pk.View([N, M], pk.double)

        if fill:
            self.y.fill(1)
            self.x.fill(1)
            self.A.fill(1)
        else:
            for i in range(N):
                self.y[i] = 1

            for i in range(M):
                self.x[i] = 1

            for j in range(N):
                for i in range(M):
                    self.A[j][i] = 1

        self.result: float = 0
        self.timer_result: float = 0

    @pk.main
    def run(self):
        timer = pk.Timer()

        for i in range(self.nrepeat):
            self.result = pk.parallel_reduce("03", self.N, self.yAx)

        self.timer_result = timer.seconds()

    @pk.callback
    def results(self):
        print(f"Computed result for {self.N} x {self.M} is {self.result}")
        solution: float = self.N * self.M

        if self.result != solution:
            pk.printf("Error: result (%lf) != solution (%lf)\n",
                      self.result, solution)

        print(f"N({self.N}) M({self.M}) nrepeat({self.nrepeat}) problem(MB) time({self.timer_result}) bandwidth(GB/s)")

    @pk.workunit
    def yAx(self, j: int, acc: pk.Acc[float]):
        temp2: float = 0
        for i in range(self.M):
            temp2 += self.A[j][i] * self.x[i]

        acc += self.y[j] * temp2


if __name__ == "__main__":
    values: Tuple[int, int, int, int, int, bool] = parse_args()
    N: int = values[0]
    M: int = values[1]
    nrepeat: int = values[4]
    fill: bool = values[-1]

    space: str = values[-2]
    if space == "":
        space = pk.ExecutionSpace.OpenMP
    else:
        space = pk.ExecutionSpace(space)

    pk.set_default_space(space)

    print(f"Total size S = {N * M} N = {N} M = {M}")
    pk.execute(pk.get_default_space(), Workload(N, M, nrepeat, fill))
