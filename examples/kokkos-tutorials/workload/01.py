from typing import Tuple

import pykokkos as pk

from parse_args import parse_args


@pk.workload
class Workload:
    def __init__(self, N: int, M: int, nrepeat: int):
        self.N: int = N
        self.M: int = M
        self.nrepeat: int = nrepeat
        self.y: pk.View1D[pk.double] = pk.View([N], pk.double)
        self.x: pk.View1D[pk.double] = pk.View([M], pk.double)
        self.A: pk.View1D[pk.double] = pk.View([N * M], pk.double)

        self.result: float = 0
        self.timer_result: float = 0

    @pk.main
    def run(self):
        pk.parallel_for(self.N, self.y_init)
        # pk.parallel_for(self.N, lambda i : self.y[i] = 1)
        pk.parallel_for(self.M, self.x_init)
        # pk.parallel_for(self.N, lambda i : self.x[i] = 1)
        pk.parallel_for(self.N, self.matrix_init)

        timer = pk.Timer()

        for i in range(self.nrepeat):
            self.result = pk.parallel_reduce("01", self.N, self.yAx)

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
    def y_init(self, i: int):
        self.y[i] = 1

    @pk.workunit
    def x_init(self, i: int):
        self.x[i] = 1

    @pk.workunit
    def matrix_init(self, j: int):
        for i in range(self.M):
            self.A[j * self.M + i] = 1

    @pk.workunit
    def yAx(self, j: int, acc: pk.Acc[float]):
        temp2: float = 0
        for i in range(self.M):
            temp2 += self.A[j * self.M + i] * self.x[i]

        acc += self.y[j] * temp2


if __name__ == "__main__":
    values: Tuple[int, int, int, int, int, str, bool] = parse_args()
    N: int = values[0]
    M: int = values[1]
    nrepeat: int = values[4]

    space: str = values[-2]
    if space == "":
        space = pk.ExecutionSpace.OpenMP
    else:
        space = pk.ExecutionSpace(space)

    pk.set_default_space(space)

    print(f"Total size S = {N * M} N = {N} M = {M}")
    pk.execute(pk.get_default_space(), Workload(N, M, nrepeat))
