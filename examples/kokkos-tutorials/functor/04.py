from typing import Tuple

import pykokkos as pk

from parse_args import parse_args

@pk.functor
class Workload:
    def __init__(self, N: int, M: int, fill: bool):
        self.N: int = N
        self.M: int = M
        self.y: pk.View1D[pk.double] = pk.View([N], pk.double, space=pk.MemorySpace.CudaSpace)
        self.x: pk.View1D[pk.double] = pk.View([M], pk.double, space=pk.MemorySpace.CudaSpace)
        self.A: pk.View2D[pk.double] = pk.View([N, M], pk.double, space=pk.MemorySpace.CudaSpace)

        # if fill:
        #     self.y.fill(1)
        #     self.x.fill(1)
        #     self.A.fill(1)
        # else:
        #     for i in range(N):
        #         self.y[i] = 1

        #     for i in range(M):
        #         self.x[i] = 1

        #     for j in range(N):
        #         for i in range(M):
        #             self.A[j][i] = 1

    @pk.workunit
    def y_init(self, i: int):
        self.y[i] = 1


    @pk.workunit
    def x_init(self, i: int):
        self.x[i] = 1


    @pk.workunit
    def matrix_init(self, j: int):
        for i in range(self.M):
            self.A[j][i] = 1

    @pk.workunit
    def yAx(self, j: int, acc: pk.Acc[float]):
        temp2: float = 0
        for i in range(self.M):
            temp2 += self.A[j][i] * self.x[i]

        acc += self.y[j] * temp2


def run() -> None:
    values: Tuple[int, int, int, int, int, bool] = parse_args()
    N: int = values[0]
    M: int = values[1]
    fill: bool = values[-1]
    nrepeat: int = 100
    print(f"Total size S = {N * M} N = {N} M = {M}")

    pk.set_default_space(pk.ExecutionSpace.Cuda)

    w = Workload(N, M, fill)
    p = pk.RangePolicy(pk.get_default_space(), 0, N)

    timer = pk.Timer()

    pk.parallel_for(pk.RangePolicy(pk.get_default_space(), 0, N), w.y_init)
    pk.parallel_for(pk.RangePolicy(pk.get_default_space(), 0, M), w.x_init)
    pk.parallel_for(pk.RangePolicy(pk.get_default_space(), 0, N), w.matrix_init)

    for i in range(nrepeat):
        result = pk.parallel_reduce(p, w.yAx)

    timer_result = timer.seconds()

    print(f"Computed result for {N} x {M} is {result}")
    solution: float = N * M

    if result != solution:
        pk.printf("Error: result (%lf) != solution (%lf)\n",
                  result, solution)

    print(f"N({N}) M({M}) nrepeat({nrepeat}) problem(MB) time({timer_result}) bandwidth(GB/s)")

if __name__ == "__main__":
    run()
