from typing import Tuple

import pykokkos as pk

from parse_args import parse_args

@pk.functor
class Workload:
    def __init__(self, N: int, M: int, fill: bool):
        self.N: int = N
        self.M: int = M
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


    w = Workload(N, M, fill)

    timer = pk.Timer()

    for i in range(nrepeat):
        result = pk.parallel_reduce(N, w.yAx)

    timer_result = timer.seconds()

    print(f"Computed result for {N} x {M} is {result}")
    solution: float = N * M

    if result != solution:
        pk.printf("Error: result (%lf) != solution (%lf)\n",
                  result, solution)

    print(f"N({N}) M({M}) nrepeat({nrepeat}) problem(MB) time({timer_result}) bandwidth(GB/s)")

if __name__ == "__main__":
    run()
