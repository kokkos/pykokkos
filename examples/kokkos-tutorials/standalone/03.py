from typing import Tuple

import pykokkos as pk

from parse_args import parse_args

@pk.workunit
def yAx(j, acc, cols, y_view, x_view, A_view):
    temp2: float = 0
    for i in range(cols):
        temp2 += A_view[j][i] * x_view[i]

    acc += y_view[j] * temp2

def run() -> None:
    values: Tuple[int, int, int, int, int, bool] = parse_args()
    N: int = values[0]
    M: int = values[1]
    fill: bool = values[-1]
    nrepeat: int = 100
    print(f"Total size S = {N * M} N = {N} M = {M}")

    y: pk.View1D = pk.View([N], pk.double)
    x: pk.View1D = pk.View([M], pk.double)
    A: pk.View2D = pk.View([N, M], pk.double)

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

    p = pk.RangePolicy(0, N)

    timer = pk.Timer()

    for i in range(nrepeat):
        result = pk.parallel_reduce(p, yAx, cols=M, y_view=y, x_view=x, A_view=A)

    timer_result = timer.seconds()

    print(f"Computed result for {N} x {M} is {result}")
    solution: float = N * M

    if result != solution:
        pk.printf("Error: result (%lf) != solution (%lf)\n",
                  result, solution)

    print(f"N({N}) M({M}) nrepeat({nrepeat}) problem(MB) time({timer_result}) bandwidth(GB/s)")

if __name__ == "__main__":
    run()
