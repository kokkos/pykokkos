from typing import Tuple

import pykokkos as pk

from parse_args import parse_args


@pk.workunit
def yAx(j: int, acc: pk.Acc[float], M: int,
        y: pk.View1D[pk.double], x: pk.View1D[pk.double], A: pk.View2D[pk.double]):
    temp2: float = 0
    A_row_j = A[j,:];
    for i in range(M):
        temp2 += A_row_j[i] * x[i]

    acc += y[j] * temp2


if __name__ == "__main__":
    values: Tuple[int, int, int, int, int, bool] = parse_args()
    N: int = values[0]
    M: int = values[1]
    fill: bool = values[-1]
    nrepeat: int = 100

    space: str = values[-2]
    if space == "":
        space = pk.ExecutionSpace.OpenMP
    else:
        space = pk.ExecutionSpace(space)

    pk.set_default_space(space)

    print(f"Total size S = {N * M} N = {N} M = {M}")
    y: pk.View1D[pk.double] = pk.View([N], pk.double)
    x: pk.View1D[pk.double] = pk.View([M], pk.double)
    A: pk.View2D[pk.double] = pk.View([N, M], pk.double)

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

    p = pk.RangePolicy(pk.get_default_space(), 0, N)

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

