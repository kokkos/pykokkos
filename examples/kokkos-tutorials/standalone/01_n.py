from typing import Tuple

import pykokkos as pk

from parse_args import parse_args

@pk.workunit
def y_init(i, y: pk.View1D[pk.double]):
    y[i] = 1

@pk.workunit
def matrix_init(j, M: int, A: pk.View1D[pk.double]):
    for i in range(M):
        A[j * M + i] = 1

@pk.workunit
def yAx(j, acc: pk.Acc[float], M: int, y: pk.View1D[pk.double], x: pk.View1D[pk.double], A: pk.View1D[pk.double]):
    temp2: float = 0
    for i in range(M):
        temp2 += A[j * M + i] * x[i]

    acc += y[j] * temp2

def run() -> None:
    values: Tuple[int, int, int, int, int, bool] = parse_args()
    N: int = values[0]
    M: int = values[1]
    nrepeat: int = 1 
    print(f"Total size S = {N * M} N = {N} M = {M}")

    y = pk.View([N], pk.double)
    x = pk.View([M], pk.double)
    A = pk.View([N * M], pk.double)

    p = pk.RangePolicy(pk.get_default_space(), 0, N)
    pk.parallel_for(p, y_init, y=y)
    pk.parallel_for(pk.RangePolicy(pk.get_default_space(), 0, M), y_init, y=x)
    pk.parallel_for(p, matrix_init, M=M, A=A)

    timer = pk.Timer()

    for i in range(nrepeat):
        result = pk.parallel_reduce(p, yAx, M=M, y=y, x=x, A=A)

    timer_result = timer.seconds()

    print(f"Computed result for {N} x {M} is {result}")
    solution = N * M

    if result != solution:
        pk.printf("Error: result (%lf) != solution (%lf)\n", result, solution)

    print(f"N({N}) M({M}) nrepeat({nrepeat}) problem(MB) time({timer_result}) bandwidth(GB/s)")

if __name__ == "__main__":
    run()
