# https://github.com/srush/GPU-Puzzles/blob/0e2426b4e4b12e0ae109f7827c14bd0aa62502e2/GPU_puzzlers.py#L123
# We would not have this case in pykokkos, as we can start with any
# number of threads.

import pykokkos as pk
import numpy as np

@pk.workunit
def map_guard(i, out, a, size):
    if i < size:
        out[i] = a[i] + 10


def main():
    SIZE = 4
    out = np.zeros((SIZE,), dtype=int)
    a = np.arange(SIZE)
    pk.parallel_for("Guard", SIZE * 2, map_guard, out=out, a=a, size=SIZE)
    print(out)


main()
