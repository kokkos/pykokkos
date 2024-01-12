# https://github.com/srush/GPU-Puzzles/blob/0e2426b4e4b12e0ae109f7827c14bd0aa62502e2/GPU_puzzlers.py#L155

import pykokkos as pk
import numpy as np

@pk.workunit
def map_2D(i, out, a, size):
    for j in range(size):
        out[i][j] = a[i][j] + 10


def main():
    SIZE = 2
    out = np.zeros((SIZE, SIZE), dtype=int)
    a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
    pk.parallel_for(SIZE, map_2D, out=out, a=a, size=SIZE)
    print(out)


main()
