# https://github.com/srush/GPU-Puzzles/blob/0e2426b4e4b12e0ae109f7827c14bd0aa62502e2/GPU_puzzlers.py#L182

import pykokkos as pk
import numpy as np

@pk.workunit
def broadcast(i, out, a, b, size):
    for j in range(size):
        out[i][j] = a[i][0] + b[j][0]


def main():
    SIZE = 2
    out = np.zeros((SIZE, SIZE), dtype=int)
    a = np.arange(SIZE).reshape(SIZE, 1)
    b = np.arange(SIZE).reshape(1, SIZE)

    pk.parallel_for("Broadcast", SIZE * SIZE, broadcast, out=out, a=a, b=b, size=SIZE)
    print(out)


main()
