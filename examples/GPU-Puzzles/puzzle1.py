# https://github.com/srush/GPU-Puzzles/blob/0e2426b4e4b12e0ae109f7827c14bd0aa62502e2/GPU_puzzlers.py#L47

import pykokkos as pk
import numpy as np

@pk.workunit
def plus_ten(i, out, a):
    out[i] = a[i] + 10


def main():
    SIZE = 4
    out = np.zeros((SIZE,), dtype=int)
    a = np.arange(SIZE)
    pk.parallel_for("Map", SIZE, plus_ten, out=out, a=a)
    print(out)


main()
