# https://github.com/srush/GPU-Puzzles/blob/0e2426b4e4b12e0ae109f7827c14bd0aa62502e2/GPU_puzzlers.py#L91

import pykokkos as pk
import numpy as np

@pk.workunit
def a_plus_b(i, out, a, b):
    out[i] = a[i] + b[i]


def main():
    SIZE = 4
    out = np.zeros((SIZE,), dtype=int)
    a = np.arange(SIZE)
    b = np.arange(SIZE)
    pk.parallel_for("Zip", SIZE, a_plus_b, out=out, a=a, b=b)
    print(out)


main()
