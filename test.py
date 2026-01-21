import numpy as np
import pykokkos as pk
from typing import List


@pk.workunit
def work(wid, a, l: List[int]):
    a[wid] = a[wid] + l[wid]


def main():
    N = 10
    a = np.ones(N)
    l = [1] * 10
    print(l)
    print(type(l))
    # ... do anything with a using numpy
    pk.parallel_for("work", 10, work, a=a, l=l)
    print(a)


main()
