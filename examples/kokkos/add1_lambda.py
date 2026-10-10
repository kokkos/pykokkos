import numpy as np
import pykokkos as pk


@pk.workunit
def add1(i: int, a: pk.View1D[pk.int32]):
    a[i] += 1


def main():
    n = 100 * 1000
    N = n

    # Initialize the array
    a = 2 * np.ones(N, dtype=np.int32)
    print(f"Initialized view: [{a[0]}, ... repeats {n-1} times]")

    pk.parallel_for(N, lambda i: a[i] + 1, a=a)

    print(f"Results: [{a[0]}, ... repeats {n-1} times]")


if __name__ == "__main__":
    main()
