import numpy as np
import pykokkos as pk


@pk.workunit
def squaresum(i: int, acc, values):
    acc += values[i]


def main():
    N: int = 10
    pk.set_default_space(pk.ExecutionSpace.OpenMP)

    # Create array with squares
    values = np.array([i * i for i in range(N)], dtype=np.int32)

    total = pk.parallel_reduce(N, squaresum, values=values)

    print("Sum:", total)


if __name__ == "__main__":
    main()
