import numpy as np
import pykokkos as pk


def main():
    N = 10

    # Initialize the array
    i = np.arange(N, dtype=np.int32)
    j = np.arange(3, dtype=np.int32)
    a = i.reshape(-1, 1) * N + j.reshape(1, -1)

    sum_result = pk.parallel_reduce(
        N, lambda i, acc: acc + a[i][0] - a[i][1] + a[i][2], a=a
    )

    print(sum_result)


if __name__ == "__main__":
    main()
