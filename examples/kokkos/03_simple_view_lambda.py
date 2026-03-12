import numpy as np
import pykokkos as pk

def main():
    N = 10

    i = np.arange(1, N + 1, dtype=np.int32)
    j = np.arange(1, 3 + 1, dtype=np.int32)
    a = i.reshape(-1, 1) ** j.reshape(1, -1)
    print(a)

    total: int = pk.parallel_reduce(N, lambda u, acc: acc + a[i][0] * a[i][1] / a[i][2], a=a)

    for row in a:
        print(row)
    print("\nResult is", total)


if __name__ == "__main__":
    main()
