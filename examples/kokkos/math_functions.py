import math
import pykokkos as pk


@pk.workunit
def my_calculation(i: int, a: pk.View1D[pk.int32], N: int):
    pk.printf("Running index %d\n", i)
    a[i] += math.cos(a[i]) + 2**i - math.pi / math.fabs(a[(i + 1) % N])


def main():
    n: int = 10
    N: int = n
    pk.set_default_space(pk.ExecutionSpace.OpenMP)

    a: pk.View1D[pk.int32] = pk.View([N], pk.int32)

    # Initialize the view
    for i in range(N):
        a[i] = math.sqrt(math.tau)
    print("Initialized view:", a)

    pk.parallel_for(N, my_calculation, a=a, N=N)

    print("Results: ", a)


if __name__ == "__main__":
    main()
