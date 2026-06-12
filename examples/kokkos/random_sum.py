import random
import pykokkos as pk


@pk.workunit
def my_reduction(i: int, accumulator: pk.Acc[pk.int32], a: pk.View1D[pk.int32]):
    accumulator += a[i]


def main():
    n: int = 10
    N: int = n
    pk.set_default_space(pk.ExecutionSpace.OpenMP)

    a: pk.View1D[pk.int32] = pk.View([N], pk.int32)

    # Initialize the view with random values
    for i in range(N):
        a[i] = random.randint(0, 10)
    print("Initialized view:", a)

    total: int = pk.parallel_reduce(N, my_reduction, a=a)

    print("Sum:", total)


if __name__ == "__main__":
    main()
