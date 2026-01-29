import pykokkos as pk


@pk.workunit
def initialize_view(i: int, a: pk.View2D[pk.int32]):
    for j in range(3):
        a[i][j] = (i + 1) ** (j + 1)


@pk.workunit
def my_reduction(i: int, accumulator: pk.Acc[pk.double], a: pk.View2D[pk.int32]):
    accumulator += a[i][0] * a[i][1] / (a[i][2])


def main():
    N: int = 10
    pk.set_default_space(pk.ExecutionSpace.OpenMP)

    a: pk.View2D[pk.int32] = pk.View([N, 3], pk.int32)

    pk.parallel_for(N, initialize_view, a=a)
    total: int = pk.parallel_reduce(N, my_reduction, a=a)

    for row in a:
        print(row)
    print("\nResult is", total)


if __name__ == "__main__":
    main()
