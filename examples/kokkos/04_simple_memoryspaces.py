import pykokkos as pk


@pk.workunit
def reduction(i: int, acc: pk.Acc[pk.double], a: pk.View2D[pk.int32]):
    acc += a[i][0] - a[i][1] + a[i][2]


def main():
    N: int = 10
    pk.set_default_space(pk.ExecutionSpace.OpenMP)

    a: pk.View2D[pk.int32] = pk.View([N, 3], pk.int32)

    # Initialize the view
    for i in range(N):
        for j in range(3):
            a[i][j] = i * N + j

    sum_result: int = pk.parallel_reduce(N, reduction, a=a)

    print(sum_result)


if __name__ == "__main__":
    main()
