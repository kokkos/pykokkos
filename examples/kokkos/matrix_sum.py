import pykokkos as pk


@pk.workunit
def sum_row(i: int, mat: pk.View2D[pk.int32], c: int):
    for j in range(1, c):
        mat[i][0] += mat[i][j]


@pk.workunit
def final_sum(i: int, accumulator: pk.Acc[pk.double], mat: pk.View2D[pk.int32]):
    accumulator += mat[i][0]


def main():
    r: int = 5
    c: int = 10
    pk.set_default_space(pk.ExecutionSpace.OpenMP)

    mat: pk.View2D[pk.int32] = pk.View([r, c], pk.int32)

    # Initialize the matrix
    for i in range(r):
        mat[i] = list(range(c * i, c * (i + 1)))

    for row in mat:
        print(row)
    print(f"Initialized {r}x{c} array")

    pk.parallel_for(r, sum_row, mat=mat, c=c)
    total: int = pk.parallel_reduce(r, final_sum, mat=mat)

    print("Total =", total)


if __name__ == "__main__":
    main()
