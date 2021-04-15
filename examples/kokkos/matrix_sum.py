import pykokkos as pk


@pk.workload
class MatrixSum:
    def __init__(self, r, c):
        self.r: int = r
        self.c: int = c
        self.total: int = 0
        self.mat: pk.View2D[pk.int32] = pk.View([r, c], pk.int32)

        for i in range(r):
            self.mat[i] = list(range(c * i, c * (i + 1)))

        for row in self.mat:
            print(row)
        print(f"Initialized {r}x{c} array")

    @pk.main
    def run(self):
        pk.parallel_for(self.r, self.sum_row)
        self.total = pk.parallel_reduce(self.r, self.final_sum)

    @pk.callback
    def results(self):
        print("Total =", self.total)

    @pk.workunit
    def sum_row(self, i: int):
        for j in range(1, self.c):
            self.mat[i][0] += self.mat[i][j]

    @pk.workunit
    def final_sum(self, i: int, accumulator: pk.Acc[pk.double]):
        accumulator += self.mat[i][0]


if __name__ == "__main__":
    pk.execute(pk.ExecutionSpace.OpenMP, MatrixSum(5, 10))
