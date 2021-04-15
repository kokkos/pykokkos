import pykokkos as pk


@pk.workload
class SimpleSpaces:
    def __init__(self, n):
        self.N: int = n
        self.sum: int = 0
        self.a: pk.View2D[pk.int32] = pk.View([n, 3], pk.int32)
        for i in range(n):
            for j in range(3):
                self.a[i][j] = i * n + j

    @pk.main
    def run(self):
        self.sum = pk.parallel_reduce(self.N, self.reduction)

    @pk.callback
    def use_results(self):
        print(self.sum)

    @pk.workunit
    def reduction(self, i: int, acc: pk.Acc[pk.double]):
        acc += self.a[i][0] - self.a[i][1] + self.a[i][2]


if __name__ == "__main__":
    pk.execute(pk.ExecutionSpace.OpenMP, SimpleSpaces(10))
