import pykokkos as pk


@pk.workload
class SimpleView:
    def __init__(self, n):
        self.N: int = n
        self.total: int = 0
        self.a: pk.View2D[pk.int32] = pk.View([self.N, 3], pk.int32)

    @pk.callback
    def results(self):
        for row in self.a:
            print(row)
        print("\nResult is", self.total)

    @pk.main
    def run(self):
        pk.parallel_for(self.N, self.initialize_view)
        self.total = pk.parallel_reduce(self.N, self.my_reduction)

    @pk.workunit
    def initialize_view(self, i: int):
        for j in range(3):
            self.a[i][j] = (i + 1) ** (j + 1)

    @pk.workunit
    def my_reduction(self, i: int, accumulator: pk.Acc[pk.double]):
        accumulator += self.a[i][0] * self.a[i][1] / (self.a[i][2])


if __name__ == "__main__":
    pk.execute(pk.ExecutionSpace.OpenMP, SimpleView(10))
