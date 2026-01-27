import pykokkos as pk


@pk.workload
class SquareSum:
    def __init__(self, n):
        self.N: int = n
        self.total: pk.double = 0

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, self.squaresum)

    @pk.callback
    def results(self):
        print("Sum:", self.total)

    @pk.workunit
    def squaresum(self, i: int, acc: pk.Acc[pk.double]):
        acc += i * i


if __name__ == "__main__":
    pk.execute(pk.ExecutionSpace.OpenMP, SquareSum(10))
