import pykokkos as pk


@pk.workload
class SquareSum:
    def __init__(self, n):
        self.N: int = n
        self.total: int = 0

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, lambda i, acc: acc + i*i)

    @pk.callback
    def results(self):
        print("Sum:", self.total)

if __name__ == "__main__":
    pk.execute(pk.ExecutionSpace.OpenMP, SquareSum(10))
