import random

import pykokkos as pk


@pk.workload
class RandomSum:
    def __init__(self, n):
        self.N: int = n
        self.total: int = 0
        self.a: pk.View1D[pk.int32] = pk.View([n], pk.int32)

        for i in range(self.N):
            self.a[i] = random.randint(0, 10)

        print("Initialized view:", self.a)

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, self.my_reduction)

    @pk.callback
    def results(self):
        print("Sum:", self.total)

    @pk.workunit
    def my_reduction(self, i: int, accumulator: pk.Acc[pk.int32]):
        accumulator += self.a[i]


if __name__ == "__main__":
    n = 10
    pk.execute(pk.ExecutionSpace.OpenMP, RandomSum(n))
