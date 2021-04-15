import math

import pykokkos as pk


@pk.workload
class Math:
    def __init__(self, n):
        self.N: int = n
        self.a: pk.View1D[pk.int32] = pk.View([n], pk.int32)

        for i in range(self.N):
            self.a[i] = math.sqrt(math.tau)

        print("Initialized view:", self.a)

    @pk.main
    def run(self):
        pk.parallel_for(self.N, self.my_calculation)

    @pk.callback
    def results(self):
        print("Results: ", self.a)

    @pk.workunit
    def my_calculation(self, i: int):
        pk.printf("Running index %d\n", i)
        self.a[i] += (
            math.cos(self.a[i]) + 2 ** i - math.pi /
            math.fabs(self.a[(i + 1) % self.N])
        )


if __name__ == "__main__":
    n = 10
    pk.execute(pk.ExecutionSpace.OpenMP, Math(n))
