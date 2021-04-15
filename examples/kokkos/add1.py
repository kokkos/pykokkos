import pykokkos as pk


@pk.workload
class AddOne:
    def __init__(self, n):
        self.N: int = n
        self.a: pk.View1D[pk.int32] = pk.View([n], pk.int32)

        for i in range(self.N):
            self.a[i] = 2
        print(f"Initialized view: [{self.a[0]}, ... repeats {n-1} times]")

    @pk.main
    def run(self):
        pk.parallel_for(self.N, self.add1)

    @pk.callback
    def results(self):
        print(f"Results: [{self.a[0]}, ... repeats {n-1} times]")

    @pk.workunit
    def add1(self, i: int):
        self.a[i] += 1


if __name__ == "__main__":
    n = 100 * 1000
    pk.execute(pk.ExecutionSpace.OpenMP, AddOne(n))
