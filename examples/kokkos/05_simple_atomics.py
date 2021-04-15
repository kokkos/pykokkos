import math
import random

import pykokkos as pk


@pk.workload(count=pk.ViewTypeInfo(trait=pk.Atomic))
class SimpleAtomics:
    def __init__(self, n):
        self.N: int = n

        self.data: pk.View1D[pk.int32] = pk.View([n], pk.int32)
        self.result: pk.View1D[pk.int32] = pk.View([n], pk.int32)
        self.count: pk.View1D[pk.int32] = pk.View([1], pk.int32, trait=pk.Trait.Atomic)

        for i in range(n):
            self.data[i] = random.randint(0, n)

    @pk.main
    def run(self):
        pk.parallel_for(self.N, self.findprimes)

    @pk.callback
    def results(self):
        for i in range(int(self.count[0])):
            print(int(self.result[i]), end=", ")
        print(
            "\nFound", int(
                self.count[0]), "prime numbers in", self.N, "random numbers"
        )

    @pk.workunit
    def findprimes(self, i: int):
        number: int = self.data[i]
        upper_bound: int = math.sqrt(number) + 1
        is_prime: bool = not (number % 2 == 0)
        k: int = 3
        idx: int = 0
        while k < upper_bound and is_prime:
            is_prime = not (number % k == 0)
            k += 2
        if is_prime:
            idx = self.count[0] = self.count[0] + 1
            self.result[idx - 1] = number


if __name__ == "__main__":
    pk.execute(pk.ExecutionSpace.OpenMP, SimpleAtomics(100))
