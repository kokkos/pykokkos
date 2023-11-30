import pykokkos as pk

import argparse
from functools import reduce
import sys

@pk.workload
class KokkosStream:
    def __init__(self, ARRAY_SIZE: int, initA: float, initB: float, initC: float,
            scalar: float, num_times: int):
        self.array_size: int = ARRAY_SIZE

        self.a: pk.View1D[pk.double] = pk.View([ARRAY_SIZE], pk.double)
        self.b: pk.View1D[pk.double] = pk.View([ARRAY_SIZE], pk.double)
        self.c: pk.View1D[pk.double] = pk.View([ARRAY_SIZE], pk.double)

        self.initA: pk.double = initA
        self.initB: pk.double = initB
        self.initC: pk.double = initC
        self.scalar: pk.double = scalar
        self.num_times: int = num_times
        self.sum: pk.double = 0

        self.runtime: float = 0
        self.runtimes: pk.View2D[pk.double] = pk.View([5, num_times], pk.double)

    @pk.main
    def run(self):
        pk.parallel_for(self.array_size, self.init_arrays)

        timer = pk.Timer()
        for i in range(self.num_times):
            pk.parallel_for("babel_stream", self.array_size, self.copy)
            pk.fence()
            # self.runtimes[0][i] = timer.seconds()
            # timer.reset()

            pk.parallel_for("babel_stream", self.array_size, self.mul)
            pk.fence()
            # self.runtimes[1][i] = timer.seconds()
            # timer.reset()

            pk.parallel_for("babel_stream", self.array_size, self.add)
            pk.fence()
            pk.parallel_for("babel_stream", self.array_size, self.triad)
            pk.fence()
            self.sum = pk.parallel_reduce("babel_stream", self.array_size, self.dot)

        self.runtime = timer.seconds()

    @pk.callback
    def results(self):
        goldA = self.initA
        goldB = self.initB
        goldC = self.initC

        for i in range(self.num_times):
            goldC = goldA
            goldB = self.scalar * goldC
            goldC = goldA + goldB
            goldA = goldB + self.scalar * goldC

        errA = reduce(lambda s, val: s + abs(val - goldA), self.a)
        errA /= len(self.a)
        errB = reduce(lambda s, val: s + abs(val - goldB), self.b)
        errB /= len(self.b)
        errC = reduce(lambda s, val: s + abs(val - goldC), self.c)
        errC /= len(self.c)

        # epsi = sys.float_info.epsilon * 100
        epsi = 1e-8
        if (errA > epsi):
            print(f"Validation failed on a[]. Average error {errA}")
        if (errB > epsi):
            print(f"Validation failed on b[]. Average error {errB}")
        if (errC > epsi):
            print(f"Validation failed on c[]. Average error {errC}")

        goldSum = goldA * goldB * self.array_size
        errSum = self.sum - goldSum
        if (abs(errSum) > 1e-8):
            print(f"Validation failed on sum. Error {errSum}")

    #     total_bytes = 3 * sys.getsizeof(0.0) * self.array_size * num_times;
    #     bandwidth = 1.0e-9 * (total_bytes / self.runtime)
    #     print(f"Runtime (seconds): {self.runtime}")
    #     print(f"Bandwidth (GB/s): {bandwidth}")


    @pk.workunit
    def init_arrays(self, index: int):
        self.a[index] = self.initA
        self.b[index] = self.initB
        self.c[index] = self.initC

    @pk.workunit
    def copy(self, index: int):
        self.c[index] = self.a[index]

    @pk.workunit
    def mul(self, index: int):
        self.b[index] = self.scalar * self.c[index]

    @pk.workunit
    def add(self, index: int):
        self.c[index] = self.a[index] + self.b[index]

    @pk.workunit
    def triad(self, index: int):
        self.a[index] = self.b[index] + self.scalar * self.c[index]

    @pk.workunit
    def dot(self, index: int, acc: pk.Acc[float]):
        acc += self.a[index] * self.b[index]


def run() -> None:
    array_size: int = 2**25 # 100000
    startA: float = 0.1
    startB: float = 0.2
    startC: float = 0.0
    startScalar: float = 0.4
    num_times = 100
    space = pk.ExecutionSpace.OpenMP

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--arraysize", type=int, help="Use SIZE elemnts in the array")
    parser.add_argument("-n", "--numtimes", type=int, help="Run the test NUM times (NUM >= 2)")
    parser.add_argument("-space", "--execution_space", type=str)
    args = parser.parse_args()

    if args.arraysize:
        array_size = 2 ** args.arraysize
    if args.numtimes:
        num_times = args.numtimes
    if args.execution_space:
        space = pk.ExecutionSpace(args.execution_space)

    pk.set_default_space(space)
    pk.execute(space, KokkosStream(array_size, startA, startB, startC, startScalar, num_times))

if __name__ == "__main__":
    run()
