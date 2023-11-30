import pykokkos as pk

import argparse
import sys

@pk.workload
class main:
    def __init__(self, iterations, length, offset):
        self.iterations: int = iterations
        self.length: int = length
        self.offset: int = offset

        self.A: pk.View1D[pk.double] = pk.View([length], pk.double)
        self.B: pk.View1D[pk.double] = pk.View([length], pk.double)
        self.C: pk.View1D[pk.double] = pk.View([length], pk.double)
        self.scalar: float = 3
        self.asum: float = 0

        self.nstream_time: float = 0

    @pk.main
    def run(self):
        pk.parallel_for(self.length, self.init)
        # pk.parallel_for(self.length, lambda i: 0, self.A)
        # pk.parallel_for(self.length, lambda i: 2, self.B)
        # pk.parallel_for(self.length, lambda i: 2, self.C)
        pk.fence()

        timer = pk.Timer()

        for i in range(self.iterations):
            pk.parallel_for("nstream", self.length, self.nstream)

        pk.fence()
        self.nstream_time = timer.seconds()

        # verify correctness
        ar: float = 0
        br: float = 2
        cr: float = 2
        for i in range(self.iterations):
            ar += br + self.scalar * cr

        ar *= self.length

        self.asum = pk.parallel_reduce(self.length, lambda i, acc: acc + abs(self.A[i]))
        pk.fence()

        episilon: float = 1.0e-8
        if (abs(ar-self.asum)/self.asum > episilon):
            pk.printf("ERROR: Failed Valication on output array\n")
        else:
            avgtime: float = self.nstream_time/self.iterations
            nbytes: float = 4.0 * self.length * 4
            pk.printf("Solution validates\n")
            pk.printf("Rate (MB/s): %.2f\n", 1.e-6*nbytes/avgtime)
            pk.printf("Avg time (ms): %f\n", avgtime/1.e-3)

    @pk.workunit
    def nstream(self, i: int):
        self.A[i] += self.B[i] + self.scalar * self.C[i]

    @pk.workunit
    def init(self, i: int):
        self.A[i] = 0
        self.B[i] = 2
        self.C[i] = 2

def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('iterations', type=int)
    parser.add_argument('length', type=int)
    parser.add_argument('offset', nargs='?', type=int, default=0)
    parser.add_argument("-space", "--execution_space", type=str)

    args = parser.parse_args()
    iterations = args.iterations
    length = args.length
    offset = args.offset

    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    if length <= 0:
        sys.exit("ERROR: vector length must be positive")

    # emulate cpp example
    if length <= 0:
        sys.exit("ERROR: offset must be nonnegative")

    if args.execution_space:
        space = pk.ExecutionSpace(args.execution_space)
        pk.set_default_space(space)

    # pk.enable_uvm()

    length = 2 ** length
    print("Number of iterations = " , iterations)
    print("Vector length        = " , length)
    print("Offset               = " , offset)
    pk.execute(pk.ExecutionSpace.Default, main(iterations, length, offset))

if __name__ == "__main__":
    run()
