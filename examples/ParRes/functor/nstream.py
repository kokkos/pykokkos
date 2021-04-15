import pykokkos as pk

import argparse
import sys

@pk.functor
class Workload:
    def __init__(self, iterations, length, offset, scalar):
        self.iterations: int = iterations
        self.length: int = length
        self.offset: int = offset
        self.scalar: int = scalar

        self.A: pk.View1D[pk.double] = pk.View([length], pk.double)
        self.B: pk.View1D[pk.double] = pk.View([length], pk.double)
        self.C: pk.View1D[pk.double] = pk.View([length], pk.double)
        self.scalar: float = 3
        self.asum: float = 0

        self.nstream_time: float = 0 

    @pk.workunit
    def init_views(self, i: int):
        self.A[i] = 0
        self.B[i] = 2
        self.C[i] = 2

    @pk.workunit
    def nstream(self, i: int):
        self.A[i] += self.B[i] + self.scalar * self.C[i]

    @pk.workunit
    def res_reduce(self, i: int, acc: pk.Acc[pk.double]):
        acc += abs(self.A[i])


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('iterations', type=int)
    parser.add_argument('length', type=int)
    parser.add_argument('offset', nargs='?', type=int, default=0)
    args = parser.parse_args()
    iterations = args.iterations
    length = args.length
    offset = args.offset
    scalar = 3

    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    if length <= 0:
        sys.exit("ERROR: vector length must be positive")

    # emulate cpp example
    if length <= 0:
        sys.exit("ERROR: offset must be nonnegative")

    print("Number of iterations = " , iterations)
    print("Vector length        = " , length)
    print("Offset               = " , offset)

    p = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, length)
    w = Workload(iterations, length, offset, scalar)
    
    pk.parallel_for(p, w.init_views)
    # pk.fence()

    timer = pk.Timer()

    for i in range(iterations):
        pk.parallel_for(p, w.nstream)

    # pk.fence()
    nstream_time = timer.seconds()

    # verify correctness
    ar: float = 0
    br: float = 2
    cr: float = 2
    for i in range(iterations):
        ar += br + scalar * cr

    ar *= length

    asum = pk.parallel_reduce(p, w.res_reduce)
    # pk.fence()

    episilon: float = 1.0e-8
    if (abs(ar-asum)/asum > episilon):
        print("ERROR: Failed Valication on output array")
    else:
        avgtime: float = nstream_time/iterations
        nbytes: float = 4.0 * length * 4
        print("Solution validates")
        print("Rate (MB/s): %.2f" % (1.e-6*nbytes/avgtime))
        print("Avg time (ms): %f" % (avgtime/1.e-3))

if __name__ == "__main__":
    run()