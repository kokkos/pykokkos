import pykokkos as pk

import argparse
import sys

@pk.workunit
def init_view(i: int, inp: pk.View1D[pk.double], init: float):
    inp[i] = init

@pk.workunit
def nstream( i: int, A: pk.View1D[pk.double], B: pk.View1D[pk.double], C: pk.View1D[pk.double], scalar: float):
    A[i] += B[i] + scalar * C[i]

@pk.workunit
def res_reduce(i: int, acc: pk.Acc[pk.double], A: pk.View1D[pk.double]):
    acc += abs(A[i])


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

    A: pk.View1D = pk.View([length], pk.double)
    B: pk.View1D = pk.View([length], pk.double)
    C: pk.View1D = pk.View([length], pk.double)

    p = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, length)
    
    pk.parallel_for(p, init_view, inp=A, init=0)
    pk.parallel_for(p, init_view, inp=B, init=2)
    pk.parallel_for(p, init_view, inp=C, init=2)
    # pk.fence()

    timer = pk.Timer()

    for i in range(iterations):
        pk.parallel_for(p, nstream, A=A, B=B, C=C, scalar=scalar)

    # pk.fence()
    nstream_time = timer.seconds()

    # verify correctness
    ar: float = 0
    br: float = 2
    cr: float = 2
    for i in range(iterations):
        ar += br + scalar * cr

    ar *= length

    asum = pk.parallel_reduce(p, res_reduce, A=A)
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