import argparse
import random
from typing import Tuple

import pykokkos as pk
import numpy as np

try:
    import cupy as cp

    cupy_available = True
except ImportError:
    cupy_available = False


def get_array_module(space: pk.ExecutionSpace):
    """Return numpy or cupy module based on execution space"""
    if cupy_available and space in (pk.ExecutionSpace.Cuda, pk.ExecutionSpace.HIP):
        return cp
    return np


@pk.functor
# use double type and unroll=8
class Benchmark_double_8:
    def __init__(
        self, N: int, K: int, D: int, R: int, F: int, space: pk.ExecutionSpace
    ):
        self.K: int = K
        self.F: int = F

        xp = get_array_module(space)
        self.connectivity = xp.zeros((N, K), dtype=np.int32)
        self.A = xp.full(N, 1.5, dtype=np.float64)
        self.B = xp.full(N, 2.0, dtype=np.float64)
        self.C = xp.zeros(N, dtype=np.float64)

        # TODO use kokkos to init in parallel
        random.seed(12313)
        connectivity_np = np.zeros((N, K), dtype=np.int32)
        for i in range(N):
            for jj in range(K):
                connectivity_np[i][jj] = (random.randrange(D) + i - D / 2 + N) % N
        if xp is cp:
            self.connectivity = cp.asarray(connectivity_np)
        else:
            self.connectivity = connectivity_np

    @pk.workunit
    def benchmark(self, i: int):
        c: pk.double = 0.0
        for jj in range(self.K):
            j: int = self.connectivity[i][jj]
            a1: pk.double = self.A[j]
            b: pk.double = self.B[j]
            a2: pk.double = a1 * 1.3
            a3: pk.double = a2 * 1.1
            a4: pk.double = a3 * 1.1
            a5: pk.double = a4 * 1.3
            a6: pk.double = a5 * 1.1
            a7: pk.double = a6 * 1.1
            a8: pk.double = a7 * 1.1

            for f in range(self.F):
                a1 += b * a1
                a2 += b * a2
                a3 += b * a3
                a4 += b * a4
                a5 += b * a5
                a6 += b * a6
                a7 += b * a7
                a8 += b * a8

            c += a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8

        self.C[i] = c


def run() -> None:
    # example args 2 100000 32 512 1000 8 8
    # NOTE S and U are hard coded to double and 8 because otherwise we would have a lot of duplicates
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "S", type=int, help="Scalar Type Size (1==float, 2==double, 4==complex<double>)"
    )
    parser.add_argument("N", type=int, help="Number of Entities")
    parser.add_argument("K", type=int, help="Number of things to gather per entity")
    parser.add_argument(
        "D", type=int, help="Max distance of gathered things of an entity"
    )
    parser.add_argument(
        "R", type=int, help="how often to loop through the K dimension with each team"
    )
    parser.add_argument("U", type=int, help="how many independent flops to do per load")
    parser.add_argument(
        "F",
        type=int,
        help="how many times to repeat the U unrolled operations before reading next element",
    )
    parser.add_argument("--execution_space", type=str)
    args = parser.parse_args()

    if args.S != 2:
        print("only support S=2")
        exit(1)
    if args.U != 8:
        print("only support U=8")
        exit(1)
    if args.N < args.D:
        print("N must be larger or equal to D")
        exit(1)

    space = pk.ExecutionSpace.OpenMP
    if args.execution_space:
        space = pk.ExecutionSpace(args.execution_space)

    pk.set_default_space(space)

    N = args.N
    K = args.K
    D = args.D
    R = args.R
    U = args.U
    F = args.F
    scalar_size = 8

    policy = pk.RangePolicy(0, N)
    w = Benchmark_double_8(N, K, D, R, F, space)

    timer = pk.Timer()
    for r in range(R):
        pk.parallel_for(policy, w.benchmark)
        pk.fence()

    seconds = timer.seconds()

    num_bytes = 1.0 * N * K * R * (2 * scalar_size + 4) + N * R * scalar_size
    flops = 1.0 * N * K * R * (F * 2 * U + 2 * (U - 1))
    gather_ops = 1.0 * N * K * R * 2
    seconds = seconds
    print(
        f"SNKDRUF: {scalar_size/4} {N} {K} {D} {R} {U} {F} Time: {seconds} "
        + f"Bandwidth: {1.0 * num_bytes / seconds / (1024**3)} GiB/s GFlop/s: {1e-9 * flops / seconds} GGather/s: {1e-9 * gather_ops / seconds}"
    )


if __name__ == "__main__":
    run()
