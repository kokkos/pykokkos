import argparse
import random
from typing import Tuple

import numpy as np
import pykokkos as pk


@pk.workload
# use double type and unroll=8
class Benchmark_double_8:
    def __init__(self, N: int, K: int, D: int, R: int, F: int):
        self.N: int = N
        self.K: int = K
        self.D: int = D
        self.R: int = R
        self.F: int = F
        self.UNROLL: int = 8
        self.scalar_size: int = 8

        self.connectivity: pk.View2D[int] = pk.View([N, K], int)
        self.A: pk.View1D[pk.double] = pk.View([N], pk.double)
        self.B: pk.View1D[pk.double] = pk.View([N], pk.double)
        self.C: pk.View1D[pk.double] = pk.View([N], pk.double)

        self.A.fill(1.5)
        self.B.fill(2.0)

        self.connectivity.data = np.random.rand(N, K) * N
        #TODO use kokkos to init in parallel
        # random.seed(12313)
        # for i in range(N):
        #     for jj in range(K):
        #         self.connectivity[i][jj] = (random.randrange(D)+i-D/2+N) % N

        self.seconds: float = 0

    @pk.main
    def run(self):
        timer = pk.Timer()
        for r in range(self.R):
            pk.parallel_for("gather", self.N, self.benchmark)
            pk.fence()

        self.seconds = timer.seconds()

    @pk.callback
    def results(self):
        N = self.N
        K = self.K
        R = self.R
        num_bytes = 1.0 * N * K * R * (2 * self.scalar_size + 4) + N * R * self.scalar_size
        flops = 1.0 * N  * K * R * (self.F * 2 * self.UNROLL + 2 * (self.UNROLL - 1))
        gather_ops = 1.0 * self.N * self.K * self.R * 2
        seconds = self.seconds
        print(f"SNKDRUF: {self.scalar_size/4} {self.N} {self.K} {self.D} {self.R} {self.UNROLL} {self.F} Time: {seconds} " +
                f"Bandwidth: {1.0 * num_bytes / seconds / (1024**3)} GiB/s GFlop/s: {1e-9 * flops / seconds} GGather/s: {1e-9 * gather_ops / seconds}")

    @pk.workunit
    def benchmark(self, i: int):
        c: pk.double = 0.0
        for jj in range(self.K):
            j: int = self.connectivity[i][jj]
            a1: pk.double = A[j]
            b: pk.double = B[j]
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


if __name__ == "__main__":
    # example args 2 100000 32 512 1000 8 8
    # NOTE S and U are hard coded to double and 8 because otherwise we would have a lot of duplicates
    parser = argparse.ArgumentParser()
    parser.add_argument("S", type=int, help="Scalar Type Size (1==float, 2==double, 4==complex<double>)")
    parser.add_argument("N", type=int, help="Number of Entities")
    parser.add_argument("K", type=int, help="Number of things to gather per entity")
    parser.add_argument("D", type=int, help="Max distance of gathered things of an entity")
    parser.add_argument("R", type=int, help="how often to loop through the K dimension with each team")
    parser.add_argument("U", type=int, help="how many independent flops to do per load")
    parser.add_argument("F", type=int, help="how many times to repeat the U unrolled operations before reading next element")
    parser.add_argument("-space", "--execution_space", type=str)
    args = parser.parse_args()

    if args.S != 2:
        print("only support S=2")
        exit(1)
    if args.U != 8:
        print("only support U=8")
        exit(1)
    if 2 ** args.N < args.D:
        print("N must be larger or equal to D")
        exit(1)

    space = pk.ExecutionSpace.OpenMP
    if args.execution_space:
        space = pk.ExecutionSpace(args.execution_space)
    
    pk.set_default_space(space)

    n = 2 ** args.N

    pk.execute(pk.get_default_space(), Benchmark_double_8(n, args.K, args.D, args.R, args.F))
