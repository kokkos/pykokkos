import argparse
import random
from typing import Tuple

import pykokkos as pk


@pk.functor
# use double type and unroll=8
class Benchmark_double_8:
    def __init__(self, N: int, K: int, R: int, D: int, F: int, T: int, S: int):
        self.K: int = K
        self.R: int = R
        self.F: int = F

        self.A: pk.View3D[pk.double] = pk.View([N, K, D], pk.double)
        self.B: pk.View3D[pk.double] = pk.View([N, K, D], pk.double)
        self.C: pk.View3D[pk.double] = pk.View([N, K, D], pk.double)

        self.A.fill(1.5)
        self.B.fill(2.5)
        self.C.fill(3.5)

    @pk.workunit
    def benchmark(self, team: pk.TeamMember):
        n: int = team.league_rank()
        for r in range(self.R):
            def team_for(i: int):
                a1: pk.double = self.A[n][i][0]
                b: pk.double = self.B[n][i][0]
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

                self.C[n][i][0] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8


            pk.parallel_for(pk.TeamThreadRange(team, self.K), team_for)

def run() -> None:
    # example args
    # Bandwidth Bound : 2 100000 1024 1 1 1 8 256 0
    # Cache Bound     : 2 100000 1024 64 1 1 8 512 0
    # Compute Bound   : 2 100000 1024 1 1 8 64 256 0
    # Load Slots Used : 2 20000 256 32 16 8 1 256 0
    # Inefficient Load: 2 20000 256 32 2 8 1 256 0
    # NOTE P and U are hard coded to double and 8 because otherwise we would have a lot of duplicates
    parser = argparse.ArgumentParser()
    parser.add_argument("P", type=int, help="Precision (1==float, 2==double)")
    parser.add_argument("N", type=int, help="N dimensions of the 2D array to allocate")
    parser.add_argument("K", type=int, help="K dimension of the 2D array to allocate")
    parser.add_argument("R", type=int, help="how often to loop through the K dimension with each team")
    parser.add_argument("D", type=int, help="distance between loaded elements (stride)")
    parser.add_argument("U", type=int, help="how many independent flops to do per load")
    parser.add_argument("F", type=int, help="how many times to repeat the U unrolled operations before reading next element")
    parser.add_argument("T", type=int, help="team size")
    # NOTE: S ignored
    parser.add_argument("S", type=int, help="shared memory per team (used to control occupancy on GPUs)")
    parser.add_argument("--execution_space", type=str)
    args = parser.parse_args()

    if args.P != 2:
        print("only support P=2")
        exit(1)
    if args.U != 8:
        print("only support U=8")
        exit(1)
    if args.D not in [1, 2, 4, 8, 16, 32]:
        print("D must be one of 1, 2, 4, 8, 16, 32")
        exit(1)
    if args.S != 0:
        print("S must be 0 (shared scratch memory not supported)")
        exit(1)

    space = pk.ExecutionSpace.OpenMP
    if args.execution_space:
        space = pk.ExecutionSpace(args.execution_space)

    N = args.N
    K = args.K
    R = args.R
    U = args.U
    F = args.F
    T = args.T
    S = args.S
    scalar_size = 8

    pk.set_default_space(space)

    r = pk.TeamPolicy(N, T)
    w = Benchmark_double_8(N, K, R, args.D, F, T, S)

    timer = pk.Timer()
    pk.parallel_for(r, w.benchmark)
    seconds = timer.seconds()

    num_bytes = 1.0 * N * K * R * 3 * scalar_size
    flops = 1.0 * N * K * R * (F * 2 * U + 2 * (U - 1))
    print(f"NKRUFTS: {N} {K} {R} {U} {F} {T} {S} Time: {seconds} " +
            f"Bandwidth: {1.0 * num_bytes / seconds / (1024**3)} GiB/s GFlop/s: {1e-9 * flops / seconds}")
    print(w.C)


if __name__ == "__main__":
    run()
