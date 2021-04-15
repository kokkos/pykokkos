import argparse
import random
from typing import Tuple

import pykokkos as pk


@pk.workload
# use double type and unroll=8
class Benchmark_double_8:
    def __init__(self, N: int, K: int, R: int, D: int, F: int, T: int, S: int):
        self.N: int = N
        self.K: int = K
        self.R: int = R
        self.D: int = D
        self.F: int = F
        self.T: int = T
        self.S: int = S
        self.UNROLL: int = 8
        self.scalar_size: int = 8

        self.A: pk.View3D[pk.double] = pk.View([N, K, D], pk.double)
        self.B: pk.View3D[pk.double] = pk.View([N, K, D], pk.double)
        self.C: pk.View3D[pk.double] = pk.View([N, K, D], pk.double)

        self.A.fill(1.5)
        self.B.fill(2.5)
        self.C.fill(3.5)
        
        self.seconds: float = 0

    @pk.main
    def run(self):
        timer = pk.Timer()
        pk.parallel_for("bytes_and_flops", pk.TeamPolicy(self.N, self.T), self.benchmark)
        pk.fence()
        self.seconds = timer.seconds()

    @pk.callback
    def results(self):
        N = self.N
        K = self.K
        R = self.R
        num_bytes = 1.0 * N * K * R * 3 * self.scalar_size
        flops = 1.0 * N * K * R * (self.F * 2 * self.UNROLL + 2 * (self.UNROLL - 1))
        seconds = self.seconds
        print(f"NKRUFTS: {self.N} {self.K} {self.R} {self.UNROLL} {self.F} {self.T} {self.S} Time: {seconds} " +
                f"Bandwidth: {1.0 * num_bytes / seconds / (1024**3)} GiB/s GFlop/s: {1e-9 * flops / seconds}")

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

if __name__ == "__main__":
    # example args
    # Bandwidth Bound : 2 100000 1024 1 1 8 1 256 0 
    # Cache Bound     : 2 100000 1024 64 1 8 1 512 0 
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
    parser.add_argument("-space", "--execution_space", type=str)
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
    
    pk.set_default_space(space)

    args.N = 2 ** args.N

    pk.execute(pk.get_default_space(), Benchmark_double_8(args.N, args.K, args.R, args.D, args.F, args.T, args.S))
