import argparse
import random
from typing import Tuple

import pykokkos as pk


# use double type and unroll=8
@pk.workunit
def benchmark(i, K, F, A_view, B_view, C_view, connectivity):
    c: pk.double = 0.0
    for jj in range(K):
        j: int = connectivity[i][jj]
        a1: pk.double = A_view[j]
        b: pk.double = B_view[j]
        a2: pk.double = a1 * 1.3
        a3: pk.double = a2 * 1.1
        a4: pk.double = a3 * 1.1
        a5: pk.double = a4 * 1.3
        a6: pk.double = a5 * 1.1
        a7: pk.double = a6 * 1.1
        a8: pk.double = a7 * 1.1
        
        for f in range(F):
            a1 += b * a1
            a2 += b * a2
            a3 += b * a3
            a4 += b * a4
            a5 += b * a5
            a6 += b * a6
            a7 += b * a7
            a8 += b * a8

        c += a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8

    C_view[i] = c


if __name__ == "__main__":
    # example args 2 100000 32 512 1000 8 8
    # NOTE S and U are hard coded to double and 8 because otherwise we would have a lot of duplicates
    parser = argparse.ArgumentParser()
    parser.add_argument("S", type=int, help="Scalar Type Size (1==float, 2==double, 4=complex<double>)")
    parser.add_argument("N", type=int, help="Number of Entities")
    parser.add_argument("K", type=int, help="Number of things to gather per entity")
    parser.add_argument("D", type=int, help="Max distance of gathered things of an entity")
    parser.add_argument("R", type=int, help="how often to loop through the K dimension with each team")
    parser.add_argument("U", type=int, help="how many independent flops to do per load")
    parser.add_argument("F", type=int, help="how many times to repeat the U unrolled operations before reading next element")
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

    connectivity: pk.View2D[int] = pk.View([N, K], int)
    A: pk.View1D[pk.double] = pk.View([N], pk.double)
    B: pk.View1D[pk.double] = pk.View([N], pk.double)
    C: pk.View1D[pk.double] = pk.View([N], pk.double)

    A.fill(1.5)
    B.fill(2.0)
    
    random.seed(12313)
    for i in range(N):
        for jj in range(K):
            connectivity[i][jj] = (random.randrange(D)+i-D/2+N) % N

    policy = pk.RangePolicy(0, N)

    timer = pk.Timer()
    for r in range(R):
        pk.parallel_for(policy, benchmark, K=K, F=F, A_view=A, B_view=B, C_view=C, connectivity=connectivity)
        pk.fence()

    seconds = timer.seconds()

    num_bytes = 1.0 * N * K * R * (2 * scalar_size + 4) + N * R * scalar_size
    flops = 1.0 * N  * K * R * (F * 2 * U + 2 * (U - 1))
    gather_ops = 1.0 * N * K * R * 2
    seconds = seconds
    print(f"SNKDRUF: {scalar_size/4} {N} {K} {D} {R} {U} {F} Time: {seconds} " +
            f"Bandwidth: {1.0 * num_bytes / seconds / (1024**3)} GiB/s GFlop/s: {1e-9 * flops / seconds} GGather/s: {1e-9 * gather_ops / seconds}")

