import math

import pykokkos as pk


@pk.workunit
def my_calculation(i: int, a: pk.View1D[pk.int32], N: int):
    pk.printf("Running index %d\n", i)
    a[i] += (
        math.cos(a[i]) + 2**i - math.pi / math.fabs(a[(i + 1) % N])
    )


if __name__ == "__main__":
    n = 10
    N = n
    a = pk.View([N], pk.int32)
    
    # Initialize view
    for i in range(N):
        a[i] = math.sqrt(math.tau)
    
    print("Initialized view:", a)
    
    pk.parallel_for(N, my_calculation, a=a, N=N)
    
    print("Results: ", a)