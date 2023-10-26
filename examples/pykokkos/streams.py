import pykokkos as pk

import cupy as cp

@pk.workunit
def print_stream(i, x):
    if x == 0:
        pk.printf("Default stream\n")
    elif x == 1:
        pk.printf("Stream 1\n")
    elif x == 2:
        pk.printf("Stream 2\n")

if __name__ == "__main__":
    space = pk.Cuda

    s1 = cp.cuda.Stream()
    s2 = cp.cuda.Stream()

    instance1 = pk.ExecutionSpaceInstance(space, s1)
    instance2 = pk.ExecutionSpaceInstance(space, s2)

    for i in range(10):
        print(f"Iteration: {i}")
        pk.parallel_for(pk.RangePolicy(space, 0, 10), print_stream, x=0)
        pk.parallel_for(pk.RangePolicy(instance1, 0, 10), print_stream, x=1)
        pk.parallel_for(pk.RangePolicy(instance2, 0, 10), print_stream, x=2)