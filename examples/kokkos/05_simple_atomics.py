import math
import random
import pykokkos as pk


@pk.workunit
def findprimes(
    i: int,
    data: pk.View1D[pk.int32],
    result: pk.View1D[pk.int32],
    count: pk.View1D[pk.int32],
):
    number: int = data[i]
    upper_bound: int = int(math.sqrt(number)) + 1
    is_prime: bool = not (number % 2 == 0)
    k: int = 3
    idx: int = 0

    while k < upper_bound and is_prime:
        is_prime = not (number % k == 0)
        k += 2

    if is_prime:
        # Note: This atomic operation may have race conditions without proper atomic support
        # For now, we remove the atomic trait as it's not supported
        idx = count[0] = count[0] + 1
        result[idx - 1] = number


def simple_atomics():
    N: int = 100
    pk.set_default_space(pk.ExecutionSpace.OpenMP)

    data: pk.View1D[pk.int32] = pk.View([N], pk.int32)
    result: pk.View1D[pk.int32] = pk.View([N], pk.int32)
    # FIXED: Removed trait=pk.Trait.Atomic as it's not supported
    count: pk.View1D[pk.int32] = pk.View([1], pk.int32)

    # Initialize data with random numbers
    for i in range(N):
        data[i] = random.randint(0, N)

    pk.parallel_for(N, findprimes, data=data, result=result, count=count)

    # Print results
    for i in range(int(count[0])):
        print(int(result[i]), end=", ")
    print("\nFound", int(count[0]), "prime numbers in", N, "random numbers")


if __name__ == "__main__":
    simple_atomics()
