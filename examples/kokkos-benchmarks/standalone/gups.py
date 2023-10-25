import argparse
import random
from typing import Tuple

import pykokkos as pk

@pk.workunit
def init_data(i, data):
    data[i] = 10101010101

@pk.workunit
def init_indices(i, indices):
    indices[i] = 0

@pk.workunit
def run_gups_atomic(i, data, indices, datum):
    pk.atomic_fetch_xor(data, [indices[i]], datum)

@pk.workunit
def run_gups(i, data, indices, datum):
    data[indices[i]] ^= datum

if __name__ == "__main__":
    random.seed(1010101)

    indices = 8192
    data = 33554432
    repeats = 10
    space = pk.ExecutionSpace.OpenMP

    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int)
    parser.add_argument("--data", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--atomics", action="store_true")
    parser.add_argument("--execution_space", type=str)
    args = parser.parse_args()
    if args.indices:
        indices = args.indices
    if args.data:
        data = args.data
    if args.repeats:
        repeats = args.repeats
    use_atomics = args.atomics
    if args.execution_space:
        space = pk.ExecutionSpace(args.execution_space)

    pk.set_default_space(space)

    indices_view: pk.View1D[pk.int64] = pk.View([indices], pk.int64)
    data_view: pk.View1D[pk.int64] = pk.View([data], pk.int64)
    datum: pk.int64 = -1

    range_indices = pk.RangePolicy(0, indices)
    range_data = pk.RangePolicy(0, data)

    print("Reports fastest timing per kernel")
    print("Creating Views...")
    print("Memory Sizes:")
    print(f"- Elements: {data} ({1e-6*data*8} MB)")
    print(f"- Indices: {indices} ({1e-6*indices*8} MB)")
    print(f"- Atomics: {'yes' if use_atomics else 'no'}")
    print(f"Benchmark kernels will be performed for {repeats} iterations")

    print("Initializing Views...")
    pk.parallel_for(range_data, init_data, data=data_view)
    pk.parallel_for(range_indices, init_indices, indices=indices_view)

    print("Starting benchmarking...")

    timer = pk.Timer()
    for i in range(repeats):
        for i in range(indices):
            indices_view[i] = random.randrange(data)

        if use_atomics:
            pk.parallel_for(range_indices, run_gups_atomic, data=data_view, 
                    indices=indices_view, datum=datum)
        else:
            pk.parallel_for(range_indices, run_gups, data=data_view, 
                    indices=indices_view, datum=datum)

    gupsTime = timer.seconds()
    print(f"GUP/s Random: {1e-9 * repeats * indices / gupsTime}")


