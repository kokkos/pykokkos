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
class Benchmark:
    def __init__(
        self,
        indices: int,
        data: int,
        repeats: int,
        use_atomics: bool,
        space: pk.ExecutionSpace,
    ):
        xp = get_array_module(space)
        self.indices = xp.zeros(indices, dtype=np.int64)
        self.data = xp.zeros(data, dtype=np.int64)
        self.datum: pk.int64 = -1

    @pk.workunit
    def init_data(self, i: int):
        self.data[i] = 10101010101

    @pk.workunit
    def init_indices(self, i: int):
        self.indices[i] = 0

    @pk.workunit
    def run_gups_atomic(self, i: int):
        pk.atomic_fetch_xor(self.data, [self.indices[i]], self.datum)

    @pk.workunit
    def run_gups(self, i: int):
        self.data[self.indices[i]] ^= self.datum


def run() -> None:
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

    w = Benchmark(indices, data, repeats, use_atomics, space)
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
    pk.parallel_for(range_data, w.init_data)
    pk.parallel_for(range_indices, w.init_indices)

    print("Starting benchmarking...")

    timer = pk.Timer()
    for i in range(repeats):
        for i in range(indices):
            w.indices[i] = random.randrange(data)

        if use_atomics:
            pk.parallel_for(range_indices, w.run_gups_atomic)
        else:
            pk.parallel_for(range_indices, w.run_gups)

    gupsTime = timer.seconds()
    print(f"GUP/s Random: {1e-9 * repeats * indices / gupsTime}")
    print(w.data)


if __name__ == "__main__":
    run()
