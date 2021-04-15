import argparse
import random
from typing import Tuple

import pykokkos as pk


@pk.workload
class Benchmark:
    def __init__(self, indices: int, data: int, repeats: int, use_atomics: bool):
        self.dataCount: int = data
        self.indicesCount: int = indices
        self.repeats: int = repeats
        self.use_atomics: bool = use_atomics

        print("Reports fastest timing per kernel")
        print("Creating Views...")
        print("Memory Sizes:")
        print(f"- Elements: {data} ({1e-6*data*8} MB)")
        print(f"- Indices: {indices} ({1e-6*indices*8} MB)")
        print(f"- Atomics: {'yes' if use_atomics else 'no'}")
        print(f"Benchmark kernels will be performed for {repeats} iterations")

        self.indices: pk.View1D[pk.int64] = pk.View([indices], pk.int64)
        self.data: pk.View1D[pk.int64] = pk.View([data], pk.int64)
        self.datum: pk.int64 = -1

        self.gupsTime: float = 0

    @pk.main
    def run(self):
        printf("Initializing Views...\n")
        pk.parallel_for(self.dataCount, self.init_data)
        pk.parallel_for(self.indicesCount, self.init_indices)

        printf("Starting benchmarking...\n")
        pk.fence()

        timer = pk.Timer()
        for i in range(self.repeats):
            # FIXME: randomize indices
            # for i in range(self.indicesCount):
            #     self.indices[i] = random.randrange(self.dataCount)

            if self.use_atomics:
                pk.parallel_for("gups", self.indicesCount, self.run_gups_atomic)
            else:
                pk.parallel_for("gups", self.indicesCount, self.run_gups)

            pk.fence()

        self.gupsTime = timer.seconds()

    @pk.workunit
    def init_data(self, i: int):
        self.data[i] = 10101010101

    @pk.workunit
    def init_indices(self, i: int):
        self.indices[i] = 0

    @pk.callback
    def results(self):
        print(f"GUP/s Random: {1e-9 * self.repeats * self.indicesCount / self.gupsTime}")
        print(self.data)

    @pk.workunit
    def run_gups_atomic(self, i: int):
        pk.atomic_fetch_xor(self.data, [self.indices[i]], self.datum)

    @pk.workunit
    def run_gups(self, i: int):
        self.data[self.indices[i]] ^= self.datum


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
    parser.add_argument("-space", "--execution_space", type=str)
    args = parser.parse_args()
    if args.indices:
        indices = args.indices
        indices = 2 ** indices
    if args.data:
        data = args.data
        data = 2 ** data
    if args.repeats:
        repeats = args.repeats
    use_atomics = args.atomics
    if args.execution_space:
        space = pk.ExecutionSpace(args.execution_space)

    pk.set_default_space(space)

    pk.execute(pk.get_default_space(), Benchmark(indices, data, repeats, use_atomics))
