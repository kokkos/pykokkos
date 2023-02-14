import cupy as cp
import numpy as np
import pykokkos as pk

@pk.functor
class Workload:
    def __init__(self, s: int):
        self.size: pk.int = s

    @pk.workunit
    def print(self, i: int):
        printf("size (on thread): %d\n", self.size)

def run() -> None:
    s: int = 500
    w = Workload(s)
    
    p = pk.RangePolicy(pk.get_default_space(), 0, 1)

    pk.parallel_for(p, w.print)
    print("Old size (from parent): ", w.size)

    print("Updating to: ", 10)
    w.size = 10

    pk.parallel_for(p, w.print)
    print("Updated size (from parent): ", w.size)

if __name__ == "__main__":
    run()
