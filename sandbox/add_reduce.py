import cupy as cp
import numpy as np
import pykokkos as pk

@pk.functor
class Workload:
    def __init__(self, s: int, fill: bool):
        # init variable list
        # is this allocateing space for them on the XPU?
        self.size: pk.int = s
        self.s: pk.View1D[int] = pk.View([1], int)
        self.arr: pk.View1D[int] = pk.View([self.size], int)

        # init initial array size
        self.s[0] = s

        # init array
        if fill:
            for i in range(self.size):
                self.arr[i] = i
        else:
            self.arr.fill(1)

    # add second half of current array size to first half of the array
    @pk.workunit
    def reduce_add(self, i: int):
        size: int = self.s[0]
        indice: int = i + size/2 + size%2
        if indice < size: 
            arr[i] += arr[indice]
            arr[indice] = 0

    # print array value from thread (at index == thread)
    @pk.workunit
    def print(self, i: int):
        printf("arr[%d] = %d\n", i, self.arr[i])

    # update array size, only run on one string... this is a hack
    @pk.workunit
    def update(self, i: int):
        size: int = self.s[0]/2 + self.s[0]%2
        self.s[0] = size
        printf("size = %d\n", self.s[0])

def run() -> None:
    # workspace init variables    
    s: int = 500
    fill: bool = True
    
    pk.set_default_space(pk.ExecutionSpace.OpenMP)
    #pk.set_default_space(pk.ExecutionSpace.Cuda)
    w = Workload(s, fill)
    
    # while the 'size' of the array is greater than 1
    # keep running the reduction and update
    # and print
    while(s > 1):
        thread_count : int = int(s/2) + s%2
        p = pk.RangePolicy(pk.get_default_space(), 0, thread_count)
        print(w.arr)
        pk.parallel_for(p, w.reduce_add)
        print("\n")
        pk.parallel_for(1, w.update)
        s = thread_count

    print(w.arr, "\n")

    # check answer
    answer: int = w.size
    if fill:
        _s: int = w.size-1
        answer = int(_s/2)*w.size + (_s%2)*(w.size/2)

    if answer == w.arr[0]:
        print(answer, " == ", w.arr[0])
    else:
        print(answer, " != ", w.arr[0])
    
if __name__ == "__main__":
    run()
