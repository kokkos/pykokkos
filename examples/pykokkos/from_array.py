import cupy as cp
import numpy as np
import pykokkos as pk

@pk.workunit(np_arr = pk.ViewTypeInfo(space=pk.HostSpace))
def addition_np(i: int, np_arr: pk.View2D[int]):
    np_arr[i][0] += 1*i
    np_arr[i][1] += 2*i

@pk.workunit(cp_arr = pk.ViewTypeInfo(space=pk.CudaSpace, layout=pk.LayoutRight))
def addition_cp(i: int, cp_arr: pk.View2D[int]):
    cp_arr[i][0] += 1*i
    cp_arr[i][1] += 2*i

size = 10

np_arr = np.zeros((size, 2)).astype(np.int32)
cp_arr = cp.zeros((size, 2)).astype(np.int32)
list_arr = [np.array([0, 0], dtype=np.int32)] * size

print(cp_arr.flags)

print(f"before {np_arr=}")
print(f"before {cp_arr=}")
print(f"before {list_arr=}")

np_view = pk.array(np_arr)
cp_view = pk.array(cp_arr)
list_view = pk.array(list_arr)

pk.parallel_for(pk.RangePolicy(pk.OpenMP, 0, size), addition_np, np_arr=np_view)
pk.parallel_for(pk.RangePolicy(pk.Cuda, 0, size), addition_cp, cp_arr=cp_view)
pk.parallel_for(pk.RangePolicy(pk.OpenMP, 0, size), addition_np, np_arr=list_view)

print(f"after {np_arr=}")
print(f"after {cp_arr=}")
list_arr = [[(list_view[i][j], type(list_view[i][j])) for j in range(2)] for i in range(size)]
print(f"after {list_arr=}")
