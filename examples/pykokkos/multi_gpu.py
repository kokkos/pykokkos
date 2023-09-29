import pykokkos as pk

import numpy as np
import cupy as cp

pk.set_default_space(pk.Cuda)

size = 10000

pk.set_device_id(0)
cp_arr_0 = cp.arange(size).astype(np.int32)

pk.set_device_id(1)
cp_arr_1 = cp.arange(size).astype(np.int32)

print(cp_arr_0.device)
print(cp_arr_1.device)

@pk.workunit(cp_arr = pk.ViewTypeInfo(space=pk.CudaSpace))
def reduction_cp(i: int, acc: pk.Acc[int], cp_arr: pk.View1D[int]):
    acc += cp_arr[i]

pk.set_device_id(1)
cp_view_0 = pk.array(cp_arr_1)
result_0 = pk.parallel_reduce(pk.RangePolicy(pk.Cuda, 0, size), reduction_cp, cp_arr=cp_view_0)
print(result_0)

pk.set_device_id(0)
cp_view_1 = pk.array(cp_arr_0)
result_1 = pk.parallel_reduce(pk.RangePolicy(pk.Cuda, 0, size), reduction_cp, cp_arr=cp_view_1)

print(f"Reducing array 0: {result_0}")
print(f"Reducing array 1: {result_1}")
print(f"Sum: {result_0 + result_1}")

pk.set_device_id(0)
view_0 = pk.View((size,), dtype=int)

pk.set_device_id(1)
view_1 = pk.View((size,), dtype=int)

@pk.workunit
def init_view(i: int, view: pk.View1D[int]):
    view[i] = i

@pk.workunit
def reduce_view(i: int, acc: pk.Acc[int], view: pk.View1D[int]):
    acc += view[i]

pk.set_device_id(0)
pk.parallel_for(pk.RangePolicy(pk.Cuda, 0, size), init_view, view=view_0)
result_0 = pk.parallel_reduce(pk.RangePolicy(pk.Cuda, 0, size), reduce_view, view=view_0)

pk.set_device_id(1)
pk.parallel_for(pk.RangePolicy(pk.Cuda, 0, size), init_view, view=view_1)
result_1 = pk.parallel_reduce(pk.RangePolicy(pk.Cuda, 0, size), reduce_view, view=view_1)

print(f"Reducing view 0: {result_0}")
print(f"Reducing view 1: {result_1}")
print(f"Sum: {result_0 + result_1}")
