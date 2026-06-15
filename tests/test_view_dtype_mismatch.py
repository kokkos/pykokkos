import numpy as np
import pytest

import pykokkos as pk
from pykokkos.interface.parallel_dispatch import convert_arrays


@pk.workunit
def scale_int32(i: int, arr: pk.View1D[pk.int32]) -> None:
    arr[i] = arr[i] * 2


@pk.workunit
def scale_int(i: int, arr: pk.View1D[int]) -> None:
    arr[i] = arr[i] * 2


@pk.workunit
def scale_float32(i: int, arr: pk.View1D[pk.float]) -> None:
    arr[i] = arr[i] * 2


@pk.workunit
def scale_float(i: int, arr: pk.View1D[float]) -> None:
    arr[i] = arr[i] * 2


@pk.workunit
def scale_unannotated(i: int, arr) -> None:
    arr[i] = arr[i] * 2


def openmp_policy(size: int):
    return pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, size)


def convert_for_openmp(workunit, **kwargs):
    convert_arrays(kwargs, workunit, pk.ExecutionSpace.OpenMP)
    return kwargs


def test_numpy_int64_rejected_for_int32_view():
    arr = pk.array(np.ones(8, dtype=int))

    with pytest.raises(TypeError, match="expects a View with dtype int32"):
        pk.parallel_for(openmp_policy(8), scale_int32, arr=arr)


def test_numpy_array_int64_rejected_for_int32_view():
    arr = np.ones(8, dtype=int)

    with pytest.raises(TypeError, match="expects a View with dtype int32"):
        pk.parallel_for(openmp_policy(8), scale_int32, arr=arr)


def test_numpy_int64_rejected_for_bare_int_view():
    arr = pk.array(np.ones(8, dtype=int))

    with pytest.raises(TypeError, match="expects a View with dtype int32"):
        pk.parallel_for(openmp_policy(8), scale_int, arr=arr)


def test_numpy_int32_accepted_for_int32_view():
    kwargs = convert_for_openmp(scale_int32, arr=np.ones(8, dtype=np.int32))

    assert kwargs["arr"].dtype is pk.int32


def test_numpy_float64_rejected_for_float32_view():
    arr = pk.array(np.ones(8, dtype=np.float64))

    with pytest.raises(TypeError, match="expects a View with dtype float32"):
        pk.parallel_for(openmp_policy(8), scale_float32, arr=arr)


def test_numpy_float32_rejected_for_bare_float_view():
    arr = pk.array(np.ones(8, dtype=np.float32))

    with pytest.raises(TypeError, match="expects a View with dtype float64"):
        pk.parallel_for(openmp_policy(8), scale_float, arr=arr)


def test_unannotated_numpy_int64_view_is_inferred_as_int64():
    kwargs = convert_for_openmp(scale_unannotated, arr=np.ones(8, dtype=int))

    assert kwargs["arr"].dtype is pk.int64
