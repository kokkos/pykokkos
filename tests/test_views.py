from pykokkos.interface.data_types import int32
import pytest
import unittest

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pykokkos as pk


class MyView1D(pk.View):
    def __init__(self, x: int, dtype: pk.DataTypeClass = pk.int32):
        super().__init__([x], dtype)


class MyView2D(pk.View):
    def __init__(self, x: int, y: int, dtype: pk.DataTypeClass = pk.int32):
        super().__init__([x, y], dtype)


class MyView3D(pk.View):
    def __init__(self, x: int, y: int, z: int, dtype: pk.DataTypeClass = pk.int32):
        super().__init__([x, y, z], dtype)


# Tests for correctness of pk.View
@pk.functor(
        subview1D=pk.ViewTypeInfo(trait=pk.Unmanaged),
        subview2D=pk.ViewTypeInfo(trait=pk.Unmanaged),
        subview3D=pk.ViewTypeInfo(trait=pk.Unmanaged),
        np_view=pk.ViewTypeInfo(space=pk.HostSpace),
        cp_view=pk.ViewTypeInfo(space=pk.CudaSpace, layout=pk.LayoutRight))
class ViewsTestFunctor:
    def __init__(self, threads: int, i_1: int, i_2: int, i_3: int, i_4: int):
        self.threads: int = threads
        self.i_1: int = i_1
        self.i_2: int = i_2
        self.i_3: int = i_3
        self.i_4: int = i_4

        self.view1D: pk.View1D[pk.int32] = pk.View([i_1], pk.int32)
        self.view2D: pk.View2D[pk.int32] = pk.View([i_1, i_2], pk.int32)
        self.view3D: pk.View3D[pk.int32] = pk.View([i_1, i_2, i_3], pk.int32)

        self.myView1D: pk.View1D[int] = MyView1D(i_1, int)
        self.myView2D: pk.View2D[int] = MyView2D(i_1, i_2, int)
        self.myView3D: pk.View3D[int] = MyView3D(i_1, i_2, i_3, int)

        # Views needed for subviews
        self.altView1D: pk.View1D[pk.int32] = pk.View([i_1], pk.int32)
        self.altView2D: pk.View2D[pk.int32] = pk.View([i_1, i_2], pk.int32)
        self.altView3D: pk.View3D[pk.int32] = pk.View([i_1, i_2, i_3], pk.int32)

        for i in range(i_1):
            self.view1D[i] = i_4
            self.myView1D[i] = i_4
            self.altView1D[i] = i_4

            for j in range(i_2):
                self.view2D[i][j] = i_4
                self.myView2D[i][j] = i_4
                self.altView2D[i][j] = i_4

                for k in range(i_3):
                    self.view3D[i][j][k] = i_4
                    self.myView3D[i][j][k] = i_4
                    self.altView3D[i][j][k] = i_4

        self.dynamicView1D: pk.View1D[pk.int32] = pk.View([i_1], pk.int32)
        self.dynamicView1D.resize(0, i_2)

        for i in range(i_2):
            self.dynamicView1D[i] = i_4

        self.dynamicView2D: pk.View2D[pk.int32] = pk.View([i_1, i_2], pk.int32)
        self.dynamicView2D.resize(0, i_2)
        self.dynamicView2D.resize(1, i_1)

        for i in range(i_2):
            for j in range(i_1):
                self.dynamicView2D[i, j] = i_4

        self.subview1D: pk.View1D[pk.int32] = self.altView1D[:]
        self.subview2D: pk.View2D[pk.int32] = self.altView2D[:, :i_2 // 2]
        self.subview3D: pk.View3D[pk.int32] = self.altView3D[:, :i_2 // 2, i_3 // 2: i_3]

        np_arr = np.zeros((threads, 2)).astype(np.int32)
        cp_arr = cp.zeros((threads, 2)).astype(np.int32)
        list_arr = [np.array([0, 0], dtype=np.int32)] * threads

        self.np_view: pk.View2D[int] = pk.array(np_arr)
        self.cp_view: pk.View2D[int] = pk.array(cp_arr)
        self.list_view: pk.View2D[int] = pk.array(list_arr)

    @pk.workunit
    def v1d(self, tid: int) -> None:
        self.view1D[tid] += tid
        self.myView1D[tid] += tid

    @pk.workunit
    def v2d(self, tid: int) -> None:
        for j in range(self.i_2):
            self.view2D[tid][j] += tid + j
            self.myView2D[tid][j] += tid + j

    @pk.workunit
    def v3d(self, tid: int) -> None:
        for j in range(self.i_2):
            for k in range(self.i_3):
                self.view3D[tid][j][k] += tid + j + k
                self.myView3D[tid][j][k] += tid + j + k

    @pk.workunit
    def sv1d(self, tid: int) -> None:
        self.subview1D[tid] += tid

    @pk.workunit
    def sv2d(self, tid: int) -> None:
        for j in range(self.i_2 // 2):
            self.subview2D[tid][j] += tid + j

    @pk.workunit
    def sv3d(self, tid: int) -> None:
        for j in range(self.i_2 // 2):
            for k in range(self.i_3 // 2, self.i_3):
                self.subview3D[tid][j][k - self.i_3 // 2] += tid + j + k

    @pk.workunit
    def dynamic1D(self, tid: int, acc: pk.Acc[pk.int32]) -> None:
        acc += self.dynamicView1D[tid]

    @pk.workunit
    def dynamic2D(self, tid: int, acc: pk.Acc[pk.int32]) -> None:
        temp: int = 0
        for j in range(self.i_1):
            temp += self.dynamicView2D[tid][j]

        acc += temp

    @pk.workunit
    def extent(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        first_dim: int = self.view3D.extent(0)
        second_dim: int = self.view3D.extent(1)
        third_dim: int = self.view3D.extent(2)

        acc += first_dim + second_dim + third_dim

@pk.functor
class RealViewTestFunctor:
    def __init__(self, view: pk.View1D[pk.real]):
        self.view: pk.View1D[pk.real] = view

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = tid

@pk.workload
class RealViewTestWorkload:
    def __init__(self, threads: int, view: pk.View1D[pk.real]):
        self.threads: int = threads
        self.view: pk.View1D[pk.real] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = tid

@pk.workunit(np_arr = pk.ViewTypeInfo(space=pk.HostSpace))
def addition_np(tid: int, np_arr: pk.View2D[int]) -> None:
    np_arr[tid][0] += 1
    np_arr[tid][1] += 2

@pk.workunit(cp_arr = pk.ViewTypeInfo(space=pk.CudaSpace, layout=pk.LayoutRight))
def addition_cp(tid: int, cp_arr: pk.View2D[int]) -> None:
    cp_arr[tid][0] += 1
    cp_arr[tid][1] += 2

class TestViews(unittest.TestCase):
    def setUp(self):
        self.threads: int = 10
        self.i_1: int = 10
        self.i_2: int = 15
        self.i_3: int = 20
        self.i_4: int = 10

        self.functor = ViewsTestFunctor(self.threads, self.i_1, self.i_2, self.i_3, self.i_4)
        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)

    @pytest.mark.skipif(not HAS_CUDA,
                        reason="CUDA/cupy not available")
    def test_v1d(self):
        pk.parallel_for(self.range_policy, self.functor.v1d)

        for i in range(self.i_1):
            expected_result: int = self.i_4 + i
            self.assertEqual(expected_result, self.functor.view1D[i])
            self.assertEqual(expected_result, self.functor.myView1D[i])

    @pytest.mark.skipif(not HAS_CUDA,
                        reason="CUDA/cupy not available")
    def test_v2d(self):
        pk.parallel_for(self.range_policy, self.functor.v2d)

        for i in range(self.i_1):
            for j in range(self.i_2):
                expected_result: int = self.i_4 + i + j
                self.assertEqual(expected_result, self.functor.view2D[i][j])
                self.assertEqual(expected_result, self.functor.myView2D[i][j])

    @pytest.mark.skipif(not HAS_CUDA,
                        reason="CUDA/cupy not available")
    def test_v3d(self):
        pk.parallel_for(self.range_policy, self.functor.v3d)

        for i in range(self.i_1):
            for j in range(self.i_2):
                for k in range(self.i_3):
                    expected_result: int = self.i_4 + i + j + k
                    self.assertEqual(
                        expected_result, self.functor.view3D[i][j][k])
                    self.assertEqual(
                        expected_result, self.functor.myView3D[i][j][k])

    @pytest.mark.skipif(not HAS_CUDA,
                        reason="CUDA/cupy not available")
    def test_sv1d(self):
        pk.parallel_for(self.range_policy, self.functor.sv1d)

        for i in range(self.i_1):
            expected_result: int = self.i_4 + i
            self.assertEqual(expected_result, self.functor.subview1D[i])

    # def test_sv2d(self):
    #     pk.parallel_for(self.range_policy, self.functor.sv2d)
    #     print(self.functor.subview2D)
    #     for i in range(self.i_1):
    #         for j in range(self.i_2 // 2):
    #             expected_result: int = self.i_4 + i + j
    #             self.assertEqual(expected_result, self.functor.subview2D[i][j])

    # def test_sv3d(self):
    #     pk.parallel_for(self.range_policy, self.functor.sv3d)

    #     for i in range(self.i_1):
    #         for j in range(self.i_2 // 2):
    #             for k in range(self.i_3 // 2, self.i_3):
    #                 expected_result: int = self.i_4 + i + j + k
    #                 self.assertEqual(expected_result, self.functor.subview3D[i][j][k - self.i_3 // 2])

    @pytest.mark.skipif(not HAS_CUDA,
                        reason="CUDA/cupy not available")
    def test_dynamic1D(self):
        expected_result: int = self.i_4 * self.i_2
        result: int = pk.parallel_reduce(pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.i_2), self.functor.dynamic1D)

        self.assertEqual(expected_result, result)

    @pytest.mark.skipif(not HAS_CUDA,
                        reason="CUDA/cupy not available")
    def test_dynamic2D(self):
        expected_result: int = self.i_4 * self.i_1 * self.i_2
        result: int = pk.parallel_reduce(pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.i_2), self.functor.dynamic2D)

        self.assertEqual(expected_result, result)

    @pytest.mark.skipif(not HAS_CUDA,
                        reason="CUDA/cupy not available")
    def test_extent(self):
        expected_result: int = (self.i_1 + self.i_2 + self.i_3) * self.threads
        result: int = pk.parallel_reduce(self.range_policy, self.functor.extent)

        self.assertEqual(expected_result, result)

    @pytest.mark.skipif(not HAS_CUDA,
                        reason="CUDA/cupy not available")
    def test_arrays(self):
        np_arr = np.zeros((self.threads, 2)).astype(np.int32)
        cp_arr = cp.zeros((self.threads, 2)).astype(np.int32)
        list_arr = [np.array([0, 0], dtype=np.int32)] * self.threads

        np_view = pk.array(np_arr)
        cp_view = pk.array(cp_arr)
        list_view = pk.array(list_arr)

        pk.parallel_for(pk.RangePolicy(pk.OpenMP, 0, self.threads), addition_np, np_arr=np_view)
        pk.parallel_for(pk.RangePolicy(pk.Cuda, 0, self.threads), addition_cp, cp_arr=cp_view)
        pk.parallel_for(pk.RangePolicy(pk.OpenMP, 0, self.threads), addition_np, np_arr=list_view)

        list_arr = [[(list_view[i][j]) for j in range(2)] for i in range(self.threads)]

        for i in range(self.threads):
            self.assertEqual(1, np_arr[i][0])
            self.assertEqual(2, np_arr[i][1])
            self.assertEqual(1, np_view[i][0])
            self.assertEqual(2, np_view[i][1])

            self.assertEqual(1, cp_arr[i][0])
            self.assertEqual(2, cp_arr[i][1])

            self.assertEqual(1, list_view[i][0])
            self.assertEqual(2, list_view[i][1])
            self.assertEqual(1, list_arr[i][0])
            self.assertEqual(2, list_arr[i][1])



    @pytest.mark.skipif(not HAS_CUDA,
                        reason="CUDA/cupy not available")
    @pytest.mark.xfail(HAS_CUDA,
                       reason="bool not supported with CUDA/GPUs yet")
    def test_real(self):
        pk.set_default_precision(pk.int32)
        view: pk.View1d = pk.View([self.threads])

        self.assertTrue(view.dtype is pk.DataType.int32)
        self.assertTrue(pk.View._get_dtype_name(str(type(view.array))) == "int32")

        f = RealViewTestFunctor(view)
        w = RealViewTestWorkload(self.threads, view)
        pk.parallel_for(self.threads, f.pfor)
        pk.execute(pk.ExecutionSpace.Default, w)

        view.set_precision(pk.float)

        self.assertTrue(view.dtype is pk.DataType.float)
        self.assertTrue(pk.View._get_dtype_name(str(type(view.array))) == "float32")
        pk.parallel_for(self.threads, f.pfor)
        pk.execute(pk.ExecutionSpace.Default, w)


@pytest.mark.parametrize("input_arr, view_dims, view_type", [
    (np.arange(10), [10], pk.View1D),
    (np.arange(50).reshape(10, 5), [10, 5], pk.View2D),
    (np.arange(500).reshape(10, 10, 5), [10, 10, 5], pk.View3D),
    ])
def test_sizes(input_arr, view_dims, view_type):
    # regression test for gh-31
    expected_size = input_arr.size
    view: view_type = pk.View(view_dims)
    view[:] = input_arr
    assert view.size == expected_size


@pytest.mark.parametrize("const", [pk.e, pk.pi, pk.inf, pk.nan])
@pytest.mark.parametrize("pk_dtype, np_dtype", [
    (None, None),
    (pk.float32, np.float32),
    (pk.float64, np.float64),
    (pk.double, np.float64),
    (pk.int64, np.int64),
    (pk.int32, np.int32),
    ])
def test_asarray_consts_vs_numpy(const, np_dtype, pk_dtype):
    actual = pk.asarray(const)
    numpy_val = np.asarray(const)
    assert_allclose(actual, numpy_val)
    # we compare dtype "strings" because our type system
    # needs a ton of work still...
    pk_type_string = str(actual.dtype).split(".")[-1][:-2]
    numpy_type_string = str(numpy_val.dtype)
    assert pk_type_string == numpy_type_string
    # none of the final types for these float
    # constants should ever be allowed to be ints
    assert not "int" in pk_type_string



@pytest.mark.parametrize("pk_dtype, np_dtype", [
    (pk.uint8, np.uint8),
    (pk.uint16, np.uint16),
    (pk.uint32, np.uint32),
    (pk.uint64, np.uint64),
    ])
def test_unsigned_int_overflow(pk_dtype, np_dtype):
    # test for gh-86
    actual = pk.View([1], dtype=pk_dtype)
    if np.__version__.startswith("2."):
        with pytest.raises(OverflowError):
            actual[:] = -1
    else:
        actual[:] = -1
        expected = np.array(-1, dtype=np_dtype)
        assert_equal(actual, expected)


@pytest.mark.parametrize("pk_dtype, pk_dtype2, expected_promo", [
    (pk.uint8, pk.uint16, pk.uint16),
    (pk.uint64, pk.uint8, pk.uint64),
    (pk.float32, pk.float64, pk.float64),
    ])
def test_result_type_supported(pk_dtype, pk_dtype2, expected_promo):
    # some basic behavior should already be covered
    # by:
    # array_api_tests/test_data_type_functions.py::test_result_type
    actual = pk.result_type(pk_dtype, pk_dtype2)
    assert actual == expected_promo


@pytest.mark.parametrize("pk_dtype, pk_dtype2", [
    (pk.array(np.array([0])), pk.uint16),
    (pk.uint64, pk.int8),
    (pk.float32, pk.int64),
    ])
def test_result_type_unsupported(pk_dtype, pk_dtype2):
    # support for views (arrays) and mixed type
    # categories is not yet available
    with pytest.raises(NotImplementedError):
        pk.result_type(pk_dtype, pk_dtype2)


if __name__ == '__main__':
    unittest.main()
