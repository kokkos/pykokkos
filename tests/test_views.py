from pykokkos.interface.data_types import int32
import pytest
import unittest

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

import numpy as np
from numpy.testing import assert_allclose
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

        self.np_view: pk.View2D[int] = pk.from_numpy(np_arr)
        self.cp_view: pk.View2D[int] = pk.from_cupy(cp_arr)

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

        np_view = pk.from_numpy(np_arr)
        cp_view = pk.from_cupy(cp_arr)

        pk.parallel_for(pk.RangePolicy(pk.OpenMP, 0, self.threads), addition_np, np_arr=np_view)
        pk.parallel_for(pk.RangePolicy(pk.Cuda, 0, self.threads), addition_cp, cp_arr=cp_view)

        for i in range(self.threads):
            self.assertEqual(1, np_arr[i][0])
            self.assertEqual(2, np_arr[i][1])
            self.assertEqual(1, np_view[i][0])
            self.assertEqual(2, np_view[i][1])

            self.assertEqual(1, cp_arr[i][0])
            self.assertEqual(2, cp_arr[i][1])


    @pytest.mark.skipif(not HAS_CUDA,
                        reason="CUDA/cupy not available")
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

@pk.workload
class SqrtView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = sqrt(self.view[tid])


@pk.workload
class ExpView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = exp(self.view[tid])

@pk.workload
class Exp2View1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = exp2(self.view[tid])

@pk.workload
class PositiveView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = positive(self.view[tid])

@pk.workload
class NegativeView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = negative(self.view[tid])

@pk.workload
class AbsoluteView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = abs(self.view[tid])

@pk.workload
class FabsoluteView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = fabs(self.view[tid])

@pk.workload
class RintView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = rint(self.view[tid])

@pk.workload
class ConjugateView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = conj(self.view[tid])

@pk.workload
class SignView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = sign(self.view[tid])

@pk.workload
class LogView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = log(self.view[tid])

@pk.workload
class Log2View1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = log2(self.view[tid])

@pk.workload
class Log10View1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = log10(self.view[tid])

@pk.workload
class Expm1View1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = expm1(self.view[tid])

@pk.workload
class Log1pView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = log1p(self.view[tid])

@pk.workload
class SquareView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = square(self.view[tid])

@pk.workload
class CbrtView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = cbrt(self.view[tid])

@pk.workload
class ReciprocalView1D:
    def __init__(self, threads: int, view: pk.View1D[pk.double]):
        self.threads: int = threads
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = reciprocal(self.view[tid])

@pytest.mark.parametrize("kokkos_test_class, numpy_ufunc", [
    (SqrtView1D, np.sqrt),
    (ExpView1D, np.exp),
    pytest.param(Exp2View1D, np.exp2, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(PositiveView1D, np.positive, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(NegativeView1D, np.negative, marks=pytest.mark.xfail(reason="see gh-27")),
    (AbsoluteView1D, np.absolute),
    (FabsoluteView1D, np.fabs),
    pytest.param(RintView1D, np.rint, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(ConjugateView1D, np.conjugate, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(SignView1D, np.sign, marks=pytest.mark.xfail(reason="see gh-27")),
    (LogView1D, np.log),
    (Log2View1D, np.log2),
    (Log10View1D, np.log10),
    (Expm1View1D, np.expm1),
    (Log1pView1D, np.log1p),
    pytest.param(SquareView1D, np.square, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(CbrtView1D, np.cbrt, marks=pytest.mark.xfail(reason="see gh-27")),
    pytest.param(ReciprocalView1D, np.reciprocal, marks=pytest.mark.xfail(reason="see gh-27")),
])
def test_1d_unary_ufunc_vs_numpy(kokkos_test_class, numpy_ufunc):
    # verify that we can easily recreate the functionality
    # of most NumPy "unary" ufuncs on 1D views/arrays without much
    # custom code
    # NOTE: maybe we directly provide i.e., pk.sqrt(view)
    # "pykokkos ufuncs" some day?
    view: pk.View1d = pk.View([10], pk.double)
    view[:] = np.arange(10, dtype=np.float64)
    sqrt_kokkos_instance = kokkos_test_class(threads=10, view=view)
    pk.execute(pk.ExecutionSpace.Default, sqrt_kokkos_instance)
    actual = sqrt_kokkos_instance.view
    expected = numpy_ufunc(range(10))
    assert_allclose(actual, expected)


if __name__ == '__main__':
    unittest.main()
