import unittest
import numpy as np
import pykokkos as pk
import pytest
from typing import List
from itertools import product

from pykokkos.interface.parallel_dispatch import convert_arrays
from pykokkos.interface.views import from_array
from pykokkos.kokkos_manager import get_available_execution_spaces

try:
    import cupy as cp

    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


# workunits
@pk.workunit
def init_view(i, view, init):
    view[i] = init


@pk.workunit
def init_view_annotated(i: int, view: pk.View1D[pk.int32], init: pk.int32):
    view[i] = init


@pk.workunit
def init_view_float32(i: int, view: pk.View1D[pk.float32], init: pk.float32):
    view[i] = init


@pk.workunit
def init_view_mixed(i: int, view, init: pk.int32):
    view[i] = init


@pk.workunit(view=pk.ViewTypeInfo(layout=pk.Layout.LayoutLeft))
def init_view_layout(i: int, view, init: pk.int32):
    view[i] = init


@pk.workunit
def init_all_views(i, view1D, view2D, view3D, max_dim, init):
    # 1D
    view1D[i] = init
    # 2D
    for j in range(max_dim):
        view2D[i][j] = init
        # 3D
        for k in range(max_dim):
            view3D[i][j][k] = init


@pk.workunit(view2D=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
def init_all_views_mixed(
    i, view1D, view2D, view3D: pk.View3D[pk.int32], max_dim: int, init
):
    # 1D
    view1D[i] = init
    # 2D
    for j in range(max_dim):
        view2D[i][j] = init
        # 3D
        for k in range(max_dim):
            view3D[i][j][k] = init


@pk.workunit
def team_reduce(team_member, acc, M, y, x, A):
    j: int = team_member.league_rank()

    def inner_reduce(i: int, inner_acc: pk.Acc[float]):
        inner_acc += A[j][i] * x[i]

    temp2: float = pk.parallel_reduce(pk.TeamThreadRange(team_member, M), inner_reduce)

    if team_member.team_rank() == 0:
        acc += y[j] * temp2


@pk.workunit
def team_reduce_mixed(team_member: pk.TeamMember, acc, M, y, x, A):
    j: int = team_member.league_rank()

    def inner_reduce(i: int, inner_acc: pk.Acc[float]):
        inner_acc += A[j][i] * x[i]

    temp2: float = pk.parallel_reduce(pk.TeamThreadRange(team_member, M), inner_reduce)

    if team_member.team_rank() == 0:
        acc += y[j] * temp2


@pk.workunit
def reduce(i, acc, view):
    acc += view[i]


@pk.workunit
def scan(i, acc, last_pass, view):
    acc += view[i]
    if last_pass:
        view[i] = acc


@pk.workunit
def scan_mixed(i, acc: pk.Acc[pk.double], last_pass: bool, view):
    acc += view[i]
    if last_pass:
        view[i] = acc


@pk.workunit
def acc64(i, acc, view, init):
    # acc is pk.double by default
    view[i] = init
    acc += init


@pk.workunit
def add_all_init(i, view, i8, i16, i32, i64):
    view[i] = i64 + i32 + i16 + i8


@pk.workunit
def add_two_init(i, view, v1, v2):
    view[i] = v1 + v2


@pk.workunit
def no_view(i: int, acc: pk.Acc[pk.double], n):
    acc = acc + n


# List[T] annotations at workunits
@pk.workunit
def list_int_add(i, view, lst: List[int]):
    view[i] = view[i] + lst[i]


@pk.workunit
def list_float_add(i, view, lst: List[float]):
    view[i] = view[i] + lst[i]


@pk.workunit
def list_int_reduce(i, acc: pk.Acc[pk.int32], lst: List[int]):
    acc += lst[i]


@pk.workunit
def list_2d_sum(tid, result, lst: List[List[int]], rows: int, cols: int):
    # Using single thread (tid==0) to avoid race conditions on result[0]
    if tid == 0:
        for i in range(rows):
            for j in range(cols):
                result[0] += lst[i][j]


@pk.workunit
def list_3d_sum(tid, result, lst: List[List[List[int]]], d0: int, d1: int, d2: int):
    # Using single thread (tid==0) to avoid race conditions on result[0]
    if tid == 0:
        for i in range(d0):
            for j in range(d1):
                for k in range(d2):
                    result[0] += lst[i][j][k]


class TestTypeInference(unittest.TestCase):
    def setUp(self):
        self.threads: int = 50
        self.np_i8: np.int8 = np.int8(2**7 - 1)
        self.np_i16: np.int16 = np.int16(2**15 - 1)
        self.np_i32: np.int32 = np.int32(2**31 - 1)
        self.np_i64: np.int64 = np.int64(2**63 - 1)

        self.np_u8: np.uint8 = np.uint8(2**8 - 1)
        self.np_u16: np.uint16 = np.uint16(2**16 - 1)
        self.np_u32: np.uint32 = np.uint32(2**32 - 1)
        self.np_u64: np.uint64 = np.uint64(2**64 - 1)

        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)
        self.team_policy = pk.TeamPolicy(self.threads, pk.AUTO)
        self.view1D: pk.View1D[pk.int32] = pk.View([self.threads], pk.int32)
        self.view2D: pk.View2D[pk.int32] = pk.View(
            [self.threads, self.threads], pk.int32
        )
        self.view3D: pk.View3D[pk.int32] = pk.View(
            [self.threads, self.threads, self.threads], pk.int32
        )

        if HAS_CUDA:
            self.range_policy_cuda = pk.RangePolicy(
                pk.ExecutionSpace.Cuda, 0, self.threads
            )
            self.view1D_cuda: pk.View1D[pk.int32] = pk.View(
                [self.threads], pk.int32, pk.CudaSpace, pk.LayoutLeft
            )
            self.view2D_cuda: pk.View1D[pk.int32] = pk.View(
                [self.threads, self.threads], pk.int32, pk.CudaSpace, pk.LayoutLeft
            )
            self.view3D_cuda: pk.View3D[pk.int32] = pk.View(
                [self.threads, self.threads, self.threads],
                pk.int32,
                pk.CudaSpace,
                pk.LayoutLeft,
            )

    def test_simple_parallelfor(self):
        expected_result: float = 1.0
        n = 1.0
        pk.parallel_for(self.range_policy, init_view, view=self.view1D, init=n)
        for i in range(0, self.threads):
            self.assertEqual(expected_result, self.view1D[i])

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA/cupy not available")
    def test_simple_parallelfor_cuda(self):
        if not HAS_CUDA:
            return
        expected_result: float = 1.0
        n = 1.0
        pk.parallel_for(
            self.range_policy_cuda, init_view, view=self.view1D_cuda, init=n
        )
        for i in range(0, self.threads):
            self.assertEqual(expected_result, self.view1D_cuda[i])

    def test_simple_parallelreduce(self):
        expect_result: float = self.threads
        n = 1
        pk.parallel_for(self.range_policy, init_view, view=self.view1D, init=n)
        result = pk.parallel_reduce(self.range_policy, reduce, view=self.view1D)
        self.assertEqual(expect_result, result)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA/cupy not available")
    def test_simple_parallelreduce_cuda(self):
        if not HAS_CUDA:
            return
        expect_result: float = self.threads
        n = 1
        pk.parallel_for(
            self.range_policy_cuda, init_view, view=self.view1D_cuda, init=n
        )
        result = pk.parallel_reduce(
            self.range_policy_cuda, reduce, view=self.view1D_cuda
        )
        self.assertEqual(expect_result, result)

    def test_simple_parallelscan(self):
        expect_result: float = np.cumsum(np.ones(self.threads))
        n = 1
        pk.parallel_for(self.range_policy, init_view, view=self.view1D, init=n)
        result = pk.parallel_scan(self.range_policy, scan, view=self.view1D)
        self.assertEqual(expect_result[self.threads - 1], result)
        for i in range(0, self.threads):
            self.assertEqual(expect_result[i], self.view1D[i])

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA/cupy not available")
    def test_simple_parallelscan_cuda(self):
        if not HAS_CUDA:
            return
        expect_result: float = np.cumsum(np.ones(self.threads))
        n = 1
        pk.parallel_for(
            self.range_policy_cuda, init_view, view=self.view1D_cuda, init=n
        )
        result = pk.parallel_scan(self.range_policy_cuda, scan, view=self.view1D_cuda)
        self.assertEqual(expect_result[self.threads - 1], result)
        for i in range(0, self.threads):
            self.assertEqual(expect_result[i], self.view1D_cuda[i])

    def test_reduceandfor_labels(self):
        # reduce and scan share the same dispatch
        expect_result: float = self.threads
        n = 1
        pk.parallel_for(
            "test_for", self.range_policy, init_view, view=self.view1D, init=n
        )
        result = pk.parallel_reduce(
            "test_reduce", self.range_policy, reduce, view=self.view1D
        )
        self.assertEqual(expect_result, result)

    def test_view_np_int8(self):
        int8_view = pk.View([self.threads], pk.int8)
        pk.parallel_for(self.range_policy, init_view, view=int8_view, init=self.np_i8)
        self.assertEqual(int8_view[0], self.np_i8)
        self.assertEqual(type(int8_view[0]), type(self.np_i8))

    def test_view_np_int16(self):
        int16_view = pk.View([self.threads], pk.int16)
        pk.parallel_for(self.range_policy, init_view, view=int16_view, init=self.np_i16)
        self.assertEqual(int16_view[0], self.np_i16)
        self.assertEqual(type(int16_view[0]), type(self.np_i16))

    def test_view_np_int64(self):
        int64_view = pk.View([self.threads], pk.int64)
        pk.parallel_for(self.range_policy, init_view, view=int64_view, init=self.np_i64)
        self.assertEqual(int64_view[0], self.np_i64)
        self.assertEqual(type(int64_view[0]), np.longlong)  # why?

    def test_view_np_uint8(self):
        uint8_view = pk.View([self.threads], pk.uint8)
        pk.parallel_for(self.range_policy, init_view, view=uint8_view, init=self.np_u8)
        self.assertEqual(uint8_view[0], self.np_u8)
        self.assertEqual(type(uint8_view[0]), type(self.np_u8))

    def test_view_np_uint16(self):
        uint16_view = pk.View([self.threads], pk.uint16)
        pk.parallel_for(
            self.range_policy, init_view, view=uint16_view, init=self.np_u16
        )
        self.assertEqual(uint16_view[0], self.np_u16)
        self.assertEqual(type(uint16_view[0]), type(self.np_u16))

    def test_view_np_uint32(self):
        uint32_view = pk.View([self.threads], pk.uint32)
        pk.parallel_for(
            self.range_policy, init_view, view=uint32_view, init=self.np_u32
        )
        self.assertEqual(uint32_view[0], self.np_u32)
        self.assertEqual(type(uint32_view[0]), type(self.np_u32))

    def test_view_np_uint64(self):
        uint64_view = pk.View([self.threads], pk.uint64)
        pk.parallel_for(
            self.range_policy, init_view, view=uint64_view, init=self.np_u64
        )
        self.assertEqual(uint64_view[0], self.np_u64)
        self.assertEqual(type(uint64_view[0]), np.ulonglong)  # Why does this happen?

    def test_layout_switchL(self):
        int64_view = pk.View([self.threads], pk.int64, layout=pk.Layout.LayoutLeft)
        pk.parallel_for(self.range_policy, init_view, view=int64_view, init=self.np_i64)
        self.assertEqual(int64_view.layout, pk.Layout.LayoutLeft)
        self.assertEqual(int64_view[0], self.np_i64)

        int64_view = pk.View([self.threads], pk.int64, layout=pk.Layout.LayoutRight)
        pk.parallel_for(self.range_policy, init_view, view=int64_view, init=self.np_i64)
        self.assertEqual(int64_view.layout, pk.Layout.LayoutRight)
        self.assertEqual(int64_view[0], self.np_i64)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA/cupy not available")
    def test_cuda_switch(self):
        if not HAS_CUDA:
            return
        int64_view = pk.View(
            [self.threads],
            pk.int64,
            space=pk.MemorySpace.CudaSpace,
            layout=pk.Layout.LayoutLeft,
        )
        pk.parallel_for(
            self.range_policy_cuda, init_view, view=int64_view, init=self.np_i64
        )
        self.assertEqual(int64_view.layout, pk.Layout.LayoutLeft)
        self.assertEqual(int64_view[0], self.np_i64)

    def test_cache_read(self):
        self.test_simple_parallelfor()
        self.test_simple_parallelreduce()
        self.test_simple_parallelscan()

    def test_all_numpyints(self):
        int64_view = pk.View([self.threads], pk.int64)
        # 32 bit will overflow
        pk.parallel_for(
            self.range_policy,
            add_all_init,
            view=int64_view,
            i8=self.np_i8,
            i16=self.np_i16,
            i32=self.np_i32,
            i64=np.int64(self.np_i32),
        )
        expected_result = np.int64(self.np_i32) + self.np_i32 + self.np_i16 + self.np_i8
        self.assertEqual(int64_view[0], expected_result)

    def test_all_numpyuints(self):
        uint64_view = pk.View([self.threads], pk.uint64)
        # 32 bit will overflow
        pk.parallel_for(
            self.range_policy,
            add_all_init,
            view=uint64_view,
            i8=self.np_u8,
            i16=self.np_u16,
            i32=self.np_u32,
            i64=np.uint64(self.np_u32),
        )
        expected_result = (
            np.uint64(self.np_u32) + self.np_u32 + self.np_u16 + self.np_u8
        )
        self.assertEqual(uint64_view[0], expected_result)

    def test_numpy_doubles(self):
        # double and float64 should be interchangable
        f64_view = pk.View([self.threads], pk.float64)
        pk.parallel_for(
            self.range_policy,
            add_two_init,
            view=f64_view,
            v1=np.double(3.4e38 + 1),
            v2=np.float64(3.4e38 + 1),
        )
        expected_result = np.double(3.4e38 + 1 + 3.4e38 + 1)
        self.assertEqual(f64_view[0], expected_result)
        self.assertEqual(type(f64_view[0]), type(expected_result))
        # does view support pk.double? swap primitive types as well
        f64_view = pk.View([self.threads], pk.double)
        pk.parallel_for(
            self.range_policy,
            add_two_init,
            view=f64_view,
            v1=np.float64(3.4e38 + 1),
            v2=np.double(3.4e38 + 1),
        )
        self.assertEqual(f64_view[0], expected_result)
        self.assertEqual(type(f64_view[0]), type(expected_result))

    def test_numpy_floats(self):
        # pk.float and np.float32 should be interchangeable
        f32_view = pk.View([self.threads], pk.float)
        pk.parallel_for(
            self.range_policy,
            add_two_init,
            view=f32_view,
            v1=np.float32(0.32),
            v2=np.float32(0.32),
        )
        expected_result = np.float32(0.32 + 0.32)
        self.assertEqual(f32_view[0], expected_result)
        self.assertEqual(type(f32_view[0]), type(expected_result))
        # does view support pk.float32?
        f32_view = pk.View([self.threads], pk.float32)
        pk.parallel_for(
            self.range_policy,
            add_two_init,
            view=f32_view,
            v1=np.float32(0.32),
            v2=np.float32(0.32),
        )
        self.assertEqual(f32_view[0], expected_result)
        self.assertEqual(type(f32_view[0]), type(expected_result))

    def test_acc64(self):
        # can we return a double value?
        expect_result = np.float64(3.4e38 * self.threads)  # max_f32_val * x
        f64_view = pk.View([self.threads], pk.double)
        result = pk.parallel_reduce(
            self.range_policy, acc64, view=f64_view, init=np.float32(3.4e38)
        )
        self.assertEqual(np.isclose(expect_result, result), True)
        self.assertEqual(np.float32(result), np.inf)  # should overflow: ignore warning
        self.assertEqual(np.float64(result), result)  # should be fine

    def test_all_Ds(self):
        expect_result = 1
        pk.parallel_for(
            self.range_policy,
            init_all_views,
            view1D=self.view1D,
            view2D=self.view2D,
            view3D=self.view3D,
            max_dim=self.threads,
            init=expect_result,
        )
        for i in range(self.threads):
            self.assertEqual(self.view1D[i], expect_result)
            for j in range(self.threads):
                self.assertEqual(self.view2D[i][j], expect_result)
                for k in range(self.threads):
                    self.assertEqual(self.view3D[i][j][k], expect_result)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA/cupy not available")
    def test_all_Ds_cuda(self):
        if not HAS_CUDA:
            return
        expect_result = 1
        pk.parallel_for(
            self.range_policy_cuda,
            init_all_views,
            view1D=self.view1D_cuda,
            view2D=self.view2D_cuda,
            view3D=self.view3D_cuda,
            max_dim=self.threads,
            init=expect_result,
        )
        for i in range(self.threads):
            self.assertEqual(self.view1D_cuda[i], expect_result)
            for j in range(self.threads):
                self.assertEqual(self.view2D_cuda[i][j], expect_result)
                for k in range(self.threads):
                    self.assertEqual(self.view3D_cuda[i][j][k], expect_result)

    def test_team_policy(self):
        # running team policy example
        y: pk.View1D = pk.View([self.threads], pk.double)
        x: pk.View1D = pk.View([self.threads], pk.double)
        A: pk.View2D = pk.View([self.threads, self.threads], pk.double)

        for i in range(self.threads):
            y[i] = 1
            x[i] = 1
            for j in range(self.threads):
                A[j][i] = 1

        p = self.team_policy
        result = pk.parallel_reduce(p, team_reduce, M=self.threads, y=y, x=x, A=A)
        expected_result = self.threads * self.threads
        self.assertEqual(result, expected_result)

    def test_already_annotated(self):
        pk.parallel_for(
            self.range_policy, init_view_annotated, view=self.view1D, init=1
        )
        self.assertEqual(type(self.view1D[0]), np.int32)

    def test_mixed_annotated(self):
        pk.parallel_for(self.range_policy, init_view_mixed, view=self.view1D, init=1)
        self.assertEqual(type(self.view1D[0]), np.int32)

    def test_layout_decorated(self):
        l_view = pk.View([self.threads], pk.int32, layout=pk.Layout.LayoutLeft)
        pk.parallel_for(self.range_policy, init_view_layout, view=l_view, init=1)
        self.assertEqual(l_view.layout, pk.Layout.LayoutLeft)

    def test_only_layoutL(self):
        l_view = pk.View([self.threads], pk.int32, layout=pk.Layout.LayoutLeft)
        pk.parallel_for(
            self.range_policy, init_view_annotated, view=l_view, init=self.np_i32
        )
        self.assertEqual(l_view.layout, pk.Layout.LayoutLeft)

    def test_only_layoutR(self):
        r_view = pk.View([self.threads], pk.int32, layout=pk.Layout.LayoutRight)
        pk.parallel_for(
            self.range_policy, init_view_annotated, view=r_view, init=self.np_i32
        )
        self.assertEqual(r_view.layout, pk.Layout.LayoutRight)

    def test_resetting_simple(self):
        expect_result = 1
        pk.parallel_for(
            self.range_policy,
            init_all_views_mixed,
            view1D=self.view1D,
            view2D=self.view2D,
            view3D=self.view3D,
            max_dim=self.threads,
            init=expect_result,
        )
        # run again to test AST resetting, change a single type/layout to trigger.
        l_view = pk.View([self.threads], pk.int32, layout=pk.Layout.LayoutLeft)
        pk.parallel_for(
            self.range_policy,
            init_all_views_mixed,
            view1D=l_view,
            view2D=self.view2D,
            view3D=self.view3D,
            max_dim=self.threads,
            init=expect_result,
        )
        for i in range(self.threads):
            self.assertEqual(self.view1D[i], expect_result)
            for j in range(self.threads):
                self.assertEqual(self.view2D[i][j], expect_result)
                for k in range(self.threads):
                    self.assertEqual(self.view3D[i][j][k], expect_result)

    def test_resetting_scan(self):
        expect_result: float = np.cumsum(np.ones(self.threads))
        n = 1
        pk.parallel_for(self.range_policy, init_view, view=self.view1D, init=n)
        result = pk.parallel_scan(self.range_policy, scan_mixed, view=self.view1D)
        # run again with a change in type, can the AST be reset with user provided annos
        new_view = pk.View([self.threads], pk.double)
        pk.parallel_for(self.range_policy, init_view, view=new_view, init=n)
        result = pk.parallel_scan(self.range_policy, scan_mixed, view=new_view)
        self.assertEqual(expect_result[self.threads - 1], result)
        for i in range(0, self.threads):
            self.assertEqual(expect_result[i], self.view1D[i])

    def test_resetting_team(self):
        # running team policy example
        y: pk.View1D = pk.View([self.threads], pk.double)
        x: pk.View1D = pk.View([self.threads], pk.double)
        A: pk.View2D = pk.View([self.threads, self.threads], pk.double)

        for i in range(self.threads):
            y[i] = 1
            x[i] = 1
            for j in range(self.threads):
                A[j][i] = 1

        p = self.team_policy
        result = pk.parallel_reduce(p, team_reduce_mixed, M=self.threads, y=y, x=x, A=A)
        expected_result = self.threads * self.threads
        # run again see if user provided annos can be reset
        new_view = pk.View([self.threads], pk.double, layout=pk.Layout.LayoutLeft)
        for i in range(self.threads):
            new_view[i] = 1
        result = pk.parallel_reduce(
            p, team_reduce_mixed, M=self.threads, y=new_view, x=x, A=A
        )
        self.assertEqual(result, expected_result)

    def test_no_view(self):
        pk.parallel_reduce(self.range_policy, no_view, n=1)
        pk.parallel_reduce(self.range_policy, no_view, n=2.1)

    def test_list_int_python_list(self):
        view = pk.View([self.threads], pk.int32)
        view.fill(1)
        python_list = [i for i in range(self.threads)]

        pk.parallel_for(self.range_policy, list_int_add, view=view, lst=python_list)

        for i in range(self.threads):
            self.assertEqual(view[i], 1 + i)

    def test_list_int_numpy_array(self):
        view = pk.View([self.threads], pk.int32)
        view.fill(2)
        numpy_array = np.arange(self.threads, dtype=np.int32)

        pk.parallel_for(self.range_policy, list_int_add, view=view, lst=numpy_array)

        for i in range(self.threads):
            self.assertEqual(view[i], 2 + i)

    def test_list_float_python_list(self):
        view = pk.View([self.threads], pk.double)
        view.fill(1.5)
        python_list = [float(i) * 0.5 for i in range(self.threads)]

        pk.parallel_for(self.range_policy, list_float_add, view=view, lst=python_list)

        for i in range(self.threads):
            expected = 1.5 + (i * 0.5)
            self.assertAlmostEqual(view[i], expected, places=5)

    def test_list_int_reduce(self):
        python_list = [1] * self.threads
        result = pk.parallel_reduce(self.range_policy, list_int_reduce, lst=python_list)

        expected_result = self.threads
        self.assertEqual(result, expected_result)

    def test_list_int_mixed_values(self):
        view = pk.View([self.threads], pk.int32)
        view.fill(10)
        python_list = [(-1) ** i * i for i in range(self.threads)]

        pk.parallel_for(self.range_policy, list_int_add, view=view, lst=python_list)

        for i in range(self.threads):
            expected = 10 + ((-1) ** i * i)
            self.assertEqual(view[i], expected)

    def test_list_2d_python_list(self):
        rows, cols = 3, 4
        python_2d = [[i * cols + j for j in range(cols)] for i in range(rows)]
        result = pk.View([1], pk.int32)
        result.fill(0)

        pk.parallel_for(
            1, list_2d_sum, result=result, lst=python_2d, rows=rows, cols=cols
        )

        expected = sum(sum(row) for row in python_2d)
        self.assertEqual(result[0], expected)

    def test_list_2d_numpy_array(self):
        rows, cols = 3, 4
        numpy_2d = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
        result = pk.View([1], pk.int32)
        result.fill(0)

        pk.parallel_for(
            1, list_2d_sum, result=result, lst=numpy_2d, rows=rows, cols=cols
        )

        expected = int(numpy_2d.sum())
        self.assertEqual(result[0], expected)

    def test_list_3d_python_list(self):
        d0, d1, d2 = 2, 3, 4
        python_3d = [
            [[i * 100 + j * 10 + k for k in range(d2)] for j in range(d1)]
            for i in range(d0)
        ]
        result = pk.View([1], pk.int32)
        result.fill(0)

        pk.parallel_for(
            1, list_3d_sum, result=result, lst=python_3d, d0=d0, d1=d1, d2=d2
        )

        expected = sum(sum(sum(row) for row in plane) for plane in python_3d)
        self.assertEqual(result[0], expected)

    def test_numpy_noncontiguous(self):
        contiguous = np.arange(10)
        noncontiguous = contiguous[::2]
        with pytest.raises(ValueError, match="not contiguous"):
            pk.parallel_for(len(noncontiguous), init_view, view=noncontiguous, init=0)

    def _convert_tensor(
        self, tensor, space=pk.ExecutionSpace.OpenMP, workunit=init_view
    ):
        kwargs = {"view": tensor}
        convert_arrays(kwargs, workunit, space)
        return kwargs["view"]

    def _cuda_tensor(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("cupy")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA not available")
        if "Cuda" not in get_available_execution_spaces():
            pytest.skip("Kokkos CUDA execution space not available")
        return torch.arange(48, dtype=torch.int32, device="cuda")

    def test_torch_cpu_conversion_preserves_metadata_and_shares_storage(self):
        torch = pytest.importorskip("torch")
        shapes = [(8,), (4, 6), (2, 4, 6)]
        dtypes = ["int32", "int64", "float32", "float64"]
        for shape, dtype_name, convert in product(
            shapes, dtypes, (self._convert_tensor, from_array)
        ):
            tensor = torch.arange(
                int(np.prod(shape)), dtype=getattr(torch, dtype_name)
            ).reshape(shape)

            view = convert(tensor)

            self.assertIsInstance(view, pk.View)
            self.assertEqual(view.shape, shape)
            self.assertIs(view.dtype, getattr(pk, dtype_name))
            self.assertIs(view.space, pk.HostSpace)
            np.testing.assert_array_equal(view.data, tensor.numpy())
            tensor.reshape(-1)[0] = 17
            self.assertEqual(view.data.flat[0], 17)
            view.data.flat[1] = 23
            self.assertEqual(tensor.reshape(-1)[1].item(), 23)

    def test_torch_cpu_matching_annotation_accepted(self):
        torch = pytest.importorskip("torch")
        for dtype, workunit, expected_dtype in (
            (torch.int32, init_view_annotated, pk.int32),
            (torch.float32, init_view_float32, pk.float32),
        ):
            tensor = torch.ones(8, dtype=dtype)
            view = self._convert_tensor(tensor, workunit=workunit)
            self.assertIs(view.dtype, expected_dtype)

    def test_torch_cpu_parallel_for(self):
        torch = pytest.importorskip("torch")
        tensor = torch.zeros(self.threads, dtype=torch.int32)
        policy = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, self.threads)

        pk.parallel_for(policy, init_view_annotated, view=tensor, init=17)

        self.assertTrue(torch.all(tensor == 17).item())

    def test_torch_cpu_parallel_reduce(self):
        torch = pytest.importorskip("torch")
        tensor = torch.arange(self.threads, dtype=torch.float64)
        policy = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, self.threads)

        result = pk.parallel_reduce(policy, reduce, view=tensor)

        self.assertEqual(result, tensor.sum().item())

    def test_torch_cpu_column_major_tensor(self):
        torch = pytest.importorskip("torch")
        tensor = torch.arange(24, dtype=torch.int32).reshape(4, 6).t()

        view = self._convert_tensor(tensor)

        self.assertIs(view.layout, pk.LayoutLeft)
        np.testing.assert_array_equal(view.data, tensor.numpy())
        view.data[0, 1] = 17
        self.assertEqual(tensor[0, 1].item(), 17)

    def test_torch_cpu_bool_shares_storage(self):
        torch = pytest.importorskip("torch")
        tensor = torch.tensor([False, True, False])

        view = self._convert_tensor(tensor)

        self.assertIs(view.dtype, pk.uint8)
        tensor[0] = True
        self.assertEqual(view.data[0], 1)
        view.data[1] = 0
        self.assertFalse(tensor[1].item())

    def test_torch_cpu_requires_grad_shares_storage(self):
        torch = pytest.importorskip("torch")
        tensor = torch.ones(8, requires_grad=True)

        view = self._convert_tensor(tensor)

        self.assertTrue(tensor.requires_grad)
        self.assertIsNone(tensor.grad)
        view.data[0] = 17
        self.assertEqual(tensor[0].item(), 17)

    def test_torch_cpu_view_retains_storage(self):
        torch = pytest.importorskip("torch")
        # The tensor (and its original storage owner) go out of scope here.
        view = self._convert_tensor(torch.arange(12, dtype=torch.int32)[4:])

        np.testing.assert_array_equal(view.data, np.arange(4, 12, dtype=np.int32))
        view.data[0] = 17
        self.assertEqual(view[0], 17)

    def test_torch_unsupported_device_rejected(self):
        torch = pytest.importorskip("torch")
        tensor = torch.empty(8, device="meta")
        with pytest.raises(TypeError, match="unsupported Torch tensor"):
            self._convert_tensor(tensor)

    def test_torch_sparse_tensor_rejected(self):
        torch = pytest.importorskip("torch")
        tensor = torch.sparse_coo_tensor([[0, 2]], [1.0, 2.0], (8,))
        with pytest.raises(TypeError, match="unsupported Torch tensor"):
            self._convert_tensor(tensor)

    def test_torch_cpu_mismatched_annotation_rejected(self):
        torch = pytest.importorskip("torch")
        for dtype, workunit, expected_dtype in (
            (torch.int64, init_view_annotated, "int32"),
            (torch.float64, init_view_float32, "float32"),
        ):
            tensor = torch.ones(8, dtype=dtype)
            with pytest.raises(
                TypeError, match=f"expects a View with dtype {expected_dtype}"
            ):
                self._convert_tensor(tensor, workunit=workunit)

    def test_torch_cpu_noncontiguous(self):
        torch = pytest.importorskip("torch")
        for shape, convert in product(
            [(8,), (4, 6), (2, 4, 6)], (self._convert_tensor, from_array)
        ):
            tensor = torch.arange(int(np.prod(shape)), dtype=torch.int32).reshape(
                shape
            )[..., ::2]
            self.assertFalse(tensor.is_contiguous())
            with pytest.raises(ValueError, match="not contiguous"):
                convert(tensor)

    def test_torch_cpu_rejected_for_device_execution(self):
        torch = pytest.importorskip("torch")
        tensor = torch.ones(8, dtype=torch.int32)
        for space in (pk.ExecutionSpace.Cuda, pk.ExecutionSpace.HIP):
            with pytest.raises(TypeError, match="cannot be accessed"):
                self._convert_tensor(tensor, space)
            with pytest.raises(TypeError, match="cannot be accessed"):
                from_array(tensor, execution_space=space)

    def test_torch_cuda_conversion_shares_storage(self):
        tensor = self._cuda_tensor()
        view = self._convert_tensor(tensor, pk.ExecutionSpace.Cuda)
        self.assertIsInstance(view, pk.View)
        self.assertEqual(view.shape, tuple(tensor.shape))
        self.assertIs(view.dtype, pk.int32)
        self.assertIs(view.space, pk.CudaSpace)
        self.assertEqual(view.xp_array.data.ptr, tensor.data_ptr())

    def test_torch_cuda_rejected_for_host_execution(self):
        tensor = self._cuda_tensor()
        with pytest.raises(TypeError, match="cannot be accessed"):
            self._convert_tensor(tensor)

    def test_torch_cuda_parallel_for(self):
        torch = pytest.importorskip("torch")
        tensor = self._cuda_tensor()
        policy = pk.RangePolicy(pk.ExecutionSpace.Cuda, 0, len(tensor))

        pk.parallel_for(policy, init_view_annotated, view=tensor, init=17)
        torch.cuda.synchronize()

        self.assertTrue(torch.all(tensor == 17).item())

    def test_torch_cuda_noncontiguous(self):
        contiguous = self._cuda_tensor()
        for shape in [(48,), (6, 8), (2, 4, 6)]:
            tensor = contiguous.reshape(shape)[..., ::2]
            self.assertFalse(tensor.is_contiguous())
            with pytest.raises(ValueError, match="not contiguous"):
                self._convert_tensor(tensor, pk.ExecutionSpace.Cuda)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("shape", [(10,), (4, 6), (4, 6, 8)], ids=["1d", "2d", "3d"])
@pytest.mark.parametrize("step", [2, -1], ids=["stepped", "reversed"])
@pytest.mark.parametrize("entrypoint", ["array", "parallel_for"])
def test_noncontiguous_array_rejected(backend, shape, step, entrypoint):
    if backend == "cupy":
        xp = pytest.importorskip("cupy")
        from pykokkos.kokkos_manager import get_available_execution_spaces

        if "Cuda" not in get_available_execution_spaces():
            pytest.skip("Kokkos CUDA execution space not available")
        try:
            device_count = xp.cuda.runtime.getDeviceCount()
        except xp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CUDA runtime not available: {exc}")
        if device_count == 0:
            pytest.skip("No CUDA devices available")
        space = pk.ExecutionSpace.Cuda
    else:
        xp = np
        space = pk.ExecutionSpace.OpenMP

    contiguous = xp.arange(int(np.prod(shape)), dtype=xp.int32).reshape(shape)
    noncontiguous = contiguous[..., ::step]
    assert not noncontiguous.flags["C_CONTIGUOUS"]
    assert not noncontiguous.flags["F_CONTIGUOUS"]

    with pytest.raises(ValueError, match="not contiguous"):
        if entrypoint == "array":
            pk.array(noncontiguous)
        else:
            policy = pk.RangePolicy(space, 0, len(noncontiguous))
            pk.parallel_for(policy, init_view, view=noncontiguous, init=0)


if __name__ == "__main__":
    unittest.main()
