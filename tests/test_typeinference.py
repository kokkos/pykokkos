import unittest
import numpy as np
import pykokkos as pk
import pytest
from typing import List

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


if __name__ == "__main__":
    unittest.main()
