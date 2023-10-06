import unittest
import numpy as np
import pykokkos as pk

# workunits
@pk.workunit
def init_view(i, view, init):
    view[i] = init

@pk.workunit
def reduce(i, acc, view):
    acc += view[i]

@pk.workunit
def scan(i, acc, last_pass, view):
    acc += view[i]
    if last_pass:
        view[i] = acc

@pk.workunit
def acc64(i, acc, view, i32):
    # acc is pk.double by default
    view[i] = i32
    acc += i32

@pk.workunit
def numpy_all_ints(i, view, i8, i16, i32, i64):
    view[i] = i64 + i32 + i16 + i8

@pk.workunit
def numpy_all_uints(i, view, u8, u16, u32, u64):
    view[i] = u64 + u32 + u16 + u8

@pk.workunit
def numpy_double(i, view, d1, f64):
    view[i] = d1 + f64

@pk.workunit
def numpy_float(i, view, f, f32):
    view[i] = f + f32

class TestTypeInference(unittest.TestCase):
    def setUp(self):
        self.threads: int = 50
        self.i_1: int = 7
        self.i_2: int = 2
        self.b_1: bool = False
        self.b_2: bool = True
        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)
        self.view1D: pk.View1D[pk.int32] = pk.View([self.threads], pk.int32)
        self.view2D: pk.View2D[pk.int32] = pk.View([self.threads, self.threads], pk.int32)
        self.view3D: pk.View3D[pk.int32] = pk.View([self.threads, self.threads, self.threads], pk.int32)

    def test_simple_parallelfor(self):
        expected_result: float = 1.0
        n=1.0
        pk.parallel_for(self.range_policy, init_view, view=self.view1D, init=n)
        for i in range(0, self.threads):
            self.assertEqual(expected_result, self.view1D[i])

    def test_simple_parallelreduce(self):
        expect_result: float = self.threads
        n = 1
        pk.parallel_for(self.range_policy, init_view, view=self.view1D, init=n)
        result = pk.parallel_reduce(self.range_policy, reduce, view=self.view1D)
        self.assertEqual(expect_result, result)

    def test_simple_parallelscan(self):
        expect_result: float = np.cumsum(np.ones(self.threads))
        n = 1
        pk.parallel_for(self.range_policy, init_view, view=self.view1D, init=n)
        result = pk.parallel_scan(self.range_policy, scan, view=self.view1D)
        self.assertEqual(expect_result[self.threads-1], result)
        for i in range(0, self.threads):
            self.assertEqual(expect_result[i], self.view1D[i])

    def test_acc(self):
        t1 = np.int32(2**31 -1)
        expect_result = t1 * self.threads
        result = pk.parallel_reduce(self.range_policy, acc64, view=self.view1D, i32=t1)
        self.assertEqual(expect_result, result)
        self.assertEqual(t1, self.view1D[0])

    def test_np_int8(self):
        n = np.int8(2**7 -1)
        int8_view = pk.View([self.threads], pk.int8)
        pk.parallel_for(self.range_policy, init_view, view=int8_view, init=n)
        self.assertEqual(int8_view[0], n)


if __name__ == "__main__":
    unittest.main()
    # test = TestTypeInference()
    # test.setUp()
    # test.test_np_int8()
