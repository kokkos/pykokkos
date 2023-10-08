import unittest
import numpy as np
import pykokkos as pk

# workunits
@pk.workunit
def init_view(i, view: pk.View1D[pk.int64], init: pk.int64):
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


class TestTypeInference(unittest.TestCase):
    def setUp(self):
        self.threads: int = 50
        self.np_i8: np.int8 = np.int8(2**7 -1)
        self.np_i16: np.int16 = np.int16(2**15 -1)
        self.np_i32: np.int32 = np.int32(2**31 -1)
        self.np_i64: np.int64 = np.int64(2**63 -1)

        self.np_u8: np.uint8 = np.uint8(2**8 -1)
        self.np_u16: np.uint16 = np.uint16(2**16 -1)
        self.np_u32: np.uint32 = np.uint32(2**32 -1)
        self.np_u64: np.uint64 = np.uint64(2**64 -1)

        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)
        self.view1D: pk.View1D[pk.int32] = pk.View([self.threads], pk.int32)

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
        self.assertEqual(type(int64_view[0]), np.longlong) # why?

    def test_view_np_uint8(self):
        uint8_view = pk.View([self.threads], pk.uint8)
        pk.parallel_for(self.range_policy, init_view, view=uint8_view, init=self.np_u8)
        self.assertEqual(uint8_view[0], self.np_u8)
        self.assertEqual(type(uint8_view[0]), type(self.np_u8))

    def test_view_np_uint16(self):
        uint16_view = pk.View([self.threads], pk.uint16)
        pk.parallel_for(self.range_policy, init_view, view=uint16_view, init=self.np_u16)
        self.assertEqual(uint16_view[0], self.np_u16)
        self.assertEqual(type(uint16_view[0]), type(self.np_u16))

    def test_view_np_uint32(self):
        uint32_view = pk.View([self.threads], pk.uint32)
        pk.parallel_for(self.range_policy, init_view, view=uint32_view, init=self.np_u32)
        self.assertEqual(uint32_view[0], self.np_u32)
        self.assertEqual(type(uint32_view[0]), type(self.np_u32))

    def test_view_np_uint64(self):
        uint64_view = pk.View([self.threads], pk.uint64)
        pk.parallel_for(self.range_policy, init_view, view=uint64_view, init=self.np_u64)
        self.assertEqual(uint64_view[0], self.np_u64)
        self.assertEqual(type(uint64_view[0]), np.ulonglong) # Why does this happen?

    def tesy_layout_switchL(self):
        int64_view = pk.View([self.threads], pk.int64, layout=pk.Layout.LayoutLeft)
        pk.parallel_for(self.range_policy, init_view, view=int64_view, init=self.np_i64)
        self.assertEqual(int64_view.layout, pk.Layout.LayoutLeft)
        self.assertEqual(int64_view[0], self.np_i64)

    def test_layout_switchR(self):
        int64_view = pk.View([self.threads], pk.int64, layout=pk.Layout.LayoutRight)
        pk.parallel_for(self.range_policy, init_view, view=int64_view, init=self.np_i64)
        self.assertEqual(int64_view.layout, pk.Layout.LayoutRight)
        self.assertEqual(int64_view[0], self.np_i64)

    def test_cache_read(self):
        self.test_simple_parallelfor()
        self.test_simple_parallelreduce()
        self.test_simple_parallelscan()

    def test_all_numpyints(self):
        int64_view = pk.View([self.threads], pk.int64)
        # 32 bit will overflow
        pk.parallel_for(self.range_policy, add_all_init, view=int64_view, i8=self.np_i8, i16=self.np_i16, i32=self.np_i32, i64=np.int64(self.np_i32))
        expected_result = np.int64(self.np_i32) + self.np_i32 + self.np_i16 + self.np_i8
        self.assertEqual(int64_view[0], expected_result)

    def test_all_numpyuints(self):
        uint64_view = pk.View([self.threads], pk.uint64)
        # 32 bit will overflow
        pk.parallel_for(self.range_policy, add_all_init, view=uint64_view, i8=self.np_u8, i16=self.np_u16, i32=self.np_u32, i64=np.uint64(self.np_u32))
        expected_result = np.uint64(self.np_u32) + self.np_u32 + self.np_u16 + self.np_u8
        self.assertEqual(uint64_view[0], expected_result)

    def test_numpy_doubles(self):
        # double and float64 should be interchangable
        f64_view = pk.View([self.threads], pk.float64)
        pk.parallel_for(self.range_policy, add_two_init, view=f64_view, v1=np.double(3.4e+38 + 1), v2=np.float64(3.4e+38 + 1))
        expected_result = np.double(3.4e+38 + 1 + 3.4e+38 + 1) 
        self.assertEqual(f64_view[0], expected_result)
        self.assertEqual(type(f64_view[0]), type(expected_result))
        # does view support pk.double? swap primitive types as well
        f64_view = pk.View([self.threads], pk.double)
        pk.parallel_for(self.range_policy, add_two_init, view=f64_view, v1=np.float64(3.4e+38 + 1), v2=np.double(3.4e+38 + 1))
        self.assertEqual(f64_view[0], expected_result)
        self.assertEqual(type(f64_view[0]), type(expected_result))

    def test_numpy_floats(self):
        # pk.float and np.float32 should be interchangeable
        f32_view = pk.View([self.threads], pk.float) 
        pk.parallel_for(self.range_policy, add_two_init, view=f32_view, v1=np.float32(0.32), v2=np.float32(0.32))
        expected_result = np.float32(0.32 + 0.32)
        self.assertEqual(f32_view[0], expected_result)
        self.assertEqual(type(f32_view[0]), type(expected_result))
        # does view support pk.float32?
        f32_view = pk.View([self.threads], pk.float32) 
        pk.parallel_for(self.range_policy, add_two_init, view=f32_view, v1=np.float32(0.32), v2=np.float32(0.32))
        self.assertEqual(f32_view[0], expected_result)
        self.assertEqual(type(f32_view[0]), type(expected_result))

    def test_acc64(self):
        # can we return a double value?
        expect_result = np.float64(3.4e+38 * self.threads) # max_f32_val * x
        f64_view = pk.View([self.threads], pk.double)
        result = pk.parallel_reduce(self.range_policy, acc64, view=f64_view, init=np.float32(3.4e+38) )
        self.assertEqual(np.isclose(expect_result, result), True)
        self.assertEqual(np.float32(result), np.inf) # should overflow: ignore warning
        self.assertEqual(np.float64(result), result) # should be fine


if __name__ == "__main__":
    # unittest.main()
    test = TestTypeInference()
    test.setUp()
    # test.test_simple_parallelfor()
    # test.test_np_int8()
    # test.test_all_numpyints()
    # test.test_simple_parallelreduce()
    # test.test_all_numpyuints()
    # test.test_numpy_doubles()
    # test.test_view_np_uint32()
    # test.test_acc64()
    test.test_view_np_int64()
