import unittest

import pykokkos as pk
from numpy.testing import assert_allclose


# Tests translation of KOKKOS_FUNCTIONS to C++
@pk.functor(
        subview_1=pk.ViewTypeInfo(trait=pk.Unmanaged),
        subview_2=pk.ViewTypeInfo(trait=pk.Unmanaged))
class KokkosFunctionsTestReduceFunctor:
    def __init__(self, threads: int, i_1: int, i_2: int, f_1: float, f_2: float, b_1: bool):
        self.threads: int = threads
        self.i_1: int = i_1
        self.i_2: int = i_2
        self.f_1: float = f_1
        self.f_2: float = f_2
        self.b_1: bool = b_1

        self.view1D: pk.View1D[pk.int32] = pk.View([threads], pk.int32)
        self.view2D: pk.View2D[pk.int32] = pk.View([threads, threads], pk.int32)
        self.view3D: pk.View3D[pk.int32] = pk.View([threads, threads, threads], pk.int32)

        self.subview_1: pk.View1D[pk.int32] = self.view1D[threads // 2:]
        self.subview_2: pk.View2D[pk.int32] = self.view2D[threads // 2:, :]

    @pk.function
    def return_constant(self) -> int:
        return 5

    @pk.workunit
    def add_constant(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.return_constant()

    @pk.function
    def return_args_sum(self, i_1_arg: int, i_2_arg: int) -> int:
        return i_1_arg + i_2_arg

    @pk.workunit
    def add_args(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.return_args_sum(self.i_1, self.i_2)

    @pk.function
    def return_fields_sum(self) -> int:
        return self.i_1 + self.i_2

    @pk.workunit
    def add_fields(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.return_fields_sum()

    @pk.function
    def return_bool(self) -> bool:
        return self.b_1

    @pk.workunit
    def add_bool(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.return_bool():
            acc += self.i_1
        else:
            acc += self.i_2

    @pk.function
    def return_float_sum(self) -> float:
        return self.f_1 + self.f_2

    @pk.workunit
    def add_floats(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.return_float_sum()

    @pk.function
    def nested_1(self, i_1_arg: int) -> int:
        return i_1_arg

    @pk.function
    def nested_2(self, i_1_arg: int, i_2_arg: int) -> int:
        return self.nested_1(i_1_arg) + i_2_arg

    @pk.workunit
    def add_nested(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.nested_2(self.i_1, self.i_2)

    @pk.function
    def use_views(self, tid: int) -> int:
        return self.view1D[tid] + self.view2D[tid][0] + self.view3D[tid][0][0]

    @pk.workunit
    def views(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        self.view1D[tid] = self.i_1
        self.view2D[tid][0] = self.i_1
        self.view3D[tid][0][0] = self.i_1

        acc += self.use_views(tid)

    @pk.function
    def nested_views_1(self, tid: int) -> int:
        return self.view1D[tid]

    @pk.function
    def nested_views_2(self, tid: int) -> int:
        return self.nested_views_1(tid) + self.view2D[tid][0]

    @pk.function
    def nested_views_3(self, tid: int) -> int:
        return self.nested_views_2(tid) + self.view3D[tid][0][0]

    @pk.workunit
    def nested_views(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        self.view1D[tid] = self.i_1
        self.view2D[tid][0] = self.i_1
        self.view3D[tid][0][0] = self.i_1

        acc += self.nested_views_3(tid)

    @pk.function
    def use_subviews(self) -> int:
        return self.subview_1[0] + self.subview_2[0][0]

    @pk.workunit
    def subviews(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.use_subviews()


class TestKokkosFunctionsTranslator(unittest.TestCase):
    def setUp(self):
        self.threads: int = 50
        self.i_1: int = 7
        self.i_2: int = 2
        self.f_1: float = 5.5
        self.f_2: float = 1.3
        self.b_1: bool = True

        self.functor = KokkosFunctionsTestReduceFunctor(self.threads,
                                                       self.i_1, self.i_2,
                                                       self.f_1, self.f_2,
                                                       self.b_1)
        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)

    def test_constant_sum(self):
        expected_result: int = self.threads * 5
        result: int = pk.parallel_reduce(self.range_policy, self.functor.add_constant)

        self.assertEqual(expected_result, result)

    def test_args_sum(self):
        expected_result: int = self.threads * (self.i_1 + self.i_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.add_args)

        self.assertEqual(expected_result, result)

    def test_fields_sum(self):
        expected_result: int = self.threads * (self.i_1 + self.i_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.add_fields)

        self.assertEqual(expected_result, result)

    def test_bool_sum(self):
        if self.b_1:
            expected_result: int = self.threads * self.i_1
        else:
            expected_result: int = self.threads * self.i_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.add_bool)

        self.assertEqual(expected_result, result)

    def test_float_sum(self):
        expected_result: float = self.threads * (self.f_1 + self.f_2)
        result: float = pk.parallel_reduce(self.range_policy, self.functor.add_floats)

        assert_allclose(result, expected_result)

    def test_nested_sum(self):
        expected_result: int = self.threads * (self.i_1 + self.i_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.add_nested)

        self.assertEqual(expected_result, result)

    def test_views_sum(self):
        expected_result: int = self.threads * (self.i_1 * 3)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.views)

        self.assertEqual(expected_result, result)

    def test_nested_views_sum(self):
        expected_result: int = self.threads * (self.i_1 * 3)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.nested_views)

        self.assertEqual(expected_result, result)

    def test_subviews_sum(self):
        expected_result: int = self.threads * (self.i_1 * 2)
        temp: int = pk.parallel_reduce(self.range_policy, self.functor.views) # initialize views
        result: int = pk.parallel_reduce(self.range_policy, self.functor.subviews)

        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
