import math
from typing import List
import unittest

import pykokkos as pk
from numpy.testing import assert_allclose


# Tests for translation of each node of type AST to C++
@pk.functor
class ASTTestReduceFunctor:
    def __init__(self, threads: int, i_1: int, i_2: int, b_1: bool, b_2: bool):
        self.threads: int = threads
        self.i_1: int = i_1
        self.i_2: int = i_2
        self.b_1: bool = b_1
        self.b_2: bool = b_2

        self.view1D: pk.View1D[pk.int32] = pk.View([threads], pk.int32)
        self.view2D: pk.View2D[pk.int32] = pk.View([threads, threads], pk.int32)
        self.view3D: pk.View3D[pk.int32] = pk.View([threads, threads, threads], pk.int32)

    @pk.workunit
    def assign(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        x: int = self.i_1
        acc += x + self.i_2

    @pk.workunit
    def aug_assign(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        x: int = self.i_1
        x += self.i_2
        acc += x

    @pk.workunit
    def math_constants(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += math.pi + math.e + math.tau

    @pk.workunit
    def constants(self, tid: int) -> None:
        int_constant: int = 5
        bool_constant: bool = True

    @pk.workunit
    def subscript(self, tid: int) -> None:
        self.view1D[tid] = self.i_1

        for i in range(self.threads):
            self.view2D[tid][i] = self.i_1

        for i in range(self.threads):
            for j in range(self.threads):
                self.view3D[tid][i][j] = self.i_1

    @pk.workunit
    def bin_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.i_1 + self.i_2

    @pk.workunit
    def unary_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += (+ self.i_1)

    @pk.workunit
    def compare(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.i_1 > self.i_2:
            acc += self.i_1
        else:
            acc += self.i_2

    @pk.workunit
    def bool_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if not self.b_1:
            acc += self.i_1
        else:
            acc += self.i_2

    @pk.workunit
    def for_stmt(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        for i in range(self.i_1):
            acc += self.i_2

    @pk.workunit
    def for_step_stmt(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        for i in range(self.i_2, self.i_1, self.i_2):
            acc += self.i_2

    @pk.workunit
    def if_stmt(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.b_1:
            acc += self.i_1
        acc += self.i_2

    @pk.workunit
    def elif_stmt(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.b_1:
            acc += self.i_1
        elif self.b_2:
            acc += self.i_2

    @pk.workunit
    def if_else_stmt(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.b_1:
            acc += self.i_1
        else:
            acc += self.i_2

    @pk.workunit
    def while_stmt(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        x: int = 0
        while x < self.i_1:
            acc += self.i_2
            x += 1

    @pk.workunit
    def pass_stmt(self, tid: int) -> None:
        pass

    @pk.workunit
    def docstring(self, tid: int) -> None:
        """
        Test docstring
        """

    @pk.workunit
    def call(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        pk.printf("Testing printf: %d\n", self.i_1)
        acc += abs(- self.i_1)

    @pk.workunit
    def break_stmt(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        for i in range(self.i_1):
            acc += self.i_2
            break

    @pk.workunit
    def continue_stmt(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        for i in range(self.i_1):
            acc += self.i_2
            continue

    @pk.workunit
    def ann_assign(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        x: int = self.i_1
        acc += x

    @pk.workunit
    def list_type(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        x: List[int] = [self.i_1, self.i_2]
        acc += x[0]
        acc += x[1]


class TestASTTranslator(unittest.TestCase):
    def setUp(self):
        self.threads: int = 50
        self.i_1: int = 7
        self.i_2: int = 2
        self.b_1: bool = False
        self.b_2: bool = True

        self.functor = ASTTestReduceFunctor(self.threads,
                                           self.i_1, self.i_2,
                                           self.b_1, self.b_2)
        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)

    def test_assign(self):
        expected_result: int = self.threads * (self.i_1 + self.i_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.assign)

        self.assertEqual(expected_result, result)

    def test_aug_assign(self):
        expected_result: int = self.threads * (self.i_1 + self.i_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.aug_assign)

        self.assertEqual(expected_result, result)

    def test_math_constants(self):
        expected_result: float = self.threads * (math.pi + math.e + math.tau)
        result: float = pk.parallel_reduce(self.range_policy, self.functor.math_constants)

        assert_allclose(expected_result, result)

    # def test_constants(self):
    #     self.assertEqual(5, self.workload.int_constant)
    #     self.assertEqual(True, self.workload.bool_constant)

    def test_subscript(self):
        expected_result = self.i_1
        pk.parallel_for(self.range_policy, self.functor.subscript)

        for i in range(self.threads):
            self.assertEqual(expected_result, self.functor.view1D[i])
            for j in range(self.threads):
                self.assertEqual(expected_result, self.functor.view2D[i][j])
                for k in range(self.threads):
                    self.assertEqual(expected_result, self.functor.view3D[i][j][k])

    def test_bin_op(self):
        expected_result: int = self.threads * (self.i_1 + self.i_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.bin_op)

        self.assertEqual(expected_result, result)

    def test_unary_op(self):
        expected_result: int = self.threads * (self.i_1)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.unary_op)

        self.assertEqual(expected_result, result)

    def test_compare(self):
        if self.i_1 > self.i_2:
            expected_result: int = self.threads * (self.i_1)
        else:
            expected_result: int = self.threads * (self.i_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.compare)

        self.assertEqual(expected_result, result)

    def test_bool_op(self):
        if not self.b_1:
            expected_result: int = self.threads * (self.i_1)
        else:
            expected_result: int = self.threads * (self.i_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.bool_op)

        self.assertEqual(expected_result, result)

    def test_for_stmt(self):
        expected_result: int = 0
        for i in range(self.i_1):
            expected_result += self.threads * self.i_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.for_stmt)

        self.assertEqual(expected_result, result)

    def test_for_step_stmt(self):
        expected_result: int = 0
        for i in range(self.i_2, self.i_1, self.i_2):
            expected_result += self.threads * self.i_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.for_step_stmt)

        self.assertEqual(expected_result, result)

    def test_if_stmt(self):
        expected_result: int = 0
        if self.b_1:
            expected_result += self.threads * self.i_1
        expected_result += self.threads * self.i_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.if_stmt)

        self.assertEqual(expected_result, result)

    def test_elif_stmt(self):
        expected_result: int = 0
        if self.b_1:
            expected_result += self.threads * self.i_1
        elif self.b_2:
            expected_result += self.threads * self.i_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.elif_stmt)

        self.assertEqual(expected_result, result)

    def test_if_else_stmt(self):
        if self.b_1:
            expected_result: int = self.threads * self.i_1
        else:
            expected_result: int = self.threads * self.i_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.if_else_stmt)

        self.assertEqual(expected_result, result)

    def test_while_stmt(self):
        x: int = 0
        expected_result: int = 0
        while x < self.i_1:
            expected_result += self.threads * self.i_2
            x += 1
        result: int = pk.parallel_reduce(self.range_policy, self.functor.while_stmt)

        self.assertEqual(expected_result, result)

    def test_pass(self):
        pk.parallel_for(self.range_policy, self.functor.pass_stmt)

    def test_call(self):
        expected_result: int = self.threads * abs(- self.i_1)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.call)

        self.assertEqual(expected_result, result)

    def test_break(self):
        expected_result: int = 0
        for i in range(self.i_1):
            expected_result += self.threads * self.i_2
            break
        result: int = pk.parallel_reduce(self.range_policy, self.functor.break_stmt)

        self.assertEqual(expected_result, result)

    def test_continue(self):
        expected_result: int = 0
        for i in range(self.i_1):
            expected_result += self.threads * self.i_2
            continue
        result: int = pk.parallel_reduce(self.range_policy, self.functor.continue_stmt)

        self.assertEqual(expected_result, result)

    def test_ann_assign(self):
        expected_result: int = self.threads * self.i_1
        result: int = pk.parallel_reduce(self.range_policy, self.functor.ann_assign)

        self.assertEqual(expected_result, result)

    def test_list_type(self):
        expected_result: int = self.threads * (self.i_1 + self.i_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.list_type)

        self.assertEqual(expected_result, result)


@pk.workunit
def scratch_with_double_float(team_member: pk.TeamMember):
    scratch_mem_d: pk.ScratchView1D[double] = pk.ScratchView1D(team_member.team_scratch(0))
    scratch_mem_f: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0))


def test_gh_180():
    pk.parallel_for("double_float_scratch",
                    pk.TeamPolicy(2, 2), scratch_with_double_float)


if __name__ == "__main__":
    unittest.main()
