import unittest

import pykokkos as pk


# Tests for translation of each operation to C++
@pk.functor
class OpsTestReduceFunctor:
    def __init__(self, threads: int, value_1: int, value_2: int, bool_1: bool, bool_2: bool):
        self.threads: int = threads
        self.value_1: int = value_1
        self.value_2: int = value_2
        self.bool_1: bool = bool_1
        self.bool_2: bool = bool_2

    @pk.workunit
    def add_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 + self.value_2

    @pk.workunit
    def sub_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 - self.value_2

    @pk.workunit
    def div_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 / self.value_2

    @pk.workunit
    def floordiv_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 // self.value_2

    @pk.workunit
    def mult_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 * self.value_2

    @pk.workunit
    def mod_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 % self.value_2

    @pk.workunit
    def pow_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 ** self.value_2

    @pk.workunit
    def lshift_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 << self.value_2

    @pk.workunit
    def rshift_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 >> self.value_2

    @pk.workunit
    def bitand_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 & self.value_2

    @pk.workunit
    def bitor_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 | self.value_2

    @pk.workunit
    def bitxor_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value_1 ^ self.value_2

    @pk.workunit
    def uadd_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += + self.value_1

    @pk.workunit
    def usub_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += - self.value_1

    @pk.workunit
    def not_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if not self.bool_1:
            acc += self.value_1
        else:
            acc += self.value_2

    @pk.workunit
    def invert_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += ~ self.value_1

    @pk.workunit
    def and_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.bool_1 and self.bool_2:
            acc += self.value_1
        else:
            acc += self.value_2

    @pk.workunit
    def or_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.bool_1 or self.bool_2:
            acc += self.value_1
        else:
            acc += self.value_2

    @pk.workunit
    def eq_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.bool_1 == self.bool_2:
            acc += self.value_1
        else:
            acc += self.value_2

    @pk.workunit
    def noteq_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.bool_1 != self.bool_2:
            acc += self.value_1
        else:
            acc += self.value_2

    @pk.workunit
    def lt_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.value_1 < self.value_2:
            acc += self.value_1
        else:
            acc += self.value_2

    @pk.workunit
    def lte_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.value_1 <= self.value_2:
            acc += self.value_1
        else:
            acc += self.value_2

    @pk.workunit
    def gt_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.value_1 > self.value_2:
            acc += self.value_1
        else:
            acc += self.value_2

    @pk.workunit
    def gte_op(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        if self.value_1 >= self.value_2:
            acc += self.value_1
        else:
            acc += self.value_2


class TestOpsTranslator(unittest.TestCase):
    def setUp(self):
        self.threads: int = 50
        self.value_1: int = 7
        self.value_2: int = 2
        self.bool_1: bool = True
        self.bool_2: bool = False

        self.functor = OpsTestReduceFunctor(self.threads,
                                           self.value_1, self.value_2,
                                           self.bool_1, self.bool_2)

        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)

    def test_add_op(self):
        expected_result: int = self.threads * (self.value_1 + self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.add_op)

        self.assertEqual(expected_result, result)

    def test_sub_op(self):
        expected_result: int = self.threads * (self.value_1 - self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.sub_op)

        self.assertEqual(expected_result, result)

    def test_div_op(self):
        expected_result: float = self.threads * (self.value_1 / self.value_2)
        result: float = pk.parallel_reduce(self.range_policy, self.functor.div_op)

        self.assertEqual(expected_result, result)

    def test_floordiv_op(self):
        expected_result: int = self.threads * (self.value_1 // self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.floordiv_op)

        self.assertEqual(expected_result, result)

    def test_mult_op(self):
        expected_result: int = self.threads * (self.value_1 * self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.mult_op)

        self.assertEqual(expected_result, result)

    def test_mod_op(self):
        expected_result: int = self.threads * (self.value_1 % self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.mod_op)

        self.assertEqual(expected_result, result)

    def test_pow_op(self):
        expected_result: int = self.threads * (self.value_1 ** self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.pow_op)

        self.assertEqual(expected_result, result)

    def test_lshift_op(self):
        expected_result: int = self.threads * (self.value_1 << self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.lshift_op)

        self.assertEqual(expected_result, result)

    def test_rshift_op(self):
        expected_result: int = self.threads * (self.value_1 >> self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.rshift_op)

        self.assertEqual(expected_result, result)

    def test_bitand_op(self):
        expected_result: int = self.threads * (self.value_1 & self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.bitand_op)

        self.assertEqual(expected_result, result)

    def test_bitor_op(self):
        expected_result: int = self.threads * (self.value_1 | self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.bitor_op)

        self.assertEqual(expected_result, result)

    def test_bitxor_op(self):
        expected_result: int = self.threads * (self.value_1 ^ self.value_2)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.bitxor_op)

        self.assertEqual(expected_result, result)

    def test_uadd_op(self):
        expected_result: int = self.threads * (+ self.value_1)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.uadd_op)

        self.assertEqual(expected_result, result)

    def test_usub_op(self):
        expected_result: int = self.threads * (- self.value_1)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.usub_op)

        self.assertEqual(expected_result, result)

    def test_not_op(self):
        if not self.bool_1:
            expected_result: int = self.threads * self.value_1
        else:
            expected_result: int = self.threads * self.value_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.not_op)

        self.assertEqual(expected_result, result)

    def test_invert_op(self):
        expected_result: int = self.threads * (~ self.value_1)
        result: int = pk.parallel_reduce(self.range_policy, self.functor.invert_op)

        self.assertEqual(expected_result, result)

    def test_and_op(self):
        if self.bool_1 and self.bool_2:
            expected_result: int = self.threads * self.value_1
        else:
            expected_result: int = self.threads * self.value_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.and_op)

        self.assertEqual(expected_result, result)

    def test_or_op(self):
        if self.bool_1 or self.bool_2:
            expected_result: int = self.threads * self.value_1
        else:
            expected_result: int = self.threads * self.value_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.or_op)

        self.assertEqual(expected_result, result)

    def test_eq_op(self):
        if self.bool_1 == self.bool_2:
            expected_result: int = self.threads * self.value_1
        else:
            expected_result: int = self.threads * self.value_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.eq_op)

        self.assertEqual(expected_result, result)

    def test_noteq_op(self):
        if self.bool_1 != self.bool_2:
            expected_result: int = self.threads * self.value_1
        else:
            expected_result: int = self.threads * self.value_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.noteq_op)

        self.assertEqual(expected_result, result)

    def test_lt_op(self):
        if self.value_1 < self.value_2:
            expected_result: int = self.threads * self.value_1
        else:
            expected_result: int = self.threads * self.value_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.lt_op)

        self.assertEqual(expected_result, result)

    def test_lte_op(self):
        if self.value_1 <= self.value_2:
            expected_result: int = self.threads * self.value_1
        else:
            expected_result: int = self.threads * self.value_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.lte_op)

        self.assertEqual(expected_result, result)

    def test_gt_op(self):
        if self.value_1 > self.value_2:
            expected_result: int = self.threads * self.value_1
        else:
            expected_result: int = self.threads * self.value_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.gt_op)

        self.assertEqual(expected_result, result)

    def test_gte_op(self):
        if self.value_1 >= self.value_2:
            expected_result: int = self.threads * self.value_1
        else:
            expected_result: int = self.threads * self.value_2
        result: int = pk.parallel_reduce(self.range_policy, self.functor.gte_op)

        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
