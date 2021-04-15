import unittest

import pykokkos as pk


@pk.functor
class AtomicsTestFunctor:
    def __init__(self, threads: int, i_1: int, i_2: int, f_1: float, f_2: float):
        self.threads: int = threads
        self.i_1: int = i_1
        self.i_2: int = i_2
        self.f_1: float = f_1
        self.f_2: float = f_2

        self.view1D_add: pk.View1D[pk.double] = pk.View([1], pk.double)
        self.view1D_and: pk.View1D[pk.int32] = pk.View([1], pk.int32)
        self.view1D_div: pk.View1D[pk.double] = pk.View([1], pk.double)
        self.view1D_lshift: pk.View1D[pk.int32] = pk.View([1], pk.int32)
        self.view1D_max: pk.View1D[pk.double] = pk.View([1], pk.double)
        self.view1D_min: pk.View1D[pk.double] = pk.View([1], pk.double)
        self.view1D_mod: pk.View1D[pk.int32] = pk.View([1], pk.int32)
        self.view1D_mul: pk.View1D[pk.double] = pk.View([1], pk.double)
        self.view1D_or: pk.View1D[pk.int32] = pk.View([1], pk.int32)
        self.view1D_rshift: pk.View1D[pk.int32] = pk.View([1], pk.int32)
        self.view1D_sub: pk.View1D[pk.double] = pk.View([1], pk.double)
        self.view1D_xor: pk.View1D[pk.int32] = pk.View([1], pk.int32)

        self.view1D_add[0] = f_1
        self.view1D_and[0] = i_1
        self.view1D_div[0] = f_1
        self.view1D_lshift[0] = i_1
        self.view1D_max[0] = f_1
        self.view1D_min[0] = f_1
        self.view1D_mod[0] = i_1
        self.view1D_mul[0] = f_1
        self.view1D_or[0] = i_1
        self.view1D_rshift[0] = i_1
        self.view1D_sub[0] = f_1
        self.view1D_xor[0] = i_1

    @pk.workunit
    def atomic_add(self, tid: int) -> None:
        pk.atomic_fetch_add(self.view1D_add, [0], self.f_2)

    @pk.workunit
    def atomic_and(self, tid: int) -> None:
        pk.atomic_fetch_and(self.view1D_and, [0], self.i_2)

    @pk.workunit
    def atomic_div(self, tid: int) -> None:
        pk.atomic_fetch_div(self.view1D_div, [0], self.f_2)

    @pk.workunit
    def atomic_lshift(self, tid: int) -> None:
        pk.atomic_fetch_lshift(self.view1D_lshift, [0], self.i_2)

    @pk.workunit
    def atomic_max(self, tid: int) -> None:
        pk.atomic_fetch_max(self.view1D_max, [0], self.f_2)

    @pk.workunit
    def atomic_min(self, tid: int) -> None:
        pk.atomic_fetch_min(self.view1D_min, [0], self.f_2)

    @pk.workunit
    def atomic_mod(self, tid: int) -> None:
        pk.atomic_fetch_mod(self.view1D_mod, [0], self.i_2)

    @pk.workunit
    def atomic_mul(self, tid: int) -> None:
        pk.atomic_fetch_mul(self.view1D_mul, [0], self.f_2)

    @pk.workunit
    def atomic_or(self, tid: int) -> None:
        pk.atomic_fetch_or(self.view1D_or, [0], self.i_2)

    @pk.workunit
    def atomic_rshift(self, tid: int) -> None:
        pk.atomic_fetch_rshift(self.view1D_rshift, [0], self.i_2)

    @pk.workunit
    def atomic_sub(self, tid: int) -> None:
        pk.atomic_fetch_sub(self.view1D_sub, [0], self.f_2)

    @pk.workunit
    def atomic_xor(self, tid: int) -> None:
        pk.atomic_fetch_xor(self.view1D_xor, [0], self.i_2)


class TestAtomic(unittest.TestCase):
    def setUp(self):
        self.threads: int = 1
        self.i_1: int = 5
        self.i_2: int = 2
        self.f_1: float = 7.0
        self.f_2: float = 3.0

        self.functor = AtomicsTestFunctor(self.threads, self.i_1, self.i_2, self.f_1, self.f_2)
        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)

    def test_atomic_add(self):
        expected_result: float = self.f_1 + self.f_2

        pk.parallel_for(self.range_policy, self.functor.atomic_add)
        result: float = self.functor.view1D_add[0]

        self.assertEqual(expected_result, result)

    def test_atomic_and(self):
        expected_result: int = self.i_1 & self.i_2

        pk.parallel_for(self.range_policy, self.functor.atomic_and)
        result: int = self.functor.view1D_and[0]

        self.assertEqual(expected_result, result)

    def test_atomic_div(self):
        expected_result: float = self.f_1 / self.f_2

        pk.parallel_for(self.range_policy, self.functor.atomic_div)
        result: float = self.functor.view1D_div[0]

        self.assertEqual(expected_result, result)

    def test_atomic_lshift(self):
        expected_result: int = self.i_1 << self.i_2

        pk.parallel_for(self.range_policy, self.functor.atomic_lshift)
        result: int = self.functor.view1D_lshift[0]

        self.assertEqual(expected_result, result)

    def test_atomic_max(self):
        expected_result: float = max(self.f_1, self.f_2)

        result: float = self.functor.view1D_max[0]
        pk.parallel_for(self.range_policy, self.functor.atomic_max)

        self.assertEqual(expected_result, result)

    def test_atomic_min(self):
        expected_result: float = min(self.f_1, self.f_2)

        pk.parallel_for(self.range_policy, self.functor.atomic_min)
        result: float = self.functor.view1D_min[0]

        self.assertEqual(expected_result, result)

    def test_atomic_mod(self):
        expected_result: int = self.i_1 % self.i_2

        pk.parallel_for(self.range_policy, self.functor.atomic_mod)
        result: int = self.functor.view1D_mod[0]

        self.assertEqual(expected_result, result)

    def test_atomic_mul(self):
        expected_result: float = self.f_1 * self.f_2

        pk.parallel_for(self.range_policy, self.functor.atomic_mul)
        result: float = self.functor.view1D_mul[0]

        self.assertEqual(expected_result, result)

    def test_atomic_or(self):
        expected_result: int = self.i_1 | self.i_2

        pk.parallel_for(self.range_policy, self.functor.atomic_or)
        result: int = self.functor.view1D_or[0]

        self.assertEqual(expected_result, result)

    def test_atomic_rshift(self):
        expected_result: int = self.i_1 >> self.i_2

        pk.parallel_for(self.range_policy, self.functor.atomic_rshift)
        result: int = self.functor.view1D_rshift[0]

        self.assertEqual(expected_result, result)

    def test_atomic_sub(self):
        expected_result: float = self.f_1 - self.f_2

        pk.parallel_for(self.range_policy, self.functor.atomic_sub)
        result: float = self.functor.view1D_sub[0]

        self.assertEqual(expected_result, result)

    def test_atomic_xor(self):
        expected_result: int = self.i_1 ^ self.i_2

        pk.parallel_for(self.range_policy, self.functor.atomic_xor)
        result: int = self.functor.view1D_xor[0]

        self.assertEqual(expected_result, result)

if __name__ == '__main__':
    unittest.main()
