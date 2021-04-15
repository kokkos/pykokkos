import unittest

import pykokkos as pk


# Tests for correctness of pk.parallel_reduce
@pk.functor
class Add1DTestReduceFunctor:
    def __init__(self, threads: int, value: int):
        self.threads: int = threads
        self.value: int = value

    @pk.workunit
    def add(self, tid: int, acc: pk.Acc[pk.double]) -> None:
        acc += self.value

    @pk.workunit
    def add_squares(self, tid: int, acc: pk.Acc[float]) -> None:
        acc += self.value * self.value


class TestParallelReduce(unittest.TestCase):
    def setUp(self):
        self.threads: int = 50
        self.value: int = 7

        self.functor = Add1DTestReduceFunctor(self.threads, self.value)
        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)

    def test_add(self):
        expected_result: int = self.value * self.threads
        result: int = pk.parallel_reduce("reduction", self.range_policy, self.functor.add)

        self.assertEqual(expected_result, result)

    def test_add_squares(self):
        expected_result: int = self.value * self.value * self.threads
        result: int = pk.parallel_reduce(self.range_policy, self.functor.add_squares)

        self.assertEqual(expected_result, result)


if __name__ == '__main__':
    unittest.main()
