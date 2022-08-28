import unittest

import pykokkos as pk

import numpy as np
import pytest
from numpy.testing import assert_allclose


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


@pk.workload
class SquareSumFloat:
    def __init__(self, n):
        self.N: int = n
        self.total: pk.double = 0

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, self.squaresum)

    @pk.workunit
    def squaresum(self, i: float, acc: pk.Acc[pk.double]):
        acc += i * i


@pk.workload
class SquareSumInt:
    def __init__(self, n):
        self.N: int = n
        self.total: pk.int64 = 0

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, self.squaresum)

    @pk.workunit
    def squaresum(self, i: pk.int64, acc: pk.Acc[pk.int64]):
        acc += i * i


@pytest.mark.parametrize("series_max", [10, 5000, 90000])
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_squaresum_types(series_max, dtype):
    # check for the ability to match NumPy in
    # sum of squares reductions with various types
    expected = np.sum(np.arange(series_max, dtype=dtype) ** 2)
    if dtype == np.float64:
        ss_instance = SquareSumFloat(series_max)
    elif dtype == np.int64:
        ss_instance = SquareSumInt(series_max)
    pk.execute(pk.ExecutionSpace.OpenMP, ss_instance)
    actual = ss_instance.total
    assert_allclose(actual, expected)

if __name__ == '__main__':
    unittest.main()
