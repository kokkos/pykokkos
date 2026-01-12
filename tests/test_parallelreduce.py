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
        result: int = pk.parallel_reduce(
            "reduction", self.range_policy, self.functor.add
        )

        self.assertEqual(expected_result, result)

    def test_add_squares(self):
        expected_result: int = self.value * self.value * self.threads
        result: int = pk.parallel_reduce(self.range_policy, self.functor.add_squares)

        self.assertEqual(expected_result, result)


@pk.workunit
def squaresum_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]):
    val: float = data[i]
    acc += val * val


@pk.workunit
def squaresum_int(i: int, acc: pk.Acc[pk.int64], data: pk.View1D[pk.int64]):
    val: pk.int64 = data[i]
    acc += val * val


@pytest.mark.parametrize("series_max", [10, 5000, 90000])
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_squaresum_types(series_max, dtype):
    # check for the ability to match NumPy in
    # sum of squares reductions with various types
    np_data = np.arange(series_max, dtype=dtype)
    expected = np.sum(np_data**2)

    view = pk.array(np_data)
    policy = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, series_max)

    if dtype == np.float64:
        actual = pk.parallel_reduce(policy, squaresum_float, data=view)
    elif dtype == np.int64:
        actual = pk.parallel_reduce(policy, squaresum_int, data=view)
    assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
