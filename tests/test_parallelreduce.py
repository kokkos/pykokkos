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


@pk.workunit
def sum_int(i: int, acc: pk.Acc[pk.int64], data: pk.View1D[pk.int64]):
    acc += data[i]


@pk.workunit
def prod_int(i: int, acc: pk.Acc[pk.int64], data: pk.View1D[pk.int64]):
    acc *= data[i]


@pk.workunit
def min_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]):
    if data[i] < acc:
        acc = data[i]


@pk.workunit
def max_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]):
    if data[i] > acc:
        acc = data[i]


@pk.workunit
def band_int(i: int, acc: pk.Acc[pk.int64], data: pk.View1D[pk.int64]):
    acc &= data[i]


@pk.workunit
def bor_int(i: int, acc: pk.Acc[pk.int64], data: pk.View1D[pk.int64]):
    acc |= data[i]


@pk.workunit
def land_bool(i: int, acc: pk.Acc[pk.bool], data: pk.View1D[pk.uint8]):
    acc = acc and data[i]


@pk.workunit
def lor_bool(i: int, acc: pk.Acc[pk.bool], data: pk.View1D[pk.uint8]):
    acc = acc or data[i]


@pk.workunit
def maxloc_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]):
    if data[i] > acc.val:
        acc.val = data[i]
        acc.loc = i


@pk.workunit
def minloc_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]):
    if data[i] < acc.val:
        acc.val = data[i]
        acc.loc = i


@pk.workunit
def minmax_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]):
    if data[i] < acc.min_val:
        acc.min_val = data[i]
    if data[i] > acc.max_val:
        acc.max_val = data[i]


@pk.workunit
def minmaxloc_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]):
    if data[i] < acc.min_val:
        acc.min_val = data[i]
        acc.min_loc = i
    if data[i] > acc.max_val:
        acc.max_val = data[i]
        acc.max_loc = i


def host_view(data):
    return pk.array(data, space=pk.HostSpace)


def openmp_policy(size: int):
    return pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, size)


def assert_value_loc(actual, expected_value, expected_loc):
    if hasattr(actual, "val") and hasattr(actual, "loc"):
        value = actual.val
        loc = actual.loc
    else:
        value, loc = actual

    assert_allclose(value, expected_value)
    assert loc == expected_loc


def assert_minmax(actual, expected_min, expected_max):
    if hasattr(actual, "min_val") and hasattr(actual, "max_val"):
        min_value = actual.min_val
        max_value = actual.max_val
    else:
        min_value, max_value = actual

    assert_allclose(min_value, expected_min)
    assert_allclose(max_value, expected_max)


def assert_minmaxloc(
    actual, expected_min, expected_min_loc, expected_max, expected_max_loc
):
    if all(
        hasattr(actual, field) for field in ("min_val", "min_loc", "max_val", "max_loc")
    ):
        min_value = actual.min_val
        min_loc = actual.min_loc
        max_value = actual.max_val
        max_loc = actual.max_loc
    else:
        min_value, min_loc, max_value, max_loc = actual

    assert_allclose(min_value, expected_min)
    assert min_loc == expected_min_loc
    assert_allclose(max_value, expected_max)
    assert max_loc == expected_max_loc


@pytest.mark.parametrize("series_max", [10, 5000, 90000])
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_squaresum_types(series_max, dtype):
    # check for the ability to match NumPy in
    # sum of squares reductions with various types
    np_data = np.arange(series_max, dtype=dtype)
    expected = np.sum(np_data**2)

    view = pk.array(np_data, space=pk.HostSpace)
    policy = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, series_max)

    if dtype == np.float64:
        actual = pk.parallel_reduce(policy, squaresum_float, data=view)
    elif dtype == np.int64:
        actual = pk.parallel_reduce(policy, squaresum_int, data=view)
    assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "name,reducer,workunit,data,expected",
    [
        ("Sum", pk.Sum, sum_int, np.array([1, 2, 3, 4], dtype=np.int64), 10),
        ("Prod", pk.Prod, prod_int, np.array([1, 2, 3, 4], dtype=np.int64), 24),
        ("Min", pk.Min, min_float, np.array([4.0, 2.0, 9.0], dtype=np.float64), 2.0),
        (
            "Max",
            pk.Max,
            max_float,
            np.array([-5.0, -1.0, -3.0], dtype=np.float64),
            -1.0,
        ),
        (
            "BAnd",
            pk.BAnd,
            band_int,
            np.array([0b1111, 0b1101, 0b0111], dtype=np.int64),
            0b0101,
        ),
        (
            "BOr",
            pk.BOr,
            bor_int,
            np.array([0b1000, 0b0101, 0b0010], dtype=np.int64),
            0b1111,
        ),
        ("LAnd", pk.LAnd, land_bool, np.array([1, 1, 0], dtype=np.uint8), False),
        ("LOr", pk.LOr, lor_bool, np.array([0, 0, 1], dtype=np.uint8), True),
    ],
)
def test_builtin_scalar_reducers(name, reducer, workunit, data, expected):
    actual = pk.parallel_reduce(
        openmp_policy(data.size), workunit, reducer=reducer, data=host_view(data)
    )

    assert_allclose(actual, expected)


def test_builtin_maxloc_reducer():
    data = np.array([1.0, 9.0, 3.0, 7.0], dtype=np.float64)
    actual = pk.parallel_reduce(
        openmp_policy(data.size), maxloc_float, reducer=pk.MaxLoc, data=host_view(data)
    )

    assert_value_loc(actual, 9.0, 1)


def test_builtin_minloc_reducer():
    data = np.array([4.0, -2.0, 8.0, -1.0], dtype=np.float64)
    actual = pk.parallel_reduce(
        openmp_policy(data.size), minloc_float, reducer=pk.MinLoc, data=host_view(data)
    )

    assert_value_loc(actual, -2.0, 1)


def test_builtin_minmax_reducer():
    data = np.array([4.0, -2.0, 8.0, -1.0], dtype=np.float64)
    actual = pk.parallel_reduce(
        openmp_policy(data.size), minmax_float, reducer=pk.MinMax, data=host_view(data)
    )

    assert_minmax(actual, -2.0, 8.0)


def test_builtin_minmaxloc_reducer():
    data = np.array([4.0, -2.0, 8.0, -1.0], dtype=np.float64)
    actual = pk.parallel_reduce(
        openmp_policy(data.size),
        minmaxloc_float,
        reducer=pk.MinMaxLoc,
        data=host_view(data),
    )

    assert_minmaxloc(actual, -2.0, 1, 8.0, 2)


if __name__ == "__main__":
    unittest.main()
