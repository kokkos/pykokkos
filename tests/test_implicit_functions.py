import unittest

import pykokkos as pk

from tests.implicit_functions_helper import external_helper


@pk.function
def implicit_leaf(i: int) -> int:
    return i + 1


@pk.function
def implicit_helper(i: int) -> int:
    return implicit_leaf(i) * 2


def unrelated_host_function() -> None:
    raise RuntimeError("This function must not be translated")


def log(value):
    raise RuntimeError("This host wrapper must not shadow the math intrinsic")


@pk.workunit
def implicit_workunit(i: int, acc: pk.Acc[pk.int64]) -> None:
    acc += implicit_helper(i)


@pk.workunit
def intrinsic_workunit(i: int, view: pk.View1D[pk.double]) -> None:
    view[i] = log(view[i])


@pk.workunit
def cross_module_workunit(i: int, acc: pk.Acc[pk.int64]) -> None:
    acc += external_helper(i)


def make_implicit_workunit(functor):
    @pk.workunit
    def workunit(i: int, acc: pk.Acc[pk.int64]) -> None:
        acc += functor(i)

    return workunit


class TestImplicitFunctions(unittest.TestCase):
    def test_executes_implicit_functions(self):
        result = pk.parallel_reduce(10, implicit_workunit)

        self.assertEqual(110, result)

    def test_executes_nested_workunit_with_captured_function(self):
        workunit = make_implicit_workunit(implicit_helper)

        result = pk.parallel_reduce(10, workunit)

        self.assertEqual(110, result)

    def test_executes_function_from_another_module(self):
        result = pk.parallel_reduce(10, cross_module_workunit)

        self.assertEqual(75, result)


if __name__ == "__main__":
    unittest.main()
