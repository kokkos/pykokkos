import unittest

import pykokkos as pk

from pykokkos.core.parsers import Parser
from pykokkos.core.translators import PyKokkosMembers


def implicit_leaf(i: int) -> int:
    return i + 1


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


class TestImplicitFunctions(unittest.TestCase):
    def test_discovers_only_reachable_functions(self):
        parser = Parser(__file__)
        entity = parser.get_entity("implicit_workunit")
        members = PyKokkosMembers()

        members.extract(entity, [])

        names = {function.declname for function in members.pk_functions}
        self.assertEqual({"implicit_helper", "implicit_leaf"}, names)

    def test_math_intrinsic_takes_precedence_over_same_name_function(self):
        parser = Parser(__file__)
        entity = parser.get_entity("intrinsic_workunit")
        members = PyKokkosMembers()

        members.extract(entity, [])

        names = {function.declname for function in members.pk_functions}
        self.assertNotIn("log", names)

    def test_executes_implicit_functions(self):
        result = pk.parallel_reduce(10, implicit_workunit)

        self.assertEqual(110, result)


if __name__ == "__main__":
    unittest.main()
