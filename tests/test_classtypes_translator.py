import unittest

import pykokkos as pk


@pk.classtype
class DepOne:
    def __init__(self, i: int, j: float, k: bool):
        self.i: int = i
        self.j: float = j
        self.k: bool = k

    def sum(self) -> float:
        if self.k:
            return self.i + self.j
        else:
            return self.j


@pk.classtype
class DepTwo:
    def __init__(self, dep_one: DepOne):
        self.dep_one: DepOne = DepOne(dep_one.i, dep_one.j, dep_one.k)

    def sum(self) -> float:
        return self.dep_one.sum()

    def get_dep_one(self) -> DepOne:
        return self.dep_one


# Tests translation of classtypes to C++
@pk.functor
class ClasstypesTestFunctor:
    def __init__(self, threads: int, i_1: int, i_2: int, f_1: float, b_1: bool):
        self.threads: int = threads
        self.i_1: int = i_1
        self.i_2: int = i_2
        self.f_1: float = f_1
        self.b_1: bool = b_1

    @pk.workunit
    def dep_one_work(self, i: int, acc: pk.Acc[pk.double]) -> None:
        dep_one: DepOne = DepOne(self.i_1, self.f_1, self.b_1)
        acc += dep_one.sum()

    @pk.function
    def mutate(self, dep_one: DepOne) -> None:
        dep_one.i = self.i_2

    @pk.workunit
    def dep_one_mutate(self, i: int, acc: pk.Acc[pk.double]) -> None:
        dep_one: DepOne = DepOne(self.i_1, self.f_1, self.b_1)
        self.mutate(dep_one)
        acc += dep_one.sum()

    @pk.function
    def ret(self) -> DepOne:
        return DepOne(self.i_1, self.f_1, self.b_1)

    @pk.workunit
    def dep_one_return(self, i: int, acc: pk.Acc[pk.double]) -> None:
        dep_one: DepOne = self.ret()
        acc += dep_one.sum()

    @pk.workunit
    def dep_two_work(self, i: int, acc: pk.Acc[pk.double]) -> None:
        dep_one: DepOne = DepOne(self.i_1, self.f_1, self.b_1)
        dep_two: DepTwo = DepTwo(dep_one)
        acc += dep_two.sum()

    @pk.workunit
    def dep_two_mutate(self, i: int, acc: pk.Acc[pk.double]) -> None:
        dep_one: DepOne = DepOne(self.i_1, self.f_1, self.b_1)
        dep_two: DepTwo = DepTwo(dep_one)

        dep_one.i = self.i_2

        acc += dep_two.sum()

    @pk.workunit
    def dep_two_get1(self, i: int, acc: pk.Acc[pk.double]) -> None:
        dep_one: DepOne = DepOne(self.i_1, self.f_1, self.b_1)
        dep_two: DepTwo = DepTwo(dep_one)
        dep_one = dep_two.get_dep_one()

        dep_one.i = self.i_2

        acc += dep_two.sum()

    @pk.workunit
    def dep_two_get2(self, i: int, acc: pk.Acc[pk.double]) -> None:
        dep_one: DepOne = DepOne(self.i_1, self.f_1, self.b_1)
        dep_two: DepTwo = DepTwo(dep_one)
        dep_one = dep_two.dep_one

        dep_one.i = self.i_2

        acc += dep_two.sum()


class TestDependenciesTranslator(unittest.TestCase):
    def setUp(self):
        self.threads: int = 50
        self.i_1: int = 5
        self.i_2: int = 1
        self.f_1: float = 5.5
        self.b_1: bool = True

        self.functor = ClasstypesTestFunctor(self.threads,
                                            self.i_1, self.i_2,
                                            self.f_1, self.b_1)
        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)

    def test_dep_one(self):
        dep_one = DepOne(self.i_1, self.f_1, self.b_1)
        expected_result: float = self.threads * dep_one.sum()
        result: float = pk.parallel_reduce(self.range_policy, self.functor.dep_one_work)

        self.assertEqual(expected_result, result)

    def test_dep_one_mutate(self):
        dep_one = DepOne(self.i_1, self.f_1, self.b_1)
        dep_one.i = self.i_2
        expected_result: float = self.threads * dep_one.sum()
        result: float = pk.parallel_reduce(self.range_policy, self.functor.dep_one_mutate)

        self.assertEqual(expected_result, result)

    def test_dep_one_return(self):
        dep_one = DepOne(self.i_1, self.f_1, self.b_1)
        expected_result: float = self.threads * dep_one.sum()
        result: float = pk.parallel_reduce(self.range_policy, self.functor.dep_one_return)

        self.assertEqual(expected_result, result)

    def test_dep_two(self):
        dep_two = DepTwo(DepOne(self.i_1, self.f_1, self.b_1))
        expected_result: float = self.threads * dep_two.sum()
        result: float = pk.parallel_reduce(self.range_policy, self.functor.dep_two_work)

        self.assertEqual(expected_result, result)

    def test_dep_two_mutate(self):
        dep_one = DepOne(self.i_1, self.f_1, self.b_1)
        dep_two = DepTwo(dep_one)
        dep_one.i = self.i_2
        expected_result: float = self.threads * dep_two.sum()
        result: float = pk.parallel_reduce(self.range_policy, self.functor.dep_two_mutate)

        self.assertEqual(expected_result, result)

    # def test_dep_two_get1(self):
    #     dep_one = DepOne(self.i_1, self.f_1, self.b_1)
    #     dep_two = DepTwo(dep_one)
    #     dep_one = dep_two.get_dep_one()
    #     dep_one.i = self.i_2
    #     expected_result: float = self.threads * dep_two.sum()
    #     result: float = self.workload.dep_two_get1_sum

    #     self.assertEqual(expected_result, result)

    # def test_dep_two_get2(self):
    #     dep_one = DepOne(self.i_1, self.f_1, self.b_1)
    #     dep_two = DepTwo(dep_one)
    #     dep_one = dep_two.dep_one
    #     dep_one.i = self.i_2
    #     expected_result: float = self.threads * dep_two.sum()
    #     result: float = self.workload.dep_two_get2_sum

    #     self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
