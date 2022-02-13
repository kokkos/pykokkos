import unittest

import pykokkos as pk


# Tests for correctness of pk.parallel_scan
@pk.functor
class Add1DTestScanFunctor:
    def __init__(self, threads: int, value: int):
        self.threads: int = threads
        self.value: int = value
        self.view: pk.View1D[pk.double] = pk.View([threads], pk.double)

    @pk.workunit
    def add(self, tid: int, acc: pk.Acc[pk.double], last_pass: bool) -> None:
        acc += self.value
        if last_pass:
            self.view[tid] = acc

    @pk.workunit
    def add_squares(self, tid: int, acc: pk.Acc[pk.double], last_pass: bool) -> None:
        acc += self.value * self.value
        if last_pass:
            self.view[tid] = acc


class TestParallelScan(unittest.TestCase):
    def setUp(self):
        self.threads: int = 50
        self.value: int = 7

        self.functor = Add1DTestScanFunctor(self.threads, self.value)
        self.range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, self.threads)

    def test_add(self):
        expected_result: int = self.value * self.threads
        result: int = pk.parallel_scan(self.range_policy, self.functor.add)

        self.assertEqual(expected_result, result)
        expected_result = 0
        for i in range(self.threads):
            result: int = self.functor.view[i]
            expected_result += self.value
            self.assertEqual(result, expected_result)

    def test_add_squares(self):
        expected_result: int = self.value * self.value * self.threads
        result: int = pk.parallel_scan(self.range_policy, self.functor.add_squares)

        self.assertEqual(expected_result, result)
        expected_result = 0
        for i in range(self.threads):
            result: int = self.functor.view[i]
            expected_result += self.value * self.value
            self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
