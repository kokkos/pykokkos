import unittest

import pykokkos as pk


# Tests for correctness of pk.parallel_reduce lambda
@pk.workload
class Add1DTestReduce:
    def __init__(self, threads: int, value: int):
        self.threads: int = threads
        self.value: int = value
        self.sum: int = 0

    @pk.main
    def run(self):
        self.sum = pk.parallel_reduce("name", self.threads, lambda i, acc: acc + self.value)


@pk.workload
class Add1DSquareTestReduce:
    def __init__(self, threads: int, value: int):
        self.threads: int = threads
        self.value: int = value
        self.sum: int = 0

    @pk.main
    def run(self):
        self.sum = pk.parallel_reduce(self.threads, lambda i, acc: acc + self.value * self.value)


@pk.workload
class Add1DTestFor:
    def __init__(self, threads: int, initial_value: int, added_value: int):
        self.threads: int = threads
        self.initial_value: int = initial_value
        self.added_value: int = added_value
        self.view: pk.View1D[pk.int32] = pk.View([threads], pk.int32)
  
    @pk.main
    def run(self):
        pk.parallel_for("name", self.threads, lambda i: self.initial_value, self.view)
        pk.parallel_for(self.threads, lambda i: self.view[i] + self.added_value, self.view)


class TestLambda(unittest.TestCase):
    def setUp(self):
        self.threads: int = 50
        self.value: int = 7

    def test_add(self):
        expected_result: int = self.value * self.threads

        workload = Add1DTestReduce(self.threads, self.value)
        pk.execute(pk.ExecutionSpace.Default, workload)
        result: int = workload.sum
        self.assertEqual(expected_result, result)

    def test_add_squares(self):
        expected_result: int = self.value * self.value * self.threads

        workload = Add1DSquareTestReduce(self.threads, self.value)
        pk.execute(pk.ExecutionSpace.Default, workload)
        result: int = workload.sum
        self.assertEqual(expected_result, result)

    def test_add_for(self):
        initial_value: int = 5
        added_value: int = 7
        expected_result: int = initial_value + added_value

        workload = Add1DTestFor(self.threads, initial_value, added_value)
        pk.execute(pk.ExecutionSpace.Default, workload)

        for i in range(self.threads):
            result: int = workload.view[i]
            self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
