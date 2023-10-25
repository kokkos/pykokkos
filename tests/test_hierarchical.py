import unittest

import pykokkos as pk


# Tests for correctness of hierarchical (nested) parallelism
@pk.functor
class HierarchicalTestFunctor:
    def __init__(self, N: int, M: int, E: int, value: int):
        self.N: int = N
        self.M: int = M
        self.E: int = E
        self.value: int = value

        self.y: pk.View1D[pk.int32] = pk.View([N], pk.int32)
        self.x: pk.View1D[pk.int32] = pk.View([M], pk.int32)
        self.A: pk.View2D[pk.int32] = pk.View([N, M], pk.int32)

        self.yprime: pk.View2D[pk.int32] = pk.View([N, N], pk.int32)
        self.for_view: pk.View1D[pk.int32] = pk.View([N], pk.int32)

        self.y_vector: pk.View2D[pk.int32] = pk.View([E, N], pk.int32)
        self.x_vector: pk.View2D[pk.int32] = pk.View([E, M], pk.int32)
        self.A_vector: pk.View3D[pk.int32] = pk.View([E, N, M], pk.int32)

        for i in range(N):
            self.y[i] = value

        for i in range(M):
            self.x[i] = value

        for j in range(N):
            for i in range(M):
                self.A[j][i] = value
            for i in range(N):
                self.yprime[j][i] = value

        for e in range(E):
            for i in range(N):
                self.y_vector[e][i] = value

            for i in range(M):
                self.x_vector[e][i] = value

            for j in range(N):
                for i in range(M):
                    self.A_vector[e][j][i] = value

    @pk.workunit
    def yAx(self, team_member: pk.TeamMember, acc: pk.Acc[pk.double]) -> None:
        j: int = team_member.league_rank()

        def inner_reduce(i: int, inner_acc: pk.Acc[pk.double]):
            inner_acc += self.A[j][i] * self.x[i]

        temp2: float = pk.parallel_reduce(pk.TeamThreadRange(team_member, self.M), inner_reduce)

        if team_member.team_rank() == 0:
            acc += self.y[j] * temp2

    @pk.workunit
    def yAx_plus1(self, team_member: pk.TeamMember, acc: pk.Acc[pk.double]) -> None:
        j: int = team_member.league_rank()

        def inner_reduce(i: int, inner_acc: pk.Acc[pk.double]):
            inner_acc += self.A[j][i] * self.x[i]

        def inner_for(i: int):
            self.yprime[j][i] += 1

        temp2: float = pk.parallel_reduce(pk.TeamThreadRange(team_member, self.M), inner_reduce)
        pk.parallel_for(pk.TeamThreadRange(team_member, self.N), inner_for)

        if team_member.team_rank() == 0:
            acc += self.yprime[j][j] * temp2

    @pk.workunit
    def outer_for(self, team_member: pk.TeamMember) -> None:
        j: int = team_member.league_rank()

        def inner_reduce(i: int, acc: pk.Acc[pk.double]):
            acc += self.value

        if team_member.team_rank() == 0:
            temp: float = pk.parallel_reduce(pk.TeamThreadRange(team_member, self.M), inner_reduce)
            self.for_view[j] = temp

    @pk.workunit
    def yAx_vector(self, team_member: pk.TeamMember, acc: pk.Acc[pk.double]) -> None:
        e: int = team_member.league_rank()

        def team_reduce(j: int, team_acc: pk.Acc[pk.double]):
            def vector_reduce(i: int, vector_acc: pk.Acc[pk.double]):
                vector_acc += self.A_vector[e][j][i] * self.x_vector[e][i]

            tempM: float = pk.parallel_reduce(pk.ThreadVectorRange(team_member, self.M), vector_reduce)

            team_acc += self.y_vector[e][j] * tempM

        tempN: float = pk.parallel_reduce(
            pk.TeamThreadRange(team_member, self.N), team_reduce)

        def single_closure():
            nonlocal acc
            acc += tempN

        pk.single(pk.PerTeam(team_member), single_closure)


class TestHierarchical(unittest.TestCase):
    def setUp(self):
        self.N: int = 64
        self.M: int = 128
        self.E: int = 256
        self.value: int = 1

        self.y = pk.View([self.N], pk.int32)
        self.x = pk.View([self.M], pk.int32)
        self.A = pk.View([self.N, self.M], pk.int32)

        for i in range(self.N):
            self.y[i] = self.value

        for i in range(self.M):
            self.x[i] = self.value

        for j in range(self.N):
            for i in range(self.M):
                self.A[j][i] = self.value

        self.y_vector = pk.View([self.E, self.N], pk.int32)
        self.x_vector = pk.View([self.E, self.M], pk.int32)
        self.A_vector = pk.View([self.E, self.E, self.M], pk.int32)

        for e in range(self.E):
            for i in range(self.N):
                self.y_vector[e][i] = self.value

            for i in range(self.M):
                self.x_vector[e][i] = self.value

            for j in range(self.E):
                for i in range(self.M):
                    self.A_vector[e][j][i] = self.value

        self.functor = HierarchicalTestFunctor(self.N, self.M, self.E, self.value)
        self.execution_space = pk.ExecutionSpace.Default

    def test_yAx(self):
        expected_result: float = 0
        for j in range(self.N):
            temp2: float = 0
            for i in range(self.M):
                temp2 += self.A[j][i] * self.x[i]
            expected_result += self.y[j] * temp2

        result: int = pk.parallel_reduce(pk.TeamPolicy(self.execution_space, self.N, pk.AUTO), self.functor.yAx)

        self.assertEqual(expected_result, result)

    def test_yAx_plus1(self):
        expected_result: float = 0
        for j in range(self.N):
            temp2: float = 0
            for i in range(self.M):
                temp2 += self.A[j][i] * self.x[i]
            expected_result += (self.y[j] + 1) * temp2

        result: int = pk.parallel_reduce(pk.TeamPolicy(self.execution_space, self.N, pk.AUTO), self.functor.yAx_plus1)

        self.assertEqual(expected_result, result)

    def test_outer_for(self):
        expected_result: float = 0
        for i in range(self.M):
            expected_result += self.value

        pk.parallel_for(pk.TeamPolicy(self.execution_space, self.N, pk.AUTO), self.functor.outer_for)
        for i in range(self.N):
            result: int = self.functor.for_view[i]
            self.assertEqual(expected_result, result)

    def test_yAx_vector(self):
        expected_result: float = 0
        for e in range(self.E):
            tempN: float = 0

            for j in range(self.N):
                tempM: float = 0

                for i in range(self.M):
                    tempM += self.A_vector[e][j][i] * self.x_vector[e][i]

                tempN += self.y_vector[e][j] * tempM

            expected_result += tempN

        result: float = pk.parallel_reduce(pk.TeamPolicy(self.execution_space, self.E, pk.AUTO, 32), self.functor.yAx_vector)

        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
