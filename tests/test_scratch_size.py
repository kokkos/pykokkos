import unittest

import pykokkos as pk


@pk.workunit
def scratch_reduce_workunit(
    team_member: pk.TeamMember,
    acc: pk.Acc[pk.double],
    input_view: pk.View1D[pk.double],
    M: int,
):
    """
    Workunit that uses scratch memory to store intermediate values.
    Verifies scratch memory allocation by using it in computation.
    """
    team_rank: int = team_member.team_rank()
    league_rank: int = team_member.league_rank()

    # Allocate scratch memory
    scratch: pk.ScratchView1D[pk.double] = pk.ScratchView1D(
        team_member.team_scratch(0), M
    )

    # Initialize scratch memory with values from input_view
    def init_scratch(i: int):
        if league_rank < input_view.extent(0) and i < M:
            scratch[i] = input_view[league_rank]
        else:
            scratch[i] = 0.0

    if team_rank == 0:
        pk.parallel_for(pk.ThreadVectorRange(team_member, M), init_scratch)

    team_member.team_barrier()

    # Sum values in scratch memory
    def sum_scratch(i: int, inner_acc: pk.Acc[pk.double]):
        inner_acc += scratch[i]

    local_sum: float = pk.parallel_reduce(
        pk.ThreadVectorRange(team_member, M), sum_scratch
    )

    # Accumulate result
    if team_rank == 0:
        acc += local_sum


class TestScratchSize(unittest.TestCase):
    def setUp(self):
        self.execution_space = pk.ExecutionSpace.OpenMP
        pk.set_default_space(self.execution_space)

        self.E: int = 8  # Number of teams
        self.M: int = 16  # Scratch memory size per team

    def test_scratch_size_inline_policy(self):
        """
        Test that scratch memory works with inline policy creation.
        Verifies scratch memory allocation by using it in computation.
        """
        input_view = pk.View([self.E], pk.double)
        for i in range(self.E):
            input_view[i] = float(i + 1)

        # Calculate expected result: sum of all input values * M
        expected_result: float = sum(input_view[i] for i in range(self.E)) * self.M

        # Calculate scratch size needed (M * sizeof(double) = M * 8 bytes)
        scratch_size: int = self.M * 8

        # Test with inline policy creation
        result: float = pk.parallel_reduce(
            "scratch_reduce_inline",
            pk.TeamPolicy(self.execution_space, self.E, "auto", 32).set_scratch_size(
                0, pk.PerTeam(scratch_size)
            ),
            scratch_reduce_workunit,
            acc=0.0,
            input_view=input_view,
            M=self.M,
        )

        self.assertAlmostEqual(expected_result, result, places=5)

    def test_scratch_size_precreated_policy(self):
        """
        Test that scratch memory works with pre-created policy.
        Verifies scratch memory allocation by using it in computation.
        """
        input_view = pk.View([self.E], pk.double)
        for i in range(self.E):
            input_view[i] = float(i + 1)

        # Calculate expected result: sum of all input values * M
        expected_result: float = sum(input_view[i] for i in range(self.E)) * self.M

        # Calculate scratch size needed (M * sizeof(double) = M * 8 bytes)
        scratch_size: int = self.M * 8

        # Test with pre-created policy
        teams = pk.TeamPolicy(
            self.execution_space, self.E, "auto", 32
        ).set_scratch_size(0, pk.PerTeam(scratch_size))

        result: float = pk.parallel_reduce(
            "scratch_reduce_precreated",
            teams,
            scratch_reduce_workunit,
            acc=0.0,
            input_view=input_view,
            M=self.M,
        )

        self.assertAlmostEqual(expected_result, result, places=5)

    def test_scratch_size_per_thread(self):
        """
        Test that PerThread scratch memory works correctly.
        """
        input_view = pk.View([self.E], pk.double)
        for i in range(self.E):
            input_view[i] = float(i + 1)

        # Calculate expected result: sum of all input values * M
        expected_result: float = sum(input_view[i] for i in range(self.E)) * self.M

        # Calculate scratch size needed (PerThread) - M * sizeof(double) = M * 8 bytes
        scratch_size: int = self.M * 8

        # Test with PerThread scratch size
        result: float = pk.parallel_reduce(
            "scratch_reduce_perthread",
            pk.TeamPolicy(self.execution_space, self.E, "auto", 32).set_scratch_size(
                0, pk.PerThread(scratch_size)
            ),
            scratch_reduce_workunit,
            acc=0.0,
            input_view=input_view,
            M=self.M,
        )

        self.assertAlmostEqual(expected_result, result, places=5)

    def test_scratch_size_multiple_iterations(self):
        """
        Test that scratch memory works correctly across multiple iterations
        with a pre-created policy.
        """
        input_view = pk.View([self.E], pk.double)
        for i in range(self.E):
            input_view[i] = float(i + 1)

        expected_result: float = sum(input_view[i] for i in range(self.E)) * self.M
        scratch_size: int = self.M * 8

        # Pre-create policy
        teams = pk.TeamPolicy(
            self.execution_space, self.E, "auto", 32
        ).set_scratch_size(0, pk.PerTeam(scratch_size))

        # Run multiple iterations
        nrepeat: int = 3
        for i in range(nrepeat):
            result: float = pk.parallel_reduce(
                "scratch_reduce_multiple",
                teams,
                scratch_reduce_workunit,
                acc=0.0,
                input_view=input_view,
                M=self.M,
            )
            self.assertAlmostEqual(expected_result, result, places=5)


if __name__ == "__main__":
    unittest.main()
