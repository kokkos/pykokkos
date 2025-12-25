import unittest
import numpy as np

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


class TestScratchViewShmemSizeFail(unittest.TestCase):
    """
    Tests that scratch view size selection fails when the type is not specified.
    """

    def test_scratch_view_unparameterized_error(self):
        """Test that unparameterized ScratchView raises error"""
        try:
            result = pk.ScratchView1D.shmem_size(10)
            print(f"No exception raised. Result: {result}")
        except Exception as e:
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {e}")

        with self.assertRaises(TypeError):
            pk.ScratchView1D.shmem_size(10)


class TestScratchViewShmemSize(unittest.TestCase):
    """
    Unit tests for ScratchView shmem_size methods.
    Tests correctness of return values for all dimensions and types.
    """

    def _calculate_expected_size(self, dtype_or_size, *dims: int, alignment: int = 8) -> int:
        """
        Helper to calculate expected scratch size with alignment.

        :param dtype_or_size: Either a numpy dtype class (e.g., np.float64) or an int size in bytes
        :param dims: Dimensions of the scratch view
        :param alignment: Alignment requirement (default 8 bytes)
        :returns: Total scratch memory size in bytes
        """
        if isinstance(dtype_or_size, int):
            type_size = dtype_or_size
        else:
            # Assume it's a numpy dtype class, convert to dtype instance and get itemsize
            type_size = int(np.dtype(dtype_or_size).itemsize)

        total_elements = 1
        for dim in dims:
            total_elements *= dim
        raw_size = total_elements * type_size
        aligned_size = ((raw_size + alignment - 1) // alignment) * alignment
        return aligned_size

    # Test ScratchView1D
    def test_scratch_view_1d_float(self):
        """Test ScratchView1D[float].shmem_size"""
        dim = 10
        result = pk.ScratchView1D[float].shmem_size(dim)
        expected = self._calculate_expected_size(np.float64, dim)  # float is float64
        self.assertEqual(result, expected)

    def test_scratch_view_1d_pk_float(self):
        """Test ScratchView1D[pk.float].shmem_size"""
        dim = 10
        result = pk.ScratchView1D[pk.float].shmem_size(dim)
        expected = self._calculate_expected_size(np.float32, dim)  # pk.float is float32
        self.assertEqual(result, expected)

    def test_scratch_view_1d_pk_double(self):
        """Test ScratchView1D[pk.double].shmem_size"""
        dim = 10
        result = pk.ScratchView1D[pk.double].shmem_size(dim)
        expected = self._calculate_expected_size(np.float64, dim)  # pk.double is float64
        self.assertEqual(result, expected)

    def test_scratch_view_1d_int(self):
        """Test ScratchView1D[int].shmem_size"""
        dim = 10
        result = pk.ScratchView1D[int].shmem_size(dim)
        expected = self._calculate_expected_size(np.int32, dim)  # int is int32
        self.assertEqual(result, expected)

    def test_scratch_view_1d_pk_int32(self):
        """Test ScratchView1D[pk.int32].shmem_size"""
        dim = 10
        result = pk.ScratchView1D[pk.int32].shmem_size(dim)
        expected = self._calculate_expected_size(np.int32, dim)  # int32
        self.assertEqual(result, expected)

    def test_scratch_view_1d_pk_int64(self):
        """Test ScratchView1D[pk.int64].shmem_size"""
        dim = 10
        result = pk.ScratchView1D[pk.int64].shmem_size(dim)
        expected = self._calculate_expected_size(np.int64, dim)  # int64
        self.assertEqual(result, expected)

    def test_scratch_view_1d_pk_uint8(self):
        """Test ScratchView1D[pk.uint8].shmem_size"""
        dim = 10
        result = pk.ScratchView1D[pk.uint8].shmem_size(dim)
        expected = self._calculate_expected_size(np.uint8, dim)  # uint8
        self.assertEqual(result, expected)

    def test_scratch_view_1d_alignment(self):
        """Test that ScratchView1D properly aligns to 8 bytes"""
        # 3 elements of float32 (4 bytes) = 12 bytes, should align to 16 bytes
        dim = 3
        result = pk.ScratchView1D[pk.float].shmem_size(dim)
        expected = self._calculate_expected_size(np.float32, dim)  # 12 bytes -> 16 bytes aligned
        self.assertEqual(result, expected)
        # Explicit check: 3 * 4 = 12, aligned to 8 = 16
        self.assertEqual(result, 16)

    # Test ScratchView2D
    def test_scratch_view_2d_float(self):
        """Test ScratchView2D[float].shmem_size"""
        dim0, dim1 = 5, 10
        result = pk.ScratchView2D[float].shmem_size(dim0, dim1)
        expected = self._calculate_expected_size(np.float64, dim0, dim1)
        self.assertEqual(result, expected)

    def test_scratch_view_2d_pk_float(self):
        """Test ScratchView2D[pk.float].shmem_size"""
        dim0, dim1 = 5, 10
        result = pk.ScratchView2D[pk.float].shmem_size(dim0, dim1)
        expected = self._calculate_expected_size(np.float32, dim0, dim1)
        self.assertEqual(result, expected)

    def test_scratch_view_2d_alignment(self):
        """Test that ScratchView2D properly aligns"""
        # 3*3*4 = 36 bytes, should align to 40 bytes
        dim0, dim1 = 3, 3
        result = pk.ScratchView2D[pk.float].shmem_size(dim0, dim1)
        expected = self._calculate_expected_size(np.float32, dim0, dim1)
        self.assertEqual(result, expected)
        # Explicit check: 3*3*4 = 36, aligned to 8 = 40
        self.assertEqual(result, 40)

    # Test ScratchView3D
    def test_scratch_view_3d_float(self):
        """Test ScratchView3D[float].shmem_size"""
        dim0, dim1, dim2 = 2, 3, 4
        result = pk.ScratchView3D[float].shmem_size(dim0, dim1, dim2)
        expected = self._calculate_expected_size(np.float64, dim0, dim1, dim2)
        self.assertEqual(result, expected)

    def test_scratch_view_3d_pk_double(self):
        """Test ScratchView3D[pk.double].shmem_size"""
        dim0, dim1, dim2 = 2, 3, 4
        result = pk.ScratchView3D[pk.double].shmem_size(dim0, dim1, dim2)
        expected = self._calculate_expected_size(np.float64, dim0, dim1, dim2)
        self.assertEqual(result, expected)

    # Test ScratchView4D
    def test_scratch_view_4d_float(self):
        """Test ScratchView4D[float].shmem_size"""
        dim0, dim1, dim2, dim3 = 2, 2, 2, 2
        result = pk.ScratchView4D[float].shmem_size(dim0, dim1, dim2, dim3)
        expected = self._calculate_expected_size(np.float64, dim0, dim1, dim2, dim3)
        self.assertEqual(result, expected)

    def test_scratch_view_4d_pk_int32(self):
        """Test ScratchView4D[pk.int32].shmem_size"""
        dim0, dim1, dim2, dim3 = 2, 2, 2, 2
        result = pk.ScratchView4D[pk.int32].shmem_size(dim0, dim1, dim2, dim3)
        expected = self._calculate_expected_size(np.int32, dim0, dim1, dim2, dim3)
        self.assertEqual(result, expected)

    # Test ScratchView5D
    def test_scratch_view_5d_float(self):
        """Test ScratchView5D[float].shmem_size"""
        dim0, dim1, dim2, dim3, dim4 = 2, 2, 2, 2, 2
        result = pk.ScratchView5D[float].shmem_size(dim0, dim1, dim2, dim3, dim4)
        expected = self._calculate_expected_size(np.float64, dim0, dim1, dim2, dim3, dim4)
        self.assertEqual(result, expected)

    # Test ScratchView6D
    def test_scratch_view_6d_float(self):
        """Test ScratchView6D[float].shmem_size"""
        dim0, dim1, dim2, dim3, dim4, dim5 = 2, 2, 2, 2, 2, 2
        result = pk.ScratchView6D[float].shmem_size(dim0, dim1, dim2, dim3, dim4, dim5)
        expected = self._calculate_expected_size(np.float64, dim0, dim1, dim2, dim3, dim4, dim5)
        self.assertEqual(result, expected)

    # Test ScratchView7D
    def test_scratch_view_7d_float(self):
        """Test ScratchView7D[float].shmem_size"""
        dim0, dim1, dim2, dim3, dim4, dim5, dim6 = 2, 2, 2, 2, 2, 2, 2
        result = pk.ScratchView7D[float].shmem_size(dim0, dim1, dim2, dim3, dim4, dim5, dim6)
        expected = self._calculate_expected_size(np.float64, dim0, dim1, dim2, dim3, dim4, dim5, dim6)
        self.assertEqual(result, expected)

    # Test ScratchView8D
    def test_scratch_view_8d_float(self):
        """Test ScratchView8D[float].shmem_size"""
        dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7 = 2, 2, 2, 2, 2, 2, 2, 2
        result = pk.ScratchView8D[float].shmem_size(dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7)
        expected = self._calculate_expected_size(np.float64, dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7)
        self.assertEqual(result, expected)

    # Test edge cases
    def test_scratch_view_1d_small_size(self):
        """Test ScratchView1D with small size"""
        dim = 1
        result = pk.ScratchView1D[float].shmem_size(dim)
        expected = self._calculate_expected_size(np.float64, dim)
        self.assertEqual(result, expected)
        # Explicit check: 1 * 8 = 8, already aligned
        self.assertEqual(result, int(np.dtype(np.float64).itemsize))

    def test_scratch_view_1d_large_size(self):
        """Test ScratchView1D with large size"""
        dim = 1000
        result = pk.ScratchView1D[float].shmem_size(dim)
        expected = self._calculate_expected_size(np.float64, dim)
        self.assertEqual(result, expected)
        # Explicit check: 1000 * 8 = 8000, already aligned
        self.assertEqual(result, 1000 * int(np.dtype(np.float64).itemsize))

    def test_scratch_view_1d_odd_alignment(self):
        """Test ScratchView1D with size requiring alignment"""
        # 5 elements of float32 = 20 bytes, should align to 24 bytes
        dim = 5
        result = pk.ScratchView1D[pk.float].shmem_size(dim)
        expected = self._calculate_expected_size(np.float32, dim)
        self.assertEqual(result, expected)
        # Explicit check: 5 * 4 = 20, aligned to 8 = 24
        self.assertEqual(result, 24)

    # Test different integer types
    def test_scratch_view_1d_all_int_types(self):
        """Test ScratchView1D with all integer types"""
        dim = 10
        test_cases = [
            (pk.int8, np.int8),
            (pk.int16, np.int16),
            (pk.int32, np.int32),
            (pk.int64, np.int64),
            (pk.uint8, np.uint8),
            (pk.uint16, np.uint16),
            (pk.uint32, np.uint32),
            (pk.uint64, np.uint64),
        ]
        for type_class, np_dtype in test_cases:
            with self.subTest(type_class=type_class):
                result = pk.ScratchView1D[type_class].shmem_size(dim)
                expected = self._calculate_expected_size(np_dtype, dim)
                self.assertEqual(result, expected)

    def test_scratch_view_2d_various_types(self):
        """Test ScratchView2D with various types"""
        dim0, dim1 = 4, 4
        test_cases = [
            (float, np.float64),
            (pk.float, np.float32),
            (pk.double, np.float64),
            (int, np.int32),
            (pk.int32, np.int32),
        ]
        for type_class, np_dtype in test_cases:
            with self.subTest(type_class=type_class):
                result = pk.ScratchView2D[type_class].shmem_size(dim0, dim1)
                expected = self._calculate_expected_size(np_dtype, dim0, dim1)
                self.assertEqual(result, expected)

    def test_scratch_view_3d_various_types(self):
        """Test ScratchView3D with various types"""
        dim0, dim1, dim2 = 3, 3, 3
        test_cases = [
            (float, np.float64),
            (pk.float, np.float32),
            (pk.int32, np.int32),
        ]
        for type_class, np_dtype in test_cases:
            with self.subTest(type_class=type_class):
                result = pk.ScratchView3D[type_class].shmem_size(dim0, dim1, dim2)
                expected = self._calculate_expected_size(np_dtype, dim0, dim1, dim2)
                self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
