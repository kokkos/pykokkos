import kokkos

import numpy as np
import pytest


@pytest.mark.parametrize(
    "type_val, expected_np_type",
    [
        (kokkos.int8, np.int8),
        (kokkos.int16, np.int16),
        (kokkos.int32, np.int32),
        (kokkos.int64, np.int64),
        (kokkos.uint8, np.uint8),
        (kokkos.uint16, np.uint16),
        (kokkos.uint32, np.uint32),
        (kokkos.uint64, np.uint64),
        (kokkos.float32, np.float32),
        (kokkos.float64, np.float64),
        (kokkos.float, np.float32),
        (kokkos.double, np.float64),
        (kokkos.short, np.int16),
        (kokkos.int, np.int32),
        (kokkos.long, np.int64),
    ],
)
def test_basic_type_equiv(type_val, expected_np_type):
    # test some view to NumPy array type equivalencies
    kokkos.initialize()
    view = kokkos.array([2], dtype=type_val, space=kokkos.DefaultHostMemorySpace)

    # NOTE: copy can still happen (attempt no copy,
    # not guarantee)
    arr = np.array(view, copy=False)
    assert arr.dtype == expected_np_type
