import pykokkos as pk
from pykokkos.linalg.l3_blas import dgemm

import numpy as np
from numpy.testing import assert_allclose
import pytest


@pytest.mark.parametrize("shape_a, shape_b", [
    ([3, 3], [3, 3]),
    ([11, 7], [7, 5]),
])
def test_dgemm_shape(shape_a, shape_b):
    view_a = pk.View(shape_a, dtype=pk.double)
    view_b = pk.View(shape_b, dtype=pk.double)
    alpha = 1.0
    actual = dgemm(alpha=alpha,
                   view_a=view_a,
                   view_b=view_b)
    # shape requirement of C
    assert actual.shape == (view_a.shape[0], view_b.shape[1])


@pytest.mark.parametrize("alpha, a, b, c, beta, expected_c", [
    (1.0,
     np.array([[1, 7, 9],
               [3, 1, 3],
               [5, 5, 22]], dtype=np.float64),
     np.array([[9, 0, 2],
               [77, 100, 4],
               [1, 500, 9]], dtype=np.float64),
     None,
     1.0,
     np.array([[557.,  5200., 111.],
               [107.,  1600., 37.],
               [452., 11500., 228.]], dtype=np.float64),
     ),
    (7.7,
     np.array([[8, 7, 1, 200, 55.3],
               [99.2, 1.11, 2.02, 17.7, 900.2],
               [5.01, 15.21, 22.07, 1.09, 22.22],
               [1, 2, 3, 4, 5]], dtype=np.float64),
     np.array([[9, 0, 2],
               [77, 100, 4],
               [1, 500, 9],
               [226.68, 11.61, 12.12],
               [17.7, 200.10, 301.17]], dtype=np.float64),
     None,
     1.0,
     np.array([[361336.437, 112323.981, 147314.0977],
               [161130.7082, 1397215.1809, 2090925.5906],
               [14466.03004, 131014.55213, 53705.17614],
               [8941.394, 21151.438, 12253.241]], dtype=np.float64),
     ),
    # this case just expands b by one column from
    # case above
    (7.7,
     np.array([[8, 7, 1, 200, 55.3],
               [99.2, 1.11, 2.02, 17.7, 900.2],
               [5.01, 15.21, 22.07, 1.09, 22.22],
               [1, 2, 3, 4, 5]], dtype=np.float64),
     np.array([[9, 0, 2, 19],
               [77, 100, 4, 19],
               [1, 500, 9, 19],
               [226.68, 11.61, 12.12, 19],
               [17.7, 200.10, 301.17, 20]], dtype=np.float64),
     None,
     1.0,
     np.array([[361336.437, 112323.981, 147314.0977, 40117.],
               [161130.7082, 1397215.1809, 2090925.5906, 156191.189],
               [14466.03004, 131014.55213, 53705.17614, 9768.374],
               [8941.394, 21151.438, 12253.241, 2233.]], dtype=np.float64),
     ),
    # this case uses the C array as input as well, so that
    # beta gets some use
    (3.6,
     np.array([[8, 7, 1, 200, 55.3],
               [99.2, 1.11, 2.02, 17.7, 900.2],
               [5.01, 15.21, 22.07, 1.09, 22.22],
               [1, 2, 3, 4, 5]], dtype=np.float64),
     np.array([[9, 0, 2, 19],
               [77, 100, 4, 19],
               [1, 500, 9, 19],
               [226.68, 11.61, 12.12, 19],
               [17.7, 200.10, 301.17, 20]], dtype=np.float64),
     np.ones((4, 4)) * 3.3,
     4.3,
     np.array([[168950.706, 52529.298, 68888.3136, 18770.19],
               [75348.0276, 653257.6512, 977589.7908, 73038.642],
               [6777.52872, 61267.74684, 25123.10352, 4581.222],
               [4194.582, 9903.174, 5742.978, 1058.19]], dtype=np.float64),
     ),
    # same thing with beta set to zero
    (3.6,
     np.array([[8, 7, 1, 200, 55.3],
               [99.2, 1.11, 2.02, 17.7, 900.2],
               [5.01, 15.21, 22.07, 1.09, 22.22],
               [1, 2, 3, 4, 5]], dtype=np.float64),
     np.array([[9, 0, 2, 19],
               [77, 100, 4, 19],
               [1, 500, 9, 19],
               [226.68, 11.61, 12.12, 19],
               [17.7, 200.10, 301.17, 20]], dtype=np.float64),
     np.ones((4, 4)) * 3.3,
     0.0,
     np.array([[168936.516, 52515.108, 68874.1236, 18756.],
               [75333.8376, 653243.4612, 977575.6008, 73024.452],
               [6763.33872, 61253.55684, 25108.91352, 4567.032],
               [4180.392, 9888.984, 5728.788, 1044.]], dtype=np.float64),
     ),
])
def test_dgemm_vs_scipy(alpha,
                        a,
                        b,
                        c,
                        beta,
                        expected_c):
    # test against expected results from
    # scipy.linalg.blas.dgemm
    view_a = pk.array(a)
    view_b = pk.array(b)
    if c is None:
        view_c = None
    else:
        view_c = pk.array(c)
    actual_c = dgemm(alpha=alpha,
                     view_a=view_a,
                     view_b=view_b,
                     view_c=view_c,
                     beta=beta)
    assert_allclose(actual_c, expected_c)


def test_dgemm_input_handling():
    alpha = 1.0
    view_a = pk.array(np.zeros((4, 3)))
    view_b = pk.array([np.array([0, 0, 0], dtype=np.int32)] * 4)
    with pytest.raises(ValueError, match="Second dimensions"):
        dgemm(alpha=alpha,
              view_a=view_a,
              view_b=view_b)
