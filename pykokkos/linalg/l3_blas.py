import pykokkos as pk
from pykokkos.linalg import workunits

# Level 3 BLAS functions

def dgemm(alpha: float,
          view_a,
          view_b,
          beta: float = 0.0,
          view_c = None):
    """
    Double precision floating point genernal matrix multiplication (GEMM).


    Parameters
    ----------
    alpha: float
    view_a : pykokkos view of type double
             Shape (m, k)
    view_b : pykokkos view of type double
             Shape (k, n)
    beta: float, optional
    view_c: pykokkos view of type double, optional

    Returns
    -------
    c : pykokkos view of type double
        Output view with shape (m, n)

    Notes
    -----
    This is currently a non-optimized implementation, representing
    algorithm 1 from Kurzak et al. (2012),
    IEEE Transactions On Parallel And Distributed Systems 23 (11).

    """
    # a has shape (m, k)
    # b has shape (k, n)
    k_a = view_a.shape[1]
    k_b = view_b.shape[0]

    if k_a != k_b:
        raise ValueError(f"Second dimensions shape of a is {k_a} "
                          "which does not match first dimension shape of "
                         f"b, {k_b}.")

    C = pk.View([view_a.shape[0], view_b.shape[1]], dtype=pk.double)

    if view_c is None:
        pk.parallel_for(view_a.shape[0],
                        workunits.dgemm_impl_no_view_c,
                        k_a=k_a,
                        alpha=alpha,
                        view_a=view_a,
                        view_b=view_b,
                        out=C)
    else:
        pk.parallel_for(view_a.shape[0],
                        workunits.dgemm_impl_view_c,
                        k_a=k_a,
                        alpha=alpha,
                        beta=beta,
                        view_a=view_a,
                        view_b=view_b,
                        view_c=view_c,
                        out=C)
    return C
