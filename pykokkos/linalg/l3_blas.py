import pykokkos as pk

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

    for m in range(view_a.shape[0]):
        for n in range(view_b.shape[1]):
            for k in range(k_a):
                subresult = view_a[m, k] * view_b[k, n] * alpha
                C[m, n] += float(subresult) # type: ignore
            if view_c is not None:
                C[m, n] += (view_c[m, n] * beta) # type: ignore

    return C
