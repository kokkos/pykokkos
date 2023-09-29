"""
Compare DGEMM performance with SciPy
(i.e., a wheel with OpenBLAS 0.3.18)
"""

import pykokkos as pk
from pykokkos.linalg.l3_blas import dgemm as pk_dgemm

import numpy as np
from scipy.linalg.blas import dgemm as scipy_dgemm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    import timeit
    num_repeats = 50
    results = {"PyKokkos": {},
               "SciPy": {}}
    alpha, a, b, c, beta = (3.6,
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
                            4.3)
    for system_size in ["small", "medium", "large"]:
        print("-" * 20)
        print(f"system size: {system_size}")

        if system_size == "medium":
            a_new = np.tile(a, (10, 0))
            b_new = np.tile(b, (0, 10))
            c_new = np.ones((40, 40)) * 3.3
        elif system_size == "large":
            a_new = np.tile(a, (40, 0))
            b_new = np.tile(b, (0, 40))
            c_new = np.ones((160, 160)) * 3.3
        else:
            a_new = a
            b_new = b
            c_new = c

        view_a = pk.array(a_new)
        view_b = pk.array(b_new)
        view_c = pk.array(c_new)

        pk_dgemm_time_sec = timeit.timeit("pk_dgemm(alpha, view_a, view_b, beta, view_c)",
                                          globals=globals(),
                                          number=num_repeats)
        results["PyKokkos"][system_size] = pk_dgemm_time_sec
        print(f"PyKokkos DGEMM execution time (s) for {num_repeats} repeats: {pk_dgemm_time_sec}")
        scipy_dgemm_time_sec = timeit.timeit("scipy_dgemm(alpha, a_new, b_new, beta, c_new)",
                                             globals=globals(),
                                             number=num_repeats)
        results["SciPy"][system_size] = scipy_dgemm_time_sec
        print(f"SciPy DGEMM execution time (s) for {num_repeats} repeats: {scipy_dgemm_time_sec}")
        ratio = pk_dgemm_time_sec / scipy_dgemm_time_sec
        if ratio == 1:
            print("PyKokkos DGEMM timing is identical to SciPy")
        elif ratio > 1:
            print(f"PyKokkos DGEMM timing is slower than SciPy with ratio: {ratio:.2f} fold")
        else:
            print(f"PyKokkos DGEMM timing is faster than SciPy with ratio: {ratio:.2f} fold")
        print("-" * 20)
    df = pd.DataFrame.from_dict(results)
    print("df:\n", df)
    fig, ax = plt.subplots()
    df.plot.bar(ax=ax,
                rot=0,
                logy=True,
                xlabel="Problem Size",
                ylabel=f"log of time (s) for {num_repeats} repeats",
                title="DGEMM Performance Comparison with timeit")
    fig.savefig("DGEMM_perf_compare.png", dpi=300)
