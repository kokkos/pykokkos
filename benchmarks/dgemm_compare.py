"""
Compare DGEMM performance with SciPy
(i.e., a wheel with OpenBLAS 0.3.18)
"""

import os

import pykokkos as pk
from pykokkos.linalg.l3_blas import dgemm as pk_dgemm

import numpy as np
from scipy.linalg.blas import dgemm as scipy_dgemm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    import timeit
    num_global_repeats = 50
    num_repeats = 5000
    results = {
               "PyKokkos": {"small": [],
                            "medium": [],
                            "large": []},
               "SciPy": {"small": [],
                         "medium": [],
                         "large": []},
               }
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
    num_threads = os.environ.get("OMP_NUM_THREADS")
    df = pd.DataFrame(np.full(shape=(num_global_repeats * 2, 4), fill_value=np.nan),
                      columns=["backend", "small", "medium", "large"])
    df["backend"] = df["backend"].astype(str)
    if num_threads is None:
        raise ValueError("must set OMP_NUM_THREADS for benchmarks!")

    counter = 0
    for global_repeat in tqdm(range(1, num_global_repeats + 1)):
        for col_num, system_size in tqdm(enumerate(["small", "medium", "large"]), total=3):
            if system_size == "medium":
                a_new = np.tile(a, (10, 1))
                b_new = np.tile(b, (1, 10))
                c_new = np.ones((40, 40)) * 3.3
            elif system_size == "large":
                a_new = np.tile(a, (40, 1))
                b_new = np.tile(b, (1, 40))
                c_new = np.ones((160, 160)) * 3.3
            else:
                a_new = a
                b_new = b
                c_new = c

            view_a = pk.from_numpy(a_new)
            view_b = pk.from_numpy(b_new)
            view_c = pk.from_numpy(c_new)
            pk_dgemm_time_sec = timeit.timeit("pk_dgemm(alpha, view_a, view_b, beta, view_c)",
                                              globals=globals(),
                                              number=num_repeats)
            results["PyKokkos"][system_size].append(pk_dgemm_time_sec)
            df.iloc[counter, 0] = "PyKokkos"
            df.iloc[counter, col_num + 1] = pk_dgemm_time_sec
            scipy_dgemm_time_sec = timeit.timeit("scipy_dgemm(alpha, a_new, b_new, beta, c_new)",
                                                 globals=globals(),
                                                 number=num_repeats)
            results["SciPy"][system_size].append(scipy_dgemm_time_sec)
            df.iloc[counter + 1, 0] = "SciPy"
            df.iloc[counter + 1, col_num + 1] = scipy_dgemm_time_sec
        counter += 2

    print("df:\n", df)
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(12, 5)
    df.boxplot(ax=axes,
               by="backend",
               )
    for ax in axes:
        ax.set_xlabel("Backend")
    axes[0].set_ylabel(f"Time (s) for {num_repeats} DGEMM executions")
    fig.suptitle(f"DGEMM performance boxplots (OMP_NUM_THREADS={num_threads}; {num_global_repeats} trials) for different problem sizes")
    fig.savefig(f"DGEMM_perf_compare_{num_threads}_threads.png", dpi=300)
