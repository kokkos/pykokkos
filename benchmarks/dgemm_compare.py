"""
Record DGEMM performance.
"""

import os
import shutil
import time

import pykokkos as pk
from pykokkos.linalg.l3_blas import dgemm as pk_dgemm

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg.blas import dgemm as scipy_dgemm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    scenario_name = "pk_gp160_dgemm_NO_tiling_CPU_OpenMP"
    space = pk.ExecutionSpace.OpenMP
    pk.set_default_space(space)

    num_global_repeats = 5
    square_matrix_width = 2 ** 9

    rng = np.random.default_rng(18898787)
    alpha = 1.0
    a = rng.random((square_matrix_width, square_matrix_width)).astype(float)
    b = rng.random((square_matrix_width, square_matrix_width)).astype(float)
    view_a = pk.from_numpy(a)
    view_b = pk.from_numpy(b)
    #cuda_a = cp.array(a)
    #cuda_b = cp.array(b)

    num_threads = os.environ.get("OMP_NUM_THREADS")
    if num_threads is None:
        raise ValueError("must set OMP_NUM_THREADS for benchmarks!")

    cwd = os.getcwd()
    shutil.rmtree(os.path.join(cwd, "pk_cpp"),
                  ignore_errors=True)

    df = pd.DataFrame(np.full(shape=(num_global_repeats, 2), fill_value=np.nan),
                      columns=["scenario", "time (s)"])
    df["scenario"] = df["scenario"].astype(str)
    print("df before trials:\n", df)

    expected = scipy_dgemm(alpha, a, b)
    counter = 0
    for global_repeat in tqdm(range(1, num_global_repeats + 1)):
        start = time.perf_counter()
        #actual = pk_dgemm(alpha, view_a, view_b, beta=0.0, view_c=None)
        #actual = pk_dgemm(alpha, view_a, view_b, beta=0.0, view_c=None, league_size=4, tile_width=2)
        actual = scipy_dgemm(alpha, a, b)
        end = time.perf_counter()
        assert_allclose(actual, expected)
		
        dgemm_time_sec = end - start
        df.iloc[counter, 0] = f"{scenario_name}"
        df.iloc[counter, 1] = dgemm_time_sec
        counter += 1


    print("df after trials:\n", df)

    filename = f"{scenario_name}_square_matrix_width_{square_matrix_width}_{num_global_repeats}_trials.parquet.gzip"
    df.to_parquet(filename,
                  engine="pyarrow",
                  compression="gzip")
