"""
Record DGEMM performance.
"""

import os
import shutil
import time
import argparse
import socket

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


def setup_data(mode):
    rng = np.random.default_rng(18898787)
    a = rng.random((square_matrix_width, square_matrix_width)).astype(float)
    b = rng.random((square_matrix_width, square_matrix_width)).astype(float)
    if "pykokkos" in mode:
        view_a = pk.View([square_matrix_width, square_matrix_width], dtype=pk.float64)
        view_b = pk.View([square_matrix_width, square_matrix_width], dtype=pk.float64)
        view_a[:] = a
        view_b[:] = b
        return view_a, view_b
    else:
        return a, b


def time_dgemm(expected, mode, league_size=4, tile_width=2):
    start = time.perf_counter()
    if mode == "pykokkos_no_tiling":
        actual = pk_dgemm(alpha, a, b, beta=0.0, view_c=None)
    elif mode == "pykokkos_with_tiling":
        actual = pk_dgemm(alpha, a, b, beta=0.0, view_c=None, league_size=4, tile_width=2)
    elif mode == "scipy":
        actual = scipy_dgemm(alpha, a, b)
    else:
        raise ValueError(f"Unknown timing mode: {mode}")
    # include check for correctness inside the
    # timer code block to prevent i.e., async GPU
    # execution; just be careful to select matrix sizes
    # large enough that the assertion isn't slower than the
    # DGEMM
    assert_allclose(actual, expected)
    end = time.perf_counter()
    dgemm_time_sec = end - start
    return dgemm_time_sec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-global-repeats', default=5)
    parser.add_argument('-m', '--mode', default="scipy")
    parser.add_argument('-p', '--power-of-two', default=10)
    parser.add_argument('-w', '--tile-width', default=2)
    parser.add_argument('-l', '--league-size', default=4)
    parser.add_argument('-s', '--space', default="OpenMP")
    args = parser.parse_args()
    hostname = socket.gethostname()

    if args.space == "OpenMP":
        space = pk.ExecutionSpace.OpenMP
    elif args.space == "Cuda":
        space = pk.ExecutionSpace.Cuda
    else:
        raise ValueError(f"Invalid execution space specified: {args.space}")
    pk.set_default_space(space)


    num_global_repeats = int(args.num_global_repeats)
    square_matrix_width = 2 ** int(args.power_of_two)


    num_threads = os.environ.get("OMP_NUM_THREADS")
    if num_threads is None:
        raise ValueError("must set OMP_NUM_THREADS for benchmarks!")

    space_name = str(space).split(".")[1]
    scenario_name = f"{hostname}_dgemm_{args.mode}_{num_threads}_OMP_threads_{space_name}_execution_space_{square_matrix_width}_square_matrix_width_{args.league_size}_league_size"

    cwd = os.getcwd()
    shutil.rmtree(os.path.join(cwd, "pk_cpp"),
                  ignore_errors=True)

    df = pd.DataFrame(np.full(shape=(num_global_repeats, 2), fill_value=np.nan),
                      columns=["scenario", "time (s)"])
    df["scenario"] = df["scenario"].astype(str)
    print("df before trials:\n", df)

    alpha = 1.0
    a, b = setup_data(mode=args.mode)
    expected = scipy_dgemm(alpha, a, b)
    counter = 0
    for global_repeat in tqdm(range(1, num_global_repeats + 1)):
        dgemm_time_sec = time_dgemm(expected, mode=args.mode, league_size=args.league_size, tile_width=args.tile_width)
        df.iloc[counter, 0] = f"{scenario_name}"
        df.iloc[counter, 1] = dgemm_time_sec
        counter += 1

    print("df after trials:\n", df)

    filename = f"{scenario_name}.parquet.gzip"
    df.to_parquet(filename,
                  engine="pyarrow",
                  compression="gzip")
