import pykokkos as pk

if pk.get_default_space() in pk.DeviceExecutionSpace:
    import cupy as np
else:
    import numpy as np

import argparse


def main(N: int, M: int):
    element: int = N * M

    val = np.full(N * M, float(N + M), dtype=np.double)
    visited = np.zeros(N * M, dtype=np.int32)

    mat = np.ones((N, M), dtype=np.double)
    mat[0][1] = 0
    mat[0][3] = 0

    max_arr = np.zeros(N, dtype=np.double)
    max_arr2D = np.zeros((N, N), dtype=np.double)

    for i in range(N + M):
        pk.parallel_for(element, check_vis, N=N, M=M, mat=mat, val=val, visited=visited)

    pk.parallel_for(N, findmax, M=M, val=val, max_arr=max_arr)

    pk.parallel_for(N, extend2D, N=N, max_arr=max_arr, max_arr2D=max_arr2D)
    pk.parallel_for(N, reduce1D, N=N, max_arr=max_arr, max_arr2D=max_arr2D)

    print(f"\ndistance of every cell:\n")
    for i in range(element):
        print(f"val ({val[i]})  ", end="")
        if (i + 1) % M == 0:
            print(f"\n")
    print(f"The farthest distance is {max_arr[0]}")

    ################################
    # check_vis will operate breadth-first search
    # self.visited[i] will be 1 if self.val[i] = 0 or if self.visited[j] = 1
    # where j is one of the neighbor of i
    ################################


@pk.workunit
def check_vis(
    i: int,
    N: int,
    M: int,
    mat: pk.View2D[pk.double],
    val: pk.View1D[pk.double],
    visited: pk.View1D[int],
):
    var_row: int = i // M
    var_col: int = i % M
    min_val: float = val[i]

    flag: int = 0

    # if the value of the current index is 0, then the distance is 0,
    # and the node is marked as visited
    # otherwise, check whether the neighbors were visited,
    # if visited, the value of the current index can be decided
    if mat[var_row][var_col] == 0 and visited[i] == 0:
        visited[i] = 1
        val[i] = 0
    else:
        # check the neighbor on the previous row
        if i >= M:
            if visited[i - M] == 1:
                flag = 1
                if min_val > val[i - M]:
                    min_val = val[i - M]

            # check the neighbor on the next row
        if i // M < (N - 1):
            if visited[i + M] == 1:
                flag = 1
                if min_val > val[i + M]:
                    min_val = val[i + M]

            # check the neighbor on the left
        if i % M > 0:
            if visited[i - 1] == 1:
                flag = 1
                if min_val > val[i - 1]:
                    min_val = val[i - 1]

            # check the neighbor on the right
        if i % M < (M - 1):
            if visited[i + 1] == 1:
                flag = 1
                if min_val > val[i + 1]:
                    min_val = val[i + 1]

        # if there is at least one neighbor visited, the value of
        # the current index can be updated and should be marked as visited
    if flag == 1:
        if val[i] > min_val:
            val[i] = min_val + 1
        visited[i] = 1

    ################################
    # findmax will find the maximum value of cell in each row
    ################################


@pk.workunit
def findmax(j: int, M: int, val: pk.View1D[pk.double], max_arr: pk.View1D[pk.double]):
    tmp_max: float = 0
    for i in range(M):
        if tmp_max < val[j * M + i]:
            tmp_max = val[j * M + i]
    max_arr[j] = tmp_max

    ################################
    # extend2D and reduce1D are for finding the maximum value of all cell
    # in findmax, the maximum value will be stored in the array self.max_arr
    # extend2D will extend the 1D array self.max_arr to 2D array self.max_arr2D, where each column has the same value
    # reduce1D will reduce self.max_arr2D to the 1D array by finding the maximum value in each row, and store it to self.max_arr
    # Example:
    # self.max_arr =
    # [0, 5, 2]
    # -> self.max_arr2D =
    # [0, 5, 2]
    # [0, 5, 2]
    # [0, 5, 2]
    # -> self.max_arr =
    # [5, 5, 5]
    # then self.max_arr[0] will be the maximum distance
    ################################


@pk.workunit
def extend2D(
    j: int, N: int, max_arr: pk.View1D[pk.double], max_arr2D: pk.View2D[pk.double]
):
    for i in range(N):
        max_arr2D[i][j] = max_arr[j]


@pk.workunit
def reduce1D(
    j: int, N: int, max_arr: pk.View1D[pk.double], max_arr2D: pk.View2D[pk.double]
):
    tmp_max: float = 0
    for i in range(N):
        if tmp_max < max_arr2D[j][i]:
            tmp_max = max_arr2D[j][i]
    max_arr[j] = tmp_max


if __name__ == "__main__":
    N: int = -1
    M: int = -1

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--rows", type=int)
    parser.add_argument("-M", "--columns", type=int)

    args = parser.parse_args()

    N = 2**args.rows if args.rows else 2**3
    M = 2**args.columns if args.columns else 2**3

    print(f"Total size: {N*M}, N={N}, M={M}")

    main(N, M)
