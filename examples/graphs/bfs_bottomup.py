from typing import Tuple

import pykokkos as pk

import argparse

@pk.workload
class Workload:
    def __init__(self, N: int, M: int):
        self.N: int = N
        self.M: int = M

        self.val: pk.View1D[pk.double] = pk.View([N*M], pk.double)
        self.visited: pk.View1D[int] = pk.View([N*M], int)
        self.mat: pk.View2D[pk.double] = pk.View([N, M], pk.double)
        self.max_arr: pk.View1D[pk.double] = pk.View([N], pk.double)
        self.max_arr2D: pk.View2D[pk.double] = pk.View([N, N], pk.double)

        self.element: float = N*M

        self.val.fill(N+M)

        self.visited.fill(0)

        # Initialize the input matrix, can be design to be any binary matrix
        # In this example, mat[0][1] & mat[0][3] will be 0, others will be 1
        self.mat.fill(1)
        self.mat[0][1] = 0
        self.mat[0][3] = 0

    @pk.main
    def run(self):
        # do the bfs
        for i in range(self.N+self.M):
            pk.parallel_for("bfs_bottomup", self.element, self.check_vis)

        # after bfs, find maximum value in each row
        pk.parallel_for("bfs_bottomup", self.N, self.findmax)

        # find the maximum value of all cell
        pk.parallel_for("bfs_bottomup", self.N, self.extend2D)
        pk.parallel_for("bfs_bottomup", self.N, self.reduce1D)

    @pk.callback
    def results(self):  
        print(f"\ndistance of every cell:\n")
        for i in range(self.element):
            print(f"val ({self.val[i]})  ", end="")
            if (i+1)% self.M == 0:
                print(f"\n")
        print(f"The farthest distance is {self.max_arr[0]}")


################################
# check_vis will operate breadth-first search
# self.visited[i] will be 1 if self.val[i] = 0 or if self.visited[j] = 1 
# where j is one of the neighbor of i
################################
    @pk.workunit
    def check_vis(self, i:int):
        var_row: int = i//self.M
        var_col: int = i % self.M
        min_val: float = self.val[i]

        flag: int = 0

        # if the value of the current index is 0, then the distance is 0, 
        # and the node is marked as visited
        # otherwise, check whether the neighbors were visited,
        # if visited, the value of the current index can be decided
        if self.mat[var_row][var_col]==0 and self.visited[i] == 0:
            self.visited[i] = 1
            self.val[i] = 0
        else:
            # check the neighbor on the previous row
            if i>=self.M:
                if self.visited[i-self.M]==1:
                    flag = 1
                    if min_val > self.val[i-self.M]:
                        min_val = self.val[i-self.M]

            # check the neighbor on the next row
            if i//self.M < (self.N - 1):
                if self.visited[i+self.M] == 1:
                    flag = 1
                    if min_val > self.val[i+self.M]:
                        min_val = self.val[i+self.M]

            # check the neighbor on the left
            if i%self.M > 0:
                if self.visited[i-1] == 1:
                    flag = 1
                    if min_val > self.val[i-1]:
                        min_val = self.val[i-1]

            # check the neighbor on the right
            if i%self.M < (self.M-1):
                if self.visited[i+1] == 1:
                    flag = 1
                    if min_val > self.val[i+1]:
                        min_val = self.val[i+1]

        # if there is at least one neighbor visited, the value of 
        # the current index can be updated and should be marked as visited
        if flag == 1:
            if self.val[i] > min_val:
                self.val[i] = min_val + 1
            self.visited[i] = 1


################################
# findmax will find the maximum value of cell in each row
################################            
    @pk.workunit
    def findmax(self, j: int):
        tmp_max: float = 0
        for i in range(self.M):
            if tmp_max < self.val[j*self.M + i]:
                tmp_max = self.val[j*self.M + i]
        self.max_arr[j] = tmp_max


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
    def extend2D(self, j: int):
        for i in range(self.N):
            self.max_arr2D[i][j] = self.max_arr[j]

    @pk.workunit
    def reduce1D(self, j: int):
        tmp_max: float = 0
        for i in range(self.N):
            if tmp_max < self.max_arr2D[j][i]:
                tmp_max = self.max_arr2D[j][i]
        self.max_arr[j] = tmp_max


if __name__ == "__main__":
    N: int = -1
    M: int = -1
    space: str = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--rows", type=int)
    parser.add_argument("-M", "--columns", type=int)
    parser.add_argument("-space", "--execution_space", type=str)

    args = parser.parse_args()
    
    if args.rows:
        N = 2** args.rows
    else:
        N = 2**3

    if args.columns:
        M = 2**args.columns
    else:
        M = 2**3

    if args.execution_space:
        space = args.execution_space

    if space == "":
        space = pk.ExecutionSpace.OpenMP
    else:
        space = pk.ExecutionSpace(space)

    print(f"Total size: {N*M}, N={N}, M={M}")

    pk.set_default_space(space)
    pk.execute(pk.get_default_space(), Workload(N, M))
