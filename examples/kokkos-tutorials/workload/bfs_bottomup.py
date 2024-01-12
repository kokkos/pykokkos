from typing import Tuple

import pykokkos as pk

from parse_args import parse_args

@pk.workload
class Workload:
    def __init__(self, N: int, M: int):
        self.N: int = N
        self.M: int = M

        self.val: pk.View1D[pk.double] = pk.View([N*M], pk.double)
        self.vis: pk.View1D[pk.double] = pk.View([N*M], pk.double)
        self.Matrix: pk.View2D[pk.double] = pk.View([N, M], pk.double)
        self.max_arr: pk.View1D[pk.double] = pk.View([N], pk.double)
        self.max_arr2D: pk.View2D[pk.double] = pk.View([N, N], pk.double)

        self.timer_result: float = 0
        self.element: float = N*M


        for i in range(N*M):
            self.val[i] = N+M

        for i in range(M):
            self.vis[i] = 0


        # Initialize the input matrix, can be design to be any binary matrix
        # In this example, Matrix[0][1] & Matrix[0][3] will be 0, others will be 1
        for j in range(N):
            for i in range(M):
                self.Matrix[j][i] = 1

        self.Matrix[0][1] = 0
        self.Matrix[0][3] = 0

    @pk.main
    def run(self):
        timer = pk.Timer()
        self.timer_result = timer.seconds()

        # do the bfs
        for i in range(self.N+self.M):
            pk.parallel_for(self.element, self.check_vis)

        # after bfs find maximum value in each row
        pk.parallel_for("02", self.N, self.findmax)

        # find the maximum value of all cell
        pk.parallel_for("02", self.N, self.extend2D)
        pk.parallel_for("02", self.N, self.reduce1D)


    @pk.callback
    def results(self):
        print(f"N({self.N}) M({self.M}) time({self.timer_result}) \n")
        print(f"distance of every cell")
        for i in range(self.element):
            print(f"val ({self.val[i]})  ", end="")
            if (i+1)% self.M == 0:
                print(f"\n")
        print(f"The farthest distance is {self.max_arr[0]}")


################################
# check_vis will operate breadth-first search
# self.vis[i] will be 1 if self.val[i] = 0 or if self.vis[j] = 1 
# where j is one of the neighbor of i
################################
    @pk.workunit
    def check_vis(self, i:int):
        var_row: int = i//self.M
        var_col: int = i % self.M
        min_val: float = self.val[i]

        flag: int = 0

        if (self.Matrix[var_row][var_col]==0) and (self.vis[i] == 0):
            self.vis[i] = 1
            self.val[i] = 0
        else:
            if i>=self.M:
                if self.vis[i-self.M]==1:
                    flag = 1
                    if min_val > self.val[i-self.M]:
                        min_val = self.val[i-self.M]
            if i//self.M < (self.N - 1):
                if self.vis[i+self.M] == 1:
                    flag = 1
                    if min_val > self.val[i+self.M]:
                        min_val = self.val[i+self.M]
            if i%self.M > 0:
                if self.vis[i-1] == 1:
                    flag = 1
                    if min_val > self.val[i-1]:
                        min_val = self.val[i-1]
            if i%self.M < (self.M-1):
                if self.vis[i+1] == 1:
                    flag = 1
                    if min_val > self.val[i+1]:
                        min_val = self.val[i+1]
        if flag == 1:
            if self.val[i] > min_val:
                self.val[i] = min_val + 1
            self.vis[i] = 1


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
# in findmax, the maximum value will store in array self.max_arr
# extend2D will extend the 1D array self.max_arr to 2D array self.max_arr2D, where each column has the same value
# reduce1D will reduce self.max_arr2D to 1D array by finding the maximum value in each row, and store it to self.max_arr
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
    values: Tuple[int, int, int, int, int, bool] = parse_args()
    N: int = values[0]
    M: int = values[1]

    space: str = values[-2]
    if space == "":
        space = pk.ExecutionSpace.OpenMP
    else:
        space = pk.ExecutionSpace(space)

    pk.set_default_space(space)
    pk.execute(pk.get_default_space(), Workload(N, M))
