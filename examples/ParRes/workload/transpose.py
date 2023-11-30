import pykokkos as pk

import argparse
import sys

@pk.workload(
    A=pk.ViewTypeInfo(layout=pk.LayoutRight),
    B=pk.ViewTypeInfo(layout=pk.LayoutRight))
class main:
    def __init__(self, iterations, order, tile_size, permute):
        self.iterations: int = iterations
        self.order: int = order
        self.tile_size: int = tile_size
        self.permute: int = permute

        self.A: pk.View2D[pk.double] = pk.View([self.order, self.order], pk.double, layout=pk.LayoutRight)
        self.B: pk.View2D[pk.double] = pk.View([self.order, self.order], pk.double, layout=pk.LayoutRight)

        self.abserr: float = 0
        self.transpose_time: float = 0
        self.addit: float = (self.iterations) * (0.5 * (self.iterations - 1))

    @pk.main
    def run(self):
        pk.parallel_for(
            pk.MDRangePolicy([0,0], [self.order, self.order], [self.tile_size, self.tile_size]), self.init)
        pk.fence()

        timer = pk.Timer()

        for i in range(self.iterations):
            if self.permute:
                pk.parallel_for("transpose", pk.MDRangePolicy([0,0], [self.order, self.order], [self.tile_size, self.tile_size],
                    rank=pk.Rank(2, pk.Iterate.Left, pk.Iterate.Right)), self.tranpose)
            else:
                pk.parallel_for("transpose", pk.MDRangePolicy([0,0], [self.order, self.order], [self.tile_size, self.tile_size],
                    rank=pk.Rank(2, pk.Iterate.Right, pk.Iterate.Left)), self.tranpose)

        self.transpose_time = timer.seconds()

        self.abserr = pk.parallel_reduce(
            pk.MDRangePolicy([0,0], [self.order, self.order], [self.tile_size, self.tile_size]),
            self.abserr_reduce)

        pk.printf("%f\n", self.abserr)
        episilon: float = 1.0e-8
        if (self.abserr > episilon):
            pk.printf("ERROR: aggregated squared error exceeds threshold %.2f\n", self.abserr)
        else:
            pk.printf("Solution validates %2.f\n", self.abserr)

    # @pk.callback
    # def print_result(self):
    #     print(self.A)
    #     print(self.B)

    @pk.workunit
    def init(self, i: int, j: int):
        self.A[i][j] = i*self.order+j
        self.B[i][j] = 0

    @pk.workunit
    def abserr_reduce(self, i: int, j: int, acc: pk.Acc[pk.double]):
        ij: int = i * self.order + j
        reference: float = ij * (self.iterations) + self.addit
        acc += abs(self.B[j][i] - reference)

    @pk.workunit
    def tranpose(self, i: int, j: int):
        self.B[i][j] += self.A[j][i]
        self.A[j][i] += 1


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('iterations', type=int)
    parser.add_argument('order', type=int)
    parser.add_argument('tile_size', nargs='?', type=int, default=32)
    parser.add_argument('permute', nargs='?', type=int, default=0)
    parser.add_argument("-space", "--execution_space", type=str)

    args = parser.parse_args()
    iterations = args.iterations
    order= args.order
    tile_size = args.tile_size
    permute = args.permute

    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    if order <= 0:
        sys.exit("ERROR: Matrix Order must be greater than 0")
    elif order > 46340:
        sys.exit("ERROR: matrix dimension too large - overflow risk")

    # a negative tile size means no tiling of the local transpose
    if (tile_size <= 0):
        tile_size = order

    if permute != 0 and permute != 1:
        sys.exit("ERROR: permute must be 0 (no) or 1 (yes)")

    if args.execution_space:
        space = pk.ExecutionSpace(args.execution_space)
        pk.set_default_space(space)

    # pk.enable_uvm()

    order = 2 ** order
    print("Number of iterations = " , iterations)
    print("Matrix order         = " , order)
    print("Tile size            = " , tile_size)
    print("Permute loops        = " , "yes" if permute else "no")
    pk.execute(pk.ExecutionSpace.Default, main(iterations, order, tile_size, permute))

if __name__ == "__main__":
    run()
