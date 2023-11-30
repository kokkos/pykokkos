import pykokkos as pk

import argparse
import sys

@pk.workload(
    inp=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight),
    out=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight))
class main:
    def __init__(self, iterations, n, tile_size, star, radius):
        self.iterations: int = iterations
        self.n: int = n
        self.tile_size: int = tile_size
        self.star: int = star
        self.radius: int = radius

        self.inp: pk.View2D[pk.double] = pk.View([self.n, self.n], pk.double, layout=pk.Layout.LayoutRight)
        self.out: pk.View2D[pk.double] = pk.View([self.n, self.n], pk.double, layout=pk.Layout.LayoutRight)
        self.norm: float = 0

        self.stencil_time: float = 0

    @pk.main
    def run(self):
        t: int = tile_size
        r: int = radius

        pk.parallel_for(pk.MDRangePolicy([0,0], [n, n], [t, t]),
            self.init)
        pk.fence()

        timer = pk.Timer()

        for i in range(iterations):
            if (i == 1):
                pk.fence()

            if r == 1:
                # star1 stencil
                pk.parallel_for("stencil", pk.MDRangePolicy([r,r], [n-r, n-r], [t, t]), self.star1)
            elif r == 2:
                # star2 stencil
                pk.parallel_for("stencil", pk.MDRangePolicy([r,r], [n-r, n-r], [t, t]), self.star2)
            else:
                # star3 stencil
                pk.parallel_for("stencil", pk.MDRangePolicy([r,r], [n-r, n-r], [t, t]), self.star3)


            pk.parallel_for(pk.MDRangePolicy([0,0], [n, n], [t, t]),
                self.increment)

        pk.fence()
        self.stencil_time = timer.seconds()

        active_points: int = (n-2*r)*(n-2*r)

        # verify correctness
        self.norm = pk.parallel_reduce(pk.MDRangePolicy([r, r], [n-r, n-r], [t, t]),
                self.norm_reduce)
        pk.fence()
        self.norm /= active_points

        episilon: float = 1.0e-8
        reference_norm: float = 2*(iterations)
        if (abs(self.norm-reference_norm) > episilon):
            pk.printf("ERROR: L1 norm != Reference norm err=%.2f\n", abs(self.norm-reference_norm))
        else:
            pk.printf("Solution validates\n")

    @pk.workunit
    def init(self, i: int, j: int):
        self.inp[i][j] = i + j + 0.0
        self.out[i][j] = 0.0

    @pk.workunit
    def increment(self, i: int, j: int):
        self.inp[i][j] += 1.0

    @pk.workunit
    def norm_reduce(self, i: int, j: int, acc: pk.Acc[pk.double]):
        acc += abs(self.out[i][j])

    # @pk.callback
    # def print_result(self):
    #     print(self.inp)
    #     print(self.out)


    @pk.workunit
    def star1(self, i: int, j: int):
        self.out[i][j] += \
              +self.inp[i][j-1] * -0.5 \
              +self.inp[i-1][j] * -0.5 \
              +self.inp[i+1][j] * 0.5 \
              +self.inp[i][j+1] * 0.5 \

    @pk.workunit
    def star2(self, i: int, j: int):
        self.out[i][j] += +self.inp[i][j-2] * -0.125 \
              +self.inp[i][j-1] * -0.25 \
              +self.inp[i-2][j] * -0.125 \
              +self.inp[i-1][j] * -0.25 \
              +self.inp[i+1][j] * 0.25 \
              +self.inp[i+2][j] * 0.125 \
              +self.inp[i][j+1] * 0.25 \
              +self.inp[i][j+2] * 0.125

    @pk.workunit
    def star3(self, i: int, j: int):
        self.out[i][j] += \
            +self.inp[i][j-3] * -0.05555555555555555 \
            +self.inp[i][j-2] * -0.08333333333333333 \
            +self.inp[i][j-1] * -0.16666666666666666 \
            +self.inp[i-3][j] * -0.05555555555555555 \
            +self.inp[i-2][j] * -0.08333333333333333 \
            +self.inp[i-1][j] * -0.16666666666666666 \
            +self.inp[i+1][j] * 0.16666666666666666 \
            +self.inp[i+2][j] * 0.08333333333333333 \
            +self.inp[i+3][j] * 0.05555555555555555 \
            +self.inp[i][j+1] * 0.16666666666666666 \
            +self.inp[i][j+2] * 0.08333333333333333 \
            +self.inp[i][j+3] * 0.05555555555555555

def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('iterations', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('tile_size', nargs='?', type=int, default=32)
    parser.add_argument('stencil', nargs='?', type=str, default="star")
    parser.add_argument('radius', nargs='?', type=int, default=2)
    parser.add_argument("-space", "--execution_space", type=str)

    args = parser.parse_args()
    iterations = args.iterations
    n = args.n
    tile_size = args.tile_size
    stencil = args.stencil
    radius = args.radius

    assert(radius-1 in range(3))

    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    # linear grid dimension
    if n < 1:
        sys.exit("ERROR: grid dimension must be positive")
    elif n > 46340:
        sys.exit("ERROR: grid dimension too large - overflow risk")

    # default tile size for tiling of local transpose
    tile_size=32
    # if tile_size <= 0:
    #     tile_size = n
    # if tile_size > n:
    #     tile_size = n

    # stencil pattern
    star = False if (stencil == "grid") else True

    if (radius < 1) or (2*radius+1 > n):
        sys.exit("ERROR: Stencil radius negative or too large")

    if args.execution_space:
        space = pk.ExecutionSpace(args.execution_space)
        pk.set_default_space(space)

    # pk.enable_uvm()

    n = 2 ** n
    print("Number of iterations = ", iterations)
    print("Grid size            = ", n)
    print("Tile size            = ", tile_size)
    print("Type of stencil      = ", "star" if star else "grid")
    print("Radius of stencil    = ", radius)
    pk.execute(pk.ExecutionSpace.Default, main(iterations, n, tile_size, star, radius))

if __name__ == "__main__":
    run()
