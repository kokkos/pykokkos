import pykokkos as pk

import argparse
from functools import reduce
import sys

@pk.workunit
def init_arrays(index, a_view, b_view, c_view, initA, initB, initC):
    a_view[index] = initA
    b_view[index] = initB
    c_view[index] = initC

@pk.workunit
def copy(index, a_view, c_view):
    c_view[index] = a_view[index]

@pk.workunit
def mul(index, b_view, scalar, c_view):
    b_view[index] = scalar * c_view[index]

@pk.workunit
def add(index, a_view, b_view, c_view):
    c_view[index] = a_view[index] + b_view[index]

@pk.workunit
def triad(index, a_view, b_view, c_view, scalar):
    a_view[index] = b_view[index] + scalar * c_view[index]

@pk.workunit
def dot(index, acc, a_view, b_view):
    acc += a_view[index] * b_view[index]


def run() -> None:
    array_size: int = 2**25 # 100000
    startA: float = 0.1
    startB: float = 0.2
    startC: float = 0.0
    scalar: float = 0.4
    num_times = 100
    space = pk.ExecutionSpace.OpenMP

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--arraysize", type=int, help="Use SIZE elemnts in the array")
    parser.add_argument("-n", "--numtimes", type=int, help="Run the test NUM times (NUM >= 2)")
    parser.add_argument("-space", "--execution_space", type=str)
    args = parser.parse_args()

    if args.arraysize:
        array_size = args.arraysize
    if args.numtimes:
        num_times = args.numtimes
    if args.execution_space:
        space = pk.ExecutionSpace(space)

    pk.set_default_space(space)

    a: pk.View1D[pk.double] = pk.View([array_size], pk.double)
    b: pk.View1D[pk.double] = pk.View([array_size], pk.double)
    c: pk.View1D[pk.double] = pk.View([array_size], pk.double)

    p = pk.RangePolicy(0, array_size)
    pk.parallel_for(p, init_arrays, a_view=a, b_view=b, c_view=c, initA=startA, initB=startB, initC=startC)

    timer = pk.Timer()
    timings = [[] for i in range(5)]
    for i in range(num_times):
        pk.parallel_for(p, copy, a_view=a, c_view=c)
        timings[0].append(timer.seconds())
        timer.reset()

        pk.parallel_for(p, mul, b_view=b, scalar=scalar, c_view=c)
        timings[1].append(timer.seconds())
        timer.reset()

        pk.parallel_for(p, add, a_view=a, b_view=b, c_view=c)
        timings[2].append(timer.seconds())
        timer.reset()

        pk.parallel_for(p, triad, a_view=a, b_view=b, c_view=c, scalar=scalar)
        timings[3].append(timer.seconds())
        timer.reset()

        n_sum = pk.parallel_reduce(p, dot, a_view=a, b_view=b)
        timings[4].append(timer.seconds())
        timer.reset()

    goldA = startA
    goldB = startB
    goldC = startC

    for i in range(num_times):
        goldC = goldA
        goldB = scalar * goldC
        goldC = goldA + goldB
        goldA = goldB + scalar * goldC

    errA = reduce(lambda s, val: s + abs(val - goldA), a)
    errA /= len(a)
    errB = reduce(lambda s, val: s + abs(val - goldB), b)
    errB /= len(b)
    errC = reduce(lambda s, val: s + abs(val - goldC), c)
    errC /= len(c)

    # epsi = sys.float_info.epsilon * 100
    epsi = 1e-8
    if (errA > epsi):
        print(f"Validation failed on a[]. Average error {errA}")
    if (errB > epsi):
        print(f"Validation failed on b[]. Average error {errB}")
    if (errC > epsi):
        print(f"Validation failed on c[]. Average error {errC}")

    goldSum = goldA * goldB * array_size
    errSum = n_sum - goldSum
    if (abs(errSum) > 1e-8):
        print(f"Validation failed on sum. Error {errSum}")

    print("%-12s%-12s%-12s%-12s%-12s" % ("Function", "MBytes/sec", "Min (sec)", "Max", "Average"))
    double_size = 8
    sizes = [
        2 * double_size * array_size,
        2 * double_size * array_size,
        3 * double_size * array_size,
        3 * double_size * array_size,
        2 * double_size * array_size,
    ]
    labels = ["Copy", "Mul", "Add", "Triad", "Dot"]
    for i in range(5):
        t_min = min(timings[i])
        t_max = max(timings[i])
        t_average = sum(timings[i])/len(timings[i])
        print("%-12s%-12.3f%-12.5f%-12.5f%-12.5f" % (labels[i], 1e-6*sizes[i]/t_min, t_min, t_max, t_average))

    # total_bytes = 3 * sys.getsizeof(0.0) * array_size * num_times;
    # bandwidth = 1.0e-9 * (total_bytes / runtime)
    # print(f"Runtime (seconds): {runtime}")
    # print(f"Bandwidth (GB/s): {bandwidth}")

if __name__ == "__main__":
    run()
