import numpy as np
import pykokkos as pk


@pk.workunit
def sum_int(i: int, acc: pk.Acc[pk.int64], data: pk.View1D[pk.int64]) -> None:
    acc += data[i]


@pk.workunit
def prod_int(i: int, acc: pk.Acc[pk.int64], data: pk.View1D[pk.int64]) -> None:
    acc *= data[i]


@pk.workunit
def min_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]) -> None:
    if data[i] < acc:
        acc = data[i]


@pk.workunit
def max_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]) -> None:
    if data[i] > acc:
        acc = data[i]


@pk.workunit
def band_int(i: int, acc: pk.Acc[pk.int64], data: pk.View1D[pk.int64]) -> None:
    acc &= data[i]


@pk.workunit
def bor_int(i: int, acc: pk.Acc[pk.int64], data: pk.View1D[pk.int64]) -> None:
    acc |= data[i]


@pk.workunit
def land_bool(i: int, acc: pk.Acc[pk.bool], data: pk.View1D[pk.uint8]) -> None:
    acc = acc and data[i]


@pk.workunit
def lor_bool(i: int, acc: pk.Acc[pk.bool], data: pk.View1D[pk.uint8]) -> None:
    acc = acc or data[i]


@pk.workunit
def maxloc_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]) -> None:
    if data[i] > acc.val:
        acc.val = data[i]
        acc.loc = i


@pk.workunit
def minloc_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]) -> None:
    if data[i] < acc.val:
        acc.val = data[i]
        acc.loc = i


@pk.workunit
def minmax_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]) -> None:
    if data[i] < acc.min_val:
        acc.min_val = data[i]
    if data[i] > acc.max_val:
        acc.max_val = data[i]


@pk.workunit
def minmaxloc_float(i: int, acc: pk.Acc[pk.double], data: pk.View1D[pk.double]) -> None:
    if data[i] < acc.min_val:
        acc.min_val = data[i]
        acc.min_loc = i
    if data[i] > acc.max_val:
        acc.max_val = data[i]
        acc.max_loc = i


def run_reducer(name: str, reducer: pk.Reducer, workunit, data: np.ndarray):
    view = pk.array(data, space=pk.HostSpace)
    policy = pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, data.size)
    result = pk.parallel_reduce(policy, workunit, reducer=reducer, data=view)

    print(f"{name}: {result}")
    return result


def run() -> None:
    scalar_reducers = [
        ("Sum", pk.Sum, sum_int, np.array([1, 2, 3, 4], dtype=np.int64)),
        ("Prod", pk.Prod, prod_int, np.array([1, 2, 3, 4], dtype=np.int64)),
        ("Min", pk.Min, min_float, np.array([4.0, 2.0, 9.0], dtype=np.float64)),
        ("Max", pk.Max, max_float, np.array([-5.0, -1.0, -3.0], dtype=np.float64)),
        (
            "BAnd",
            pk.BAnd,
            band_int,
            np.array([0b1111, 0b1101, 0b0111], dtype=np.int64),
        ),
        (
            "BOr",
            pk.BOr,
            bor_int,
            np.array([0b1000, 0b0101, 0b0010], dtype=np.int64),
        ),
        ("LAnd", pk.LAnd, land_bool, np.array([1, 1, 0], dtype=np.uint8)),
        ("LOr", pk.LOr, lor_bool, np.array([0, 0, 1], dtype=np.uint8)),
    ]

    value_loc_reducers = [
        (
            "MaxLoc",
            pk.MaxLoc,
            maxloc_float,
            np.array([1.0, 9.0, 3.0, 7.0], dtype=np.float64),
        ),
        (
            "MinLoc",
            pk.MinLoc,
            minloc_float,
            np.array([4.0, -2.0, 8.0, -1.0], dtype=np.float64),
        ),
    ]

    minmax_reducers = [
        (
            "MinMax",
            pk.MinMax,
            minmax_float,
            np.array([4.0, -2.0, 8.0, -1.0], dtype=np.float64),
        ),
    ]

    minmax_loc_reducers = [
        (
            "MinMaxLoc",
            pk.MinMaxLoc,
            minmaxloc_float,
            np.array([4.0, -2.0, 8.0, -1.0], dtype=np.float64),
        ),
    ]

    for reducer_group in (
        scalar_reducers,
        value_loc_reducers,
        minmax_reducers,
        minmax_loc_reducers,
    ):
        for name, reducer, workunit, data in reducer_group:
            run_reducer(name, reducer, workunit, data)


if __name__ == "__main__":
    run()
