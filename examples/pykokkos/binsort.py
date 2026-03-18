import pykokkos as pk

if pk.get_default_space() in pk.DeviceExecutionSpace:
        import cupy as np
else:
        import numpy as np


def main():
    total_threads: int = 10

    view = np.zeros(total_threads, dtype=np.int32)

    pk.parallel_for(total_threads, work, total_threads=total_threads, view=view)
    bin_op = pk.BinOp1D(
        view,
        (total_threads // 2),
        total_threads,
        total_threads * 2 - 1,
    )
    bin_sort = pk.BinSort(view, bin_op)
    bin_sort.create_permute_vector()
    permute_vector = bin_sort.get_permute_vector()
    bin_offsets = bin_sort.get_bin_offsets()
    bin_count = bin_sort.get_bin_count()
    bin_sort.sort(view)

    print(view)
    print(permute_vector)
    print(bin_offsets)
    print(bin_count)


@pk.workunit
def work(i: int, total_threads: int, view: pk.View1D[pk.int32]):
    view[i] = 2 * i + total_threads - i


if __name__ == "__main__":
    main()
