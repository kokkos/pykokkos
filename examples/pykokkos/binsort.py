import pykokkos as pk

if pk.get_default_space() in pk.DeviceExecutionSpace:
    import cupy as np
else:
    import numpy as np


def main():
    total_threads: int = 10

    view = np.zeros(total_threads, dtype=np.int32)

    pk.parallel_for(total_threads, work, total_threads=total_threads, view=view)
    max_bins = total_threads // 2
    min_key = total_threads
    max_key = total_threads * 2 - 1
    bin_op = pk.BinOp1D(
        view,
        max_bins,
        min_key,
        max_key,
    )
    bin_sort = pk.BinSort(view, bin_op)
    bin_sort.create_permute_vector()
    permute_vector = bin_sort.get_permute_vector()
    bin_offsets = bin_sort.get_bin_offsets()
    bin_count = bin_sort.get_bin_count()
    bin_sort.sort(view)

    print(
        "PyKokkos BinSort demo: fill a 1D key view on the device, bucket keys into "
        f"{max_bins} bins over [{min_key}, {max_key}], then sort.\n"
        f"  Initial keys: view[i] = i + {total_threads} (see work unit).\n"
    )
    print(
        "Sorted keys (same 1D view, after bin_sort.sort(view) — in-place reorder):\n",
        view,
        "\n",
    )
    print(
        "Permute vector from create_permute_vector / get_permute_vector — "
        "indices describing how elements were reordered:\n",
        permute_vector,
        "\n",
    )
    print(
        "Bin offsets (get_bin_offsets) — start index of each bin in the sorted layout:\n",
        bin_offsets,
        "\n",
    )
    print(
        "Bin counts (get_bin_count) — number of keys in each bin:\n",
        bin_count,
    )


@pk.workunit
def work(i: int, total_threads: int, view: pk.View1D[pk.int32]):
    view[i] = 2 * i + total_threads - i


if __name__ == "__main__":
    main()
