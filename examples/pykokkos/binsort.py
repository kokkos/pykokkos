from typing import List

import pykokkos as pk


class MyView(pk.View):
    def __init__(self, x: int, data_type: pk.DataTypeClass = pk.int32):
        super().__init__([x], data_type)


@pk.workload
class Workload:
    def __init__(self, total_threads: int):
        self.total_threads: int = total_threads
        self.view: pk.View1D[pk.int32] = MyView(total_threads, data_type=pk.int32)

        self.x_0: int = 4
        self.permute_vector: pk.View1D[pk.int32] = pk.View([total_threads], pk.int32)
        self.bin_offsets: pk.View1D[pk.int32] = pk.View([6], pk.int32)
        self.bin_count: pk.View1D[pk.int32] = pk.View([6], pk.int32)

    @pk.main
    def run(self) -> None:
        x: List[int] = [self.x_0, 2, 3]
        pk.parallel_for(self.total_threads, self.work)
        bin_op = pk.BinOp1D(self.view, (self.total_threads // 2),
                            self.total_threads, self.total_threads * 2 - 1)
        bin_sort = pk.BinSort(self.view, bin_op)
        bin_sort.create_permute_vector()
        self.permute_vector = bin_sort.get_permute_vector()
        self.bin_offsets = bin_sort.get_bin_offsets()
        self.bin_count = bin_sort.get_bin_count()

        bin_sort.sort(self.view)

    @pk.workunit
    def work(self, i: int) -> None:
        self.view[i] = 2 * i + self.total_threads - i

    @pk.callback
    def results(self) -> None:
        for i in range(self.total_threads):
            print(f"{self.view[i]} ")


def run() -> None:
    workload = Workload(10)
    pk.execute(pk.ExecutionSpace.Default, workload)
    print(workload.view)
    print(workload.permute_vector)
    print(workload.bin_offsets)
    print(workload.bin_count)

if __name__ == "__main__":
    run()
