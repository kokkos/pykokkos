import pykokkos as pk


@pk.workload(subview=pk.ViewTypeInfo(trait=pk.Unmanaged))
class Workload:
    def __init__(self):
        self.view: pk.View2D[pk.int32] = pk.View([10, 10], pk.int32)
        self.subview: pk.View1D[pk.int32] = self.view[3, 2:5]

    @pk.main
    def run(self):
        pk.parallel_for(10, self.work)

    @pk.workunit
    def work(self, i: int):
        self.view[i][i] = 1

    @pk.callback
    def callback(self) -> None:
        print(self.view)

def run() -> None:
    pk.execute(pk.ExecutionSpace.Default, Workload())

if __name__ == "__main__":
    run()
