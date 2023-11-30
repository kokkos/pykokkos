import pykokkos as pk

import cupy as cp

@pk.workunit
def print_stream(i, x, id):
    if x == 0:
        pk.printf("Default stream GPU %d\n", id)
    elif x == 1:
        pk.printf("Stream 1 GPU %d\n", id)
    elif x == 2:
        pk.printf("Stream 2 GPU %d\n", id)
    elif x == 3:
        pk.printf("Default stream GPU %d\n", id)
    elif x == 4:
        pk.printf("Stream 3 GPU %d\n", id)

def run() -> None:
    space = pk.Cuda

    # Create streams on GPU 0 (default GPU)
    s1 = cp.cuda.Stream()
    s2 = cp.cuda.Stream()

    instance1 = pk.ExecutionSpaceInstance(space, s1)
    instance2 = pk.ExecutionSpaceInstance(space, s2)

    for i in range(3):
        print(f"Iteration: {i}")
        pk.parallel_for(pk.RangePolicy(space, 0, 2), print_stream, x=0, id=cp.cuda.runtime.getDevice())
        pk.parallel_for(pk.RangePolicy(instance1, 0, 2), print_stream, x=1, id=cp.cuda.runtime.getDevice())
        pk.parallel_for(pk.RangePolicy(instance2, 0, 2), print_stream, x=2, id=cp.cuda.runtime.getDevice())

    if cp.cuda.runtime.getDeviceCount() > 1:
        # Create a stream on GPU 1
        with cp.cuda.Device(1):
            s3 = cp.cuda.Stream()

        pk.set_device_id(1)
        instance3 = pk.ExecutionSpaceInstance(space, s3)

        pk.parallel_for(pk.RangePolicy(space, 0, 2), print_stream, x=3, id=cp.cuda.runtime.getDevice())
        pk.parallel_for(pk.RangePolicy(instance3, 0, 2), print_stream, x=4, id=cp.cuda.runtime.getDevice())

        pk.set_device_id(0)
        pk.parallel_for(pk.RangePolicy(space, 0, 2), print_stream, x=0, id=cp.cuda.runtime.getDevice())
        pk.parallel_for(pk.RangePolicy(instance1, 0, 2), print_stream, x=1, id=cp.cuda.runtime.getDevice())
        pk.parallel_for(pk.RangePolicy(instance2, 0, 2), print_stream, x=2, id=cp.cuda.runtime.getDevice())

    print("Done launching kernels")


if __name__ == "__main__":
    run()
