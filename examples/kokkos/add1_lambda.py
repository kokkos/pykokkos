import pykokkos as pk

@pk.workunit
def add_y(i: int, a: pk.View1D[pk.int32], y: int):
    a[i] += y

def main():
    n: int = 100 * 1000
    N: int = n
    pk.set_default_space(pk.ExecutionSpace.OpenMP)
    
    a: pk.View1D[pk.int32] = pk.View([N], pk.int32)
    
    # Initialize the view
    for i in range(N):
        a[i] = 2
    print(f"Initialized view: [{a[0]}, ... repeats {n-1} times]")
    
    y: int = 1
    pk.parallel_for(N, add_y, a=a, y=y)
    
    print(f"Results: [{a[0]}, ... repeats {n-1} times]")

if __name__ == "__main__":
    main()