import pykokkos as pk
import sys

@pk.workunit
def double_loop(tid, v):
    for i in range(3):
        x: int = 3
        pk.printf("%d\n", x)
    for j in range(3):
        y: int = 5
        pk.printf("%d\n", y)

@pk.workunit
def simple_nested(tid, v):
    for i in range(3):
        x: int = 1
        pk.printf("%d\n", x)
        for k in range(3):
            a: int = 2
            pk.printf("%d\n", a)

@pk.workunit
def nested_doubles(tid, v):
    for i in range(3):
        x: int = 1
        pk.printf("%d\n", x)
        for k in range(3):
            a: int = 2
            pk.printf("%d\n", a)

    for i in range(3):
        x: int = 3
        pk.printf("%d\n", x)
        for k in range(3):
            a: int = 4
            pk.printf("%d\n", a)

@pk.workunit
def nested_triples(tid, v): 
    pk.printf("%f\n", 0.5)
    for i in range(3):
        x: int = 1
        pk.printf("%d\n", x)
        
        for k in range(3):
            a : int = 2
            pk.printf("%d\n", a)
   
    pk.printf("%f\n", 1.5)
    for j in range(3):
        y: int = 3
        pk.printf("%d\n", y)

        for k in range(3):
            a: int = 4
            pk.printf("%d\n", a)

    pk.printf("%f\n", 2.5)
    for j in range(3):
        y: int = 5
        pk.printf("%d\n", y)

        for k in range(3):
            a : int = 6
            pk.printf("%d\n", a)

@pk.workunit
def nested_triples_noprint(tid, v): # removing prints in between loops should allow fusion?

    for i in range(3):
        x: int = 1
        pk.printf("%d\n", x)
        
        for k in range(3):
            a: int = 2
            pk.printf("%d ", a)
   

    for j in range(3):
        y: int = 3
        pk.printf("%d\n", y)

        for k in range(3):
            a : int = 4
            pk.printf("%d ", a)


    for j in range(3):
        y: int = 5
        pk.printf("%d\n", y)

        for k in range(3):
            a : int = 6
            pk.printf("%d ", a)

@pk.workunit
def view_manip_inbetween(tid, v): # I guess manually inspect c++? 
    for i in range(3):
        x: int = 1
        pk.printf("%d\n", x)
    
    v[tid] = -1
    v[tid] = v[tid] + 3

    for j in range(3):
        y: int = 2
        pk.printf("%d\n", y)

@pk.workunit
def inner_scopes(tid, v): # 1 2 3 4 4 4 5 5 5 ...

    for i in range(3):
        x: int = 1
        pk.printf("%d\n", x)

        for j in range(3):
            y: int = 2
            pk.printf("%d\n", y)

        for j in range(3):
            y: int = 3
            pk.printf("%d\n", y)

            for j in range(3):
                y: int = 4
                pk.printf("%d\n", y)

            for j in range(3):
                y: int = 5
                pk.printf("%d\n", y)

@pk.workunit
def nader_fusable(tid, v):
    for i in range(1):
        x: int = 3
        pk.printf("%d\n", x)
        for j in range(x):
            q: int = 0
            pk.printf("lol %d\n", x)
            for n in range(x, q):
                pk.printf("%d\n", q)

        for k in range(2):
            pk.printf("print 1 %d\n", k)
        up_here: int = 0

    for j in range(1):
        y: int = 4
        x: int = y
        for k in range(2):
            pk.printf("print 2 %d\n", k)

@pk.workunit
def simple_neg_dist(tid, v):
    x: int = 0
    for i in range(3):
        x += 1
    for i in range(3):
        v[i] = x

def main():

    TESTS = [double_loop, 
             simple_nested, 
             nested_doubles, 
             nested_triples, 
             nested_triples_noprint,
             view_manip_inbetween, 
             inner_scopes, 
             nader_fusable]

    N = int(sys.argv[2]) # no of iterations -> 1 is the best for simple checks
    run_test = int(sys.argv[1]) # idx of kernel to run
    my_view: pk.View1D[pk.int32] = pk.View([N], pk.int32) # toy view
    policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, N)

    pk.parallel_for(policy, TESTS[run_test], v=my_view)


if __name__ == "__main__":
    main()