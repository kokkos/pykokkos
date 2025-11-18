import pykokkos as pk


@pk.workunit
def kernel(team, A, B, C, N1, N2):
    league_rank: int = team.league_rank()

    def inner_for(i0: int, i1: int):
        A[league_rank][i0][i1] = B[league_rank][i0] + C[i1]

    pk.parallel_for(pk.TeamThreadMDRange(team, N1, N2), inner_for)
    team.team_barrier()

def run():
    N0 = 16
    N1 = 4
    N2 = 4

    A = pk.View((N0, N1, N2))
    B = pk.View((N0, N1))
    C = pk.View((N2,))

    B.fill(1)
    C.fill(1)

    print(N0)
    print(N1 * N2)

    policy = pk.TeamPolicy(N0, N1 * N2)
    pk.parallel_for(policy, kernel, A=A, B=B, C=C, N1=N1, N2=N2)

    print(A)

if __name__ == "__main__":
    run()