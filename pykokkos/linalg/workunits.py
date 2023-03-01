import pykokkos as pk


@pk.workunit
def dgemm_impl_view_c(tid: int,
                      k_a: int,
                      alpha: float,
                      beta: float,
                      view_a: pk.View2D[pk.double],
                      view_b: pk.View2D[pk.double],
                      view_c: pk.View2D[pk.double],
                      out: pk.View2D[pk.double]):
    for n in range(view_b.extent(1)):
        for k in range(k_a):
            out[tid][n] += float(view_a[tid][k] * view_b[k][n] * alpha)
        out[tid][n] += (view_c[tid][n] * beta)


@pk.workunit
def dgemm_impl_no_view_c(tid: int,
                         k_a: int,
                         alpha: float,
                         view_a: pk.View2D[pk.double],
                         view_b: pk.View2D[pk.double],
                         out: pk.View2D[pk.double]):
    for n in range(view_b.extent(1)):
        for k in range(k_a):
            out[tid][n] += float(view_a[tid][k] * view_b[k][n] * alpha)


@pk.workunit
def dgemm_impl_tiled_no_view_c(team_member: pk.TeamMember,
                               k_a: int,
                               alpha: float,
                               view_a: pk.View2D[pk.double],
                               view_b: pk.View2D[pk.double],
                               out: pk.View2D[pk.double]):
    # early attempt at tiled matrix multiplication in PyKokkos

    # for now, let's assume a 2x2 tiling arrangement and
    # that `view_a`, `view_b`, and `out` views are all 4 x 4 matrices

    # start off by getting a global thread id
    global_tid: int = team_member.league_rank() * team_member.team_size() + team_member.team_rank()
    printf("global tid: %d\n", global_tid)
    # TODO: should be a simple equation for row/column indices
    # in output, right?? not this conditional mess...
    # assume data layout is in "C" order in memory
    row: int = 0
    column: int = 0
    if team_member.league_rank() < 2 and team_member.team_rank() < 2:
        row = 0
    elif team_member.league_rank() < 2 and team_member.team_rank() >= 2:
        row = 1
    elif team_member.league_rank() >= 2 and team_member.team_rank() < 2:
        row = 2
    else:
        row = 3
    if team_member.league_rank() == 0 and team_member.team_rank() < 2:
        column = team_member.team_rank()
    elif team_member.league_rank() == 2 and team_member.team_rank() < 2:
        column = team_member.team_rank()
    elif team_member.league_rank() == 1 and team_member.team_rank() < 2:
        column = 2 + team_member.team_rank()
    elif team_member.league_rank() == 3 and team_member.team_rank() < 2:
        column = 2 + team_member.team_rank()
    elif team_member.league_rank() == 0 and team_member.team_rank() >= 2:
        column = team_member.team_rank() - 2
    elif team_member.league_rank() == 2 and team_member.team_rank() >= 2:
        column = team_member.team_rank() - 2
    elif team_member.league_rank() == 1 and team_member.team_rank() >= 2:
        column = team_member.team_rank()
    elif team_member.league_rank() == 3 and team_member.team_rank() >= 2:
        column = team_member.team_rank()
    # TODO: assign actual value here
    out[row][column] = 5
