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
    tile_size: int = 4 # this is really just the team size...

    # start off by getting a global thread id
    global_tid: int = team_member.league_rank() * team_member.team_size() + team_member.team_rank()
    # TODO: should be a simple equation for row/column indices
    # in output, right?? not this conditional mess...
    # assume data layout is in "C" order in memory
    row: int = global_tid / 4
    column: int = team_member.team_rank()

    #printf("global_tid, row, column, and element from a: %d: (%d, %d), %f\n", global_tid, row, column, view_a[row][column])

    # start setting up the scratch (shared) memory for each team
    scratch_mem_a: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), tile_size)
    scratch_mem_b: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), tile_size)
    tmp: float = 0
    # each thread should load a single element into the local
    # shared memory from A and B, which will then be shared with other members
    # of the team
    scratch_mem_a[team_member.team_rank()] = view_a[row][column]
    scratch_mem_b[team_member.team_rank()] = view_b[row][column]
    # sync threads to ensure memory is ready for shared
    # usage in the team
    team_member.team_barrier()

    for k in range(0, 2):
        tmp += scratch_mem_a[0] * scratch_mem_b[0]
        tmp += scratch_mem_a[1] * scratch_mem_b[2]

    # TODO: assign actual value here
    out[row][column] = tmp
