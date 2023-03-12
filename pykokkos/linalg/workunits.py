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
    width: int = 4

    # start off by getting a global thread id
    global_tid: int = team_member.league_rank() * team_member.team_size() + team_member.team_rank()

    # TODO: I have no idea how to get 2D scratch memory views?
    scratch_mem_a: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), tile_size)
    scratch_mem_b: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), tile_size)
    # in a 4 x 4 matrix with 2 x 2 tiling the leagues
    # and teams have matching row/col assignment approaches
    bx: int = team_member.league_rank() / 2
    by: int = 0
    if team_member.league_rank() % 2 != 0:
        by = 1
    tx: int = team_member.team_rank() / 2
    ty: int = 0
    if team_member.team_rank() % 2 != 0:
        ty = 1
    tmp: float = 0
    col: int = by * 2 + ty
    row: int = bx * 2 + tx

    # these variables are a bit silly--can we not get
    # 2D scratch memory indexing?
    a_index: int = 0
    b_index: int = 0

    for i in range(out.extent(1) / 2):
        scratch_mem_a[team_member.team_rank()] = view_a[row][i * 2 + ty]
        scratch_mem_b[team_member.team_rank()] = view_b[i * 2 + tx][col]
        team_member.team_barrier()

        for k in range(2):
            a_index = tx + (k * 2)
            b_index = ty + (k * 2)
            tmp += scratch_mem_a[a_index] * scratch_mem_b[b_index]
            team_member.team_barrier()

    out[row][col] = tmp
