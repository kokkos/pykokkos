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
    row: int = 0
    column: int = 0
    counter: int = 0
    for league_rank in range(4):
        for base_row in range(tile_size / 2):
            for base_column in range(tile_size / 2):
                if global_tid == counter:
                    if league_rank % 2 != 0:
                        column = base_column + 2
                    else:
                        column = base_column
                    if league_rank < 2:
                        row = base_row
                    else:
                        row = base_row + 2
                counter += 1

    #printf("global_tid, row, column, and element from a: %d: (%d, %d), %f\n", global_tid, row, column, view_a[row][column])

    # start setting up the scratch (shared) memory for each team
    scratch_mem_a: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), tile_size)
    scratch_mem_b: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), tile_size)
    tmp: float = 0
    # each thread should load a single element into the local
    # shared memory from A and B, which will then be shared with other members
    # of the team
    if team_member.league_rank() == 0 or team_member.league_rank() == 3:
        scratch_mem_a[team_member.team_rank()] = view_a[row][column]
        scratch_mem_b[team_member.team_rank()] = view_b[row][column]
    elif team_member.league_rank() == 1:
        scratch_mem_a[team_member.team_rank()] = view_a[row][column - 2]
        scratch_mem_b[team_member.team_rank()] = view_b[row][column]
    elif team_member.league_rank() == 2:
        scratch_mem_a[team_member.team_rank()] = view_a[row][column]
        scratch_mem_b[team_member.team_rank()] = view_b[row - 2][column]
    # sync threads to ensure memory is ready for shared
    # usage in the team
    team_member.team_barrier()
    # the first multiplication in the dot product
    # is just the intersection of the row and column vectors
    # in a and b:
    if global_tid == 8:
        printf("tmp checkpoint 1: %f\n", tmp)
        printf("value to add to tmp: %f\n", scratch_mem_a[team_member.team_rank()] * scratch_mem_b[team_member.team_rank()])
    tmp += scratch_mem_a[team_member.team_rank()] * scratch_mem_b[team_member.team_rank()]
    if global_tid == 8:
        printf("tmp checkpoint 2: %f\n", tmp)
    # the second multiplication in the dot product
    # should include the adjacent tile members
    new_index_a: int = 0
    new_index_b: int = 0
    if team_member.team_rank() == 0:
        new_index_a = 1
        new_index_b = 2
    elif team_member.team_rank() == 1:
        new_index_a = 0
        new_index_b = 3
    elif team_member.team_rank() == 2:
        new_index_a = 3
        new_index_b = 0
    elif team_member.team_rank() == 3:
        new_index_a = 2
        new_index_b = 1
    #if team_member.league_rank() == 3:
        #for i in range(4):
            #printf("global tid %d, scratch b element %d: %f\n", global_tid, i, scratch_mem_b[i])

    #printf("global_tid: next A element, next B element in tile: %d: (%f, %f)\n", global_tid, scratch_mem_a[new_index_a], scratch_mem_b[new_index_b])
    tmp += scratch_mem_a[new_index_a] * scratch_mem_b[new_index_b]
    if global_tid == 8:
        printf("new a value to add in: %f\n", scratch_mem_a[new_index_a])
        printf("new b value to add in: %f\n", scratch_mem_b[new_index_b])
        printf("value to add to tmp: %f\n", scratch_mem_a[new_index_a] * scratch_mem_b[new_index_b])
        printf("tmp checkpoint 3: %f\n", tmp)
    team_member.team_barrier()
    # next, we need to load two more tiles from A and B to complete
    # the row/column dot product
    row_A: int = 0
    row_B: int = 0
    column_A: int = 0
    column_B: int = 0
    if team_member.league_rank() == 0:
        # for the new A (row-wise) tile:
        # the row number shouldn't change;
        # the columns will iterate
        row_A = row
        column_A = column + 2 
        # the reverse for the B tile
        row_B = row + 2
        column_B = column
    elif team_member.league_rank() == 1:
        row_A = row
        column_A = column - 2
        row_B = row + 2
        column_B = column
    elif team_member.league_rank() == 2:
        row_A = row
        column_A = column + 2
        row_B = row
        column_B = column
    elif team_member.league_rank() == 3:
        row_A = row
        column_A = column - 2
        row_B = row - 2
        column_B = column

    # TODO: it should be possible to avoid this verbosity
    # by looping...
    scratch_mem_a[team_member.team_rank()] = view_a[row_A][column_A]
    scratch_mem_b[team_member.team_rank()] = view_b[row_B][column_B]
    team_member.team_barrier()
    tmp += scratch_mem_a[team_member.team_rank()] * scratch_mem_b[team_member.team_rank()]
    if global_tid == 8:
        printf("row_A: %d\n", row_A)
        printf("column_A: %d\n", column_A)
        printf("row_B: %d\n", row_B)
        printf("column_B: %d\n", column_B)
        printf("new a value to add in: %f\n", scratch_mem_a[team_member.team_rank()])
        printf("new b value to add in: %f\n", scratch_mem_b[team_member.team_rank()])
        printf("value to add to tmp: %f\n", scratch_mem_a[team_member.team_rank()] * scratch_mem_b[team_member.team_rank()])
        printf("tmp checkpoint 4: %f\n", tmp)
    tmp += scratch_mem_a[new_index_a] * scratch_mem_b[new_index_b]
    if global_tid == 8:
        printf("value to add to tmp: %f\n", scratch_mem_a[new_index_a] * scratch_mem_b[new_index_b])
        printf("tmp checkpoint 5: %f\n", tmp)
    team_member.team_barrier()

    # TODO: assign actual value here
    out[row][column] = tmp
