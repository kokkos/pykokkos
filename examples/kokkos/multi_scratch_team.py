import cupy as cp
import pykokkos as pk


@pk.workunit
def init_data(i: int, view):
    view[i] = i + 1


# Two scratch regions:
#   scratch_idx - n_ints int32 elements  (passed as kwarg, variable size)
#   scratch_val - N      float64 elements (passed as kwarg)
@pk.workunit(
    scratch=[
        (int, lambda p: p.n_ints),
        (float, lambda p: p.N),
    ]
)
def scale_kernel(
    team_member: pk.TeamMember, input_view, output_view, team_size, scale, n_ints, N
):
    offset: int = team_member.league_rank() * team_size
    rank: int = team_member.team_rank()

    scratch_idx: pk.ScratchView1D[int] = pk.ScratchView1D(
        team_member.team_scratch(0), n_ints
    )
    scratch_val: pk.ScratchView1D[float] = pk.ScratchView1D(
        team_member.team_scratch(0), N
    )

    scratch_idx[rank] = input_view[offset + rank]
    team_member.team_barrier()

    scratch_val[rank] = float(scratch_idx[rank]) * scale
    team_member.team_barrier()

    output_view[offset + rank] = scratch_val[rank]


def main():
    N = 64
    team_size = 32
    n_ints = 100
    scale = 2.5
    num_teams = (N + team_size - 1) // team_size

    input_view = cp.zeros(N, dtype=cp.int32)
    output_view = cp.zeros(N, dtype=cp.float64)

    pk.parallel_for(
        pk.RangePolicy(pk.ExecutionSpace.Cuda, 0, N),
        init_data,
        view=input_view,
    )

    print(f"N={N}  team_size={team_size}  n_ints={n_ints}  scale={scale}")

    pk.parallel_for(
        "scale_kernel",
        pk.TeamPolicy(pk.ExecutionSpace.Cuda, num_teams, team_size),
        scale_kernel,
        input_view=input_view,
        output_view=output_view,
        team_size=team_size,
        scale=scale,
        n_ints=n_ints,
        N=N,
    )

    print("input :", cp.asnumpy(input_view))
    print("output:", cp.asnumpy(output_view))
    expected = cp.arange(1, N + 1, dtype=cp.float64) * scale
    assert cp.allclose(output_view, expected), "Result mismatch!"
    print("Assertion passed: output == input * scale")


if __name__ == "__main__":
    main()
