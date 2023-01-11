import pykokkos as pk


@pk.workunit
def add_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double], ):
    out[tid] = viewA[tid] + viewB[tid % viewB.extent(0)]


@pk.workunit
def add_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] + viewB[tid]


@pk.workunit
def add_impl_1d_int8(tid: int, viewA: pk.View1D[pk.int8], viewB: pk.View1D[pk.int8], out: pk.View1D[pk.int8]):
    out[tid] = viewA[tid] + viewB[tid]


@pk.workunit
def add_impl_1d_int16(tid: int, viewA: pk.View1D[pk.int16], viewB: pk.View1D[pk.int16], out: pk.View1D[pk.int16]):
    out[tid] = viewA[tid] + viewB[tid]


@pk.workunit
def add_impl_1d_int32(tid: int, viewA: pk.View1D[pk.int32], viewB: pk.View1D[pk.int32], out: pk.View1D[pk.int32]):
    out[tid] = viewA[tid] + viewB[tid]


@pk.workunit
def add_impl_1d_int64(tid: int, viewA: pk.View1D[pk.int64], viewB: pk.View1D[pk.int64], out: pk.View1D[pk.int64]):
    out[tid] = viewA[tid] + viewB[tid]


@pk.workunit
def add_impl_1d_uint8(tid: int, viewA: pk.View1D[pk.uint8], viewB: pk.View1D[pk.uint8], out: pk.View1D[pk.uint8]):
    out[tid] = viewA[tid] + viewB[tid]


@pk.workunit
def add_impl_1d_uint16(tid: int, viewA: pk.View1D[pk.uint16], viewB: pk.View1D[pk.uint16], out: pk.View1D[pk.uint16]):
    out[tid] = viewA[tid] + viewB[tid]


@pk.workunit
def add_impl_1d_uint32(tid: int, viewA: pk.View1D[pk.uint32], viewB: pk.View1D[pk.uint32], out: pk.View1D[pk.uint32]):
    out[tid] = viewA[tid] + viewB[tid]


@pk.workunit
def add_impl_1d_uint64(tid: int, viewA: pk.View1D[pk.uint64], viewB: pk.View1D[pk.uint64], out: pk.View1D[pk.uint64]):
    out[tid] = viewA[tid] + viewB[tid]


@pk.workunit
def add_impl_2d_double(tid: int, viewA: pk.View2D[pk.double], viewB: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[tid][i]


@pk.workunit
def add_impl_2d_uint8(tid: int, viewA: pk.View2D[pk.uint8], viewB: pk.View2D[pk.uint8], out: pk.View2D[pk.uint8]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[tid][i]


@pk.workunit
def add_impl_2d_float(tid: int, viewA: pk.View2D[pk.float], viewB: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[tid][i]


@pk.workunit
def add_impl_2d_int8(tid: int, viewA: pk.View2D[pk.int8], viewB: pk.View2D[pk.int8], out: pk.View2D[pk.int8]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[tid][i]


@pk.workunit
def add_impl_2d_int16(tid: int, viewA: pk.View2D[pk.int16], viewB: pk.View2D[pk.int16], out: pk.View2D[pk.int16]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[tid][i]


@pk.workunit
def add_impl_2d_int32(tid: int, viewA: pk.View2D[pk.int32], viewB: pk.View2D[pk.int32], out: pk.View2D[pk.int32]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[tid][i]


@pk.workunit
def add_impl_2d_int64(tid: int, viewA: pk.View2D[pk.int64], viewB: pk.View2D[pk.int64], out: pk.View2D[pk.int64]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[tid][i]


@pk.workunit
def add_impl_3d_float(tid: int, viewA: pk.View3D[pk.float], viewB: pk.View3D[pk.float], out: pk.View3D[pk.float]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            out[tid][i][j] = viewA[tid][i][j] + viewB[tid][i][j]


@pk.workunit
def add_impl_3d_double(tid: int, viewA: pk.View3D[pk.double], viewB: pk.View3D[pk.double], out: pk.View3D[pk.double]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            out[tid][i][j] = viewA[tid][i][j] + viewB[tid][i][j]


@pk.workunit
def add_impl_3d_uint8(tid: int, viewA: pk.View3D[pk.uint8], viewB: pk.View3D[pk.uint8], out: pk.View3D[pk.uint8]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            out[tid][i][j] = viewA[tid][i][j] + viewB[tid][i][j]


@pk.workunit
def add_impl_3d_uint16(tid: int, viewA: pk.View3D[pk.uint16], viewB: pk.View3D[pk.uint16], out: pk.View3D[pk.uint16]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            out[tid][i][j] = viewA[tid][i][j] + viewB[tid][i][j]


@pk.workunit
def add_impl_3d_uint32(tid: int, viewA: pk.View3D[pk.uint32], viewB: pk.View3D[pk.uint32], out: pk.View3D[pk.uint32]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            out[tid][i][j] = viewA[tid][i][j] + viewB[tid][i][j]


@pk.workunit
def add_impl_3d_uint64(tid: int, viewA: pk.View3D[pk.uint64], viewB: pk.View3D[pk.uint64], out: pk.View3D[pk.uint64]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            out[tid][i][j] = viewA[tid][i][j] + viewB[tid][i][j]

@pk.workunit
def add_impl_3d_int8(tid: int, viewA: pk.View3D[pk.int8], viewB: pk.View3D[pk.int8], out: pk.View3D[pk.int8]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            out[tid][i][j] = viewA[tid][i][j] + viewB[tid][i][j]


@pk.workunit
def add_impl_3d_int16(tid: int, viewA: pk.View3D[pk.int16], viewB: pk.View3D[pk.int16], out: pk.View3D[pk.int16]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            out[tid][i][j] = viewA[tid][i][j] + viewB[tid][i][j]


@pk.workunit
def add_impl_3d_int32(tid: int, viewA: pk.View3D[pk.int32], viewB: pk.View3D[pk.int32], out: pk.View3D[pk.int32]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            out[tid][i][j] = viewA[tid][i][j] + viewB[tid][i][j]


@pk.workunit
def add_impl_3d_int64(tid: int, viewA: pk.View3D[pk.int64], viewB: pk.View3D[pk.int64], out: pk.View3D[pk.int64]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            out[tid][i][j] = viewA[tid][i][j] + viewB[tid][i][j]


@pk.workunit
def add_impl_4d_float(tid: int, viewA: pk.View4D[pk.float], viewB: pk.View4D[pk.float], out: pk.View4D[pk.float]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                out[tid][i][j][k] = viewA[tid][i][j][k] + viewB[tid][i][j][k]


@pk.workunit
def add_impl_4d_double(tid: int, viewA: pk.View4D[pk.double], viewB: pk.View4D[pk.double], out: pk.View4D[pk.double]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                out[tid][i][j][k] = viewA[tid][i][j][k] + viewB[tid][i][j][k]


@pk.workunit
def add_impl_4d_uint8(tid: int, viewA: pk.View4D[pk.uint8], viewB: pk.View4D[pk.uint8], out: pk.View4D[pk.uint8]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                out[tid][i][j][k] = viewA[tid][i][j][k] + viewB[tid][i][j][k]


@pk.workunit
def add_impl_4d_uint16(tid: int, viewA: pk.View4D[pk.uint16], viewB: pk.View4D[pk.uint16], out: pk.View4D[pk.uint16]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                out[tid][i][j][k] = viewA[tid][i][j][k] + viewB[tid][i][j][k]


@pk.workunit
def add_impl_4d_uint32(tid: int, viewA: pk.View4D[pk.uint32], viewB: pk.View4D[pk.uint32], out: pk.View4D[pk.uint32]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                out[tid][i][j][k] = viewA[tid][i][j][k] + viewB[tid][i][j][k]


@pk.workunit
def add_impl_4d_uint64(tid: int, viewA: pk.View4D[pk.uint64], viewB: pk.View4D[pk.uint64], out: pk.View4D[pk.uint64]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                out[tid][i][j][k] = viewA[tid][i][j][k] + viewB[tid][i][j][k]


@pk.workunit
def add_impl_4d_int8(tid: int, viewA: pk.View4D[pk.int8], viewB: pk.View4D[pk.int8], out: pk.View4D[pk.int8]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                out[tid][i][j][k] = viewA[tid][i][j][k] + viewB[tid][i][j][k]


@pk.workunit
def add_impl_4d_int16(tid: int, viewA: pk.View4D[pk.int16], viewB: pk.View4D[pk.int16], out: pk.View4D[pk.int16]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                out[tid][i][j][k] = viewA[tid][i][j][k] + viewB[tid][i][j][k]


@pk.workunit
def add_impl_4d_int32(tid: int, viewA: pk.View4D[pk.int32], viewB: pk.View4D[pk.int32], out: pk.View4D[pk.int32]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                out[tid][i][j][k] = viewA[tid][i][j][k] + viewB[tid][i][j][k]


@pk.workunit
def add_impl_4d_int64(tid: int, viewA: pk.View4D[pk.int64], viewB: pk.View4D[pk.int64], out: pk.View4D[pk.int64]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                out[tid][i][j][k] = viewA[tid][i][j][k] + viewB[tid][i][j][k]


@pk.workunit
def add_impl_5d_float(tid: int, viewA: pk.View5D[pk.float], viewB: pk.View5D[pk.float], out: pk.View5D[pk.float]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                for l in range(viewA.extent(4)):
                    out[tid][i][j][k][l] = viewA[tid][i][j][k][l] + viewB[tid][i][j][k][l]


@pk.workunit
def add_impl_5d_double(tid: int, viewA: pk.View5D[pk.double], viewB: pk.View5D[pk.double], out: pk.View5D[pk.double]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                for l in range(viewA.extent(4)):
                    out[tid][i][j][k][l] = viewA[tid][i][j][k][l] + viewB[tid][i][j][k][l]


@pk.workunit
def add_impl_5d_uint8(tid: int, viewA: pk.View5D[pk.uint8], viewB: pk.View5D[pk.uint8], out: pk.View5D[pk.uint8]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                for l in range(viewA.extent(4)):
                    out[tid][i][j][k][l] = viewA[tid][i][j][k][l] + viewB[tid][i][j][k][l]


@pk.workunit
def add_impl_5d_uint16(tid: int, viewA: pk.View5D[pk.uint16], viewB: pk.View5D[pk.uint16], out: pk.View5D[pk.uint16]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                for l in range(viewA.extent(4)):
                    out[tid][i][j][k][l] = viewA[tid][i][j][k][l] + viewB[tid][i][j][k][l]


@pk.workunit
def add_impl_5d_uint32(tid: int, viewA: pk.View5D[pk.uint32], viewB: pk.View5D[pk.uint32], out: pk.View5D[pk.uint32]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                for l in range(viewA.extent(4)):
                    out[tid][i][j][k][l] = viewA[tid][i][j][k][l] + viewB[tid][i][j][k][l]


@pk.workunit
def add_impl_5d_uint64(tid: int, viewA: pk.View5D[pk.uint64], viewB: pk.View5D[pk.uint64], out: pk.View5D[pk.uint64]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                for l in range(viewA.extent(4)):
                    out[tid][i][j][k][l] = viewA[tid][i][j][k][l] + viewB[tid][i][j][k][l]


@pk.workunit
def add_impl_5d_int8(tid: int, viewA: pk.View5D[pk.int8], viewB: pk.View5D[pk.int8], out: pk.View5D[pk.int8]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                for l in range(viewA.extent(4)):
                    out[tid][i][j][k][l] = viewA[tid][i][j][k][l] + viewB[tid][i][j][k][l]


@pk.workunit
def add_impl_5d_int16(tid: int, viewA: pk.View5D[pk.int16], viewB: pk.View5D[pk.int16], out: pk.View5D[pk.int16]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                for l in range(viewA.extent(4)):
                    out[tid][i][j][k][l] = viewA[tid][i][j][k][l] + viewB[tid][i][j][k][l]


@pk.workunit
def add_impl_5d_int32(tid: int, viewA: pk.View5D[pk.int32], viewB: pk.View5D[pk.int32], out: pk.View5D[pk.int32]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                for l in range(viewA.extent(4)):
                    out[tid][i][j][k][l] = viewA[tid][i][j][k][l] + viewB[tid][i][j][k][l]


@pk.workunit
def add_impl_5d_int64(tid: int, viewA: pk.View5D[pk.int64], viewB: pk.View5D[pk.int64], out: pk.View5D[pk.int64]):
    for i in range(viewA.extent(1)):
        for j in range(viewA.extent(2)):
            for k in range(viewA.extent(3)):
                for l in range(viewA.extent(4)):
                    out[tid][i][j][k][l] = viewA[tid][i][j][k][l] + viewB[tid][i][j][k][l]


@pk.workunit
def add_impl_2d_uint16(tid: int, viewA: pk.View2D[pk.uint16], viewB: pk.View2D[pk.uint16], out: pk.View2D[pk.uint16]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[tid][i]


@pk.workunit
def add_impl_2d_uint32(tid: int, viewA: pk.View2D[pk.uint32], viewB: pk.View2D[pk.uint32], out: pk.View2D[pk.uint32]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[tid][i]


@pk.workunit
def add_impl_2d_uint64(tid: int, viewA: pk.View2D[pk.uint64], viewB: pk.View2D[pk.uint64], out: pk.View2D[pk.uint64]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[tid][i]


@pk.workunit
def floor_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = floor(view[tid])


@pk.workunit
def floor_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)):
        out[tid][i] = floor(view[tid][i])


@pk.workunit
def floor_impl_3d_double(tid: int, view: pk.View3D[pk.double], out: pk.View3D[pk.double]):
    for i in range(view.extent(1)):
        for j in range(view.extent(2)):
            out[tid][i][j] = floor(view[tid][i][j])

@pk.workunit
def floor_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = floor(view[tid])


@pk.workunit
def floor_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)):
        out[tid][i] = floor(view[tid][i])


@pk.workunit
def floor_impl_3d_float(tid: int, view: pk.View3D[pk.float], out: pk.View3D[pk.float]):
    for i in range(view.extent(1)):
        for j in range(view.extent(2)):
            out[tid][i][j] = floor(view[tid][i][j])

@pk.workunit
def ceil_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = ceil(view[tid])


@pk.workunit
def ceil_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)):
        out[tid][i] = ceil(view[tid][i])


@pk.workunit
def ceil_impl_3d_double(tid: int, view: pk.View3D[pk.double], out: pk.View3D[pk.double]):
    for i in range(view.extent(1)):
        for j in range(view.extent(2)):
            out[tid][i][j] = ceil(view[tid][i][j])

@pk.workunit
def ceil_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = ceil(view[tid])


@pk.workunit
def ceil_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)):
        out[tid][i] = ceil(view[tid][i])


@pk.workunit
def ceil_impl_3d_float(tid: int, view: pk.View3D[pk.float], out: pk.View3D[pk.float]):
    for i in range(view.extent(1)):
        for j in range(view.extent(2)):
            out[tid][i][j] = ceil(view[tid][i][j])

@pk.workunit
def trunc_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = trunc(view[tid])


@pk.workunit
def trunc_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)):
        out[tid][i] = trunc(view[tid][i])


@pk.workunit
def trunc_impl_3d_double(tid: int, view: pk.View3D[pk.double], out: pk.View3D[pk.double]):
    for i in range(view.extent(1)):
        for j in range(view.extent(2)):
            out[tid][i][j] = trunc(view[tid][i][j])

@pk.workunit
def trunc_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = trunc(view[tid])


@pk.workunit
def trunc_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)):
        out[tid][i] = trunc(view[tid][i])


@pk.workunit
def trunc_impl_3d_float(tid: int, view: pk.View3D[pk.float], out: pk.View3D[pk.float]):
    for i in range(view.extent(1)):
        for j in range(view.extent(2)):
            out[tid][i][j] = trunc(view[tid][i][j])

@pk.workunit
def round_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = round(view[tid])


@pk.workunit
def round_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)):
        out[tid][i] = round(view[tid][i])


@pk.workunit
def round_impl_3d_double(tid: int, view: pk.View3D[pk.double], out: pk.View3D[pk.double]):
    for i in range(view.extent(1)):
        for j in range(view.extent(2)):
            out[tid][i][j] = round(view[tid][i][j])

@pk.workunit
def round_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = round(view[tid])


@pk.workunit
def round_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)):
        out[tid][i] = round(view[tid][i])


@pk.workunit
def round_impl_3d_float(tid: int, view: pk.View3D[pk.float], out: pk.View3D[pk.float]):
    for i in range(view.extent(1)):
        for j in range(view.extent(2)):
            out[tid][i][j] = round(view[tid][i][j])

@pk.workunit
def isfinite_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.uint8]):
    out[tid] = isfinite(view[tid])


@pk.workunit
def isfinite_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isfinite(view[tid][i]) # type: ignore


@pk.workunit
def isfinite_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.uint8]):
    out[tid] = isfinite(view[tid])


@pk.workunit
def isfinite_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isfinite(view[tid][i]) # type: ignore


@pk.workunit
def isfinite_impl_1d_uint8(tid: int, view: pk.View1D[pk.uint8], out: pk.View1D[pk.uint8]):
    out[tid] = isfinite(view[tid])


@pk.workunit
def isfinite_impl_2d_uint8(tid: int, view: pk.View2D[pk.uint8], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isfinite(view[tid][i]) # type: ignore


@pk.workunit
def isfinite_impl_1d_int8(tid: int, view: pk.View1D[pk.int8], out: pk.View1D[pk.uint8]):
    out[tid] = isfinite(view[tid])


@pk.workunit
def isfinite_impl_2d_int8(tid: int, view: pk.View2D[pk.int8], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isfinite(view[tid][i]) # type: ignore


@pk.workunit
def isfinite_impl_1d_int16(tid: int, view: pk.View1D[pk.int16], out: pk.View1D[pk.uint8]):
    out[tid] = isfinite(view[tid])


@pk.workunit
def isfinite_impl_2d_int16(tid: int, view: pk.View2D[pk.int16], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isfinite(view[tid][i]) # type: ignore


@pk.workunit
def isfinite_impl_1d_uint16(tid: int, view: pk.View1D[pk.uint16], out: pk.View1D[pk.uint8]):
    out[tid] = isfinite(view[tid])


@pk.workunit
def isfinite_impl_2d_uint16(tid: int, view: pk.View2D[pk.uint16], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isfinite(view[tid][i]) # type: ignore


@pk.workunit
def isfinite_impl_1d_int32(tid: int, view: pk.View1D[pk.int32], out: pk.View1D[pk.uint8]):
    out[tid] = isfinite(view[tid])


@pk.workunit
def isfinite_impl_2d_int32(tid: int, view: pk.View2D[pk.int32], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isfinite(view[tid][i]) # type: ignore


@pk.workunit
def isfinite_impl_1d_uint32(tid: int, view: pk.View1D[pk.uint32], out: pk.View1D[pk.uint8]):
    out[tid] = isfinite(view[tid])


@pk.workunit
def isfinite_impl_2d_uint32(tid: int, view: pk.View2D[pk.uint32], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isfinite(view[tid][i]) # type: ignore


@pk.workunit
def isfinite_impl_1d_int64(tid: int, view: pk.View1D[pk.int64], out: pk.View1D[pk.uint8]):
    out[tid] = isfinite(view[tid])


@pk.workunit
def isfinite_impl_2d_int64(tid: int, view: pk.View2D[pk.int64], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isfinite(view[tid][i]) # type: ignore


@pk.workunit
def isfinite_impl_1d_uint64(tid: int, view: pk.View1D[pk.uint64], out: pk.View1D[pk.uint8]):
    out[tid] = isfinite(view[tid])


@pk.workunit
def isfinite_impl_2d_uint64(tid: int, view: pk.View2D[pk.uint64], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isfinite(view[tid][i]) # type: ignore


@pk.workunit
def equal_impl_1d_double(tid: int,
                         view1: pk.View1D[pk.double],
                         view2: pk.View1D[pk.double],
                         view2_size: int,
                         view_result: pk.View1D[pk.uint8]):
    view2_idx: int = 0
    if view2_size == 1:
        view2_idx = 0
    else:
        view2_idx = tid
    if view1[tid] == view2[view2_idx]:
        view_result[tid] = 1
    else:
        view_result[tid] = 0


@pk.workunit
def equal_impl_1d_uint16(tid: int,
                         view1: pk.View1D[pk.uint16],
                         view2: pk.View1D[pk.uint16],
                         view2_size: int,
                         view_result: pk.View1D[pk.uint8]):
    view2_idx: int = 0
    if view2_size == 1:
        view2_idx = 0
    else:
        view2_idx = tid
    if view1[tid] == view2[view2_idx]:
        view_result[tid] = 1
    else:
        view_result[tid] = 0


@pk.workunit
def equal_impl_1d_int16(tid: int,
                         view1: pk.View1D[pk.int16],
                         view2: pk.View1D[pk.int16],
                         view2_size: int,
                         view_result: pk.View1D[pk.uint8]):
    view2_idx: int = 0
    if view2_size == 1:
        view2_idx = 0
    else:
        view2_idx = tid
    if view1[tid] == view2[view2_idx]:
        view_result[tid] = 1
    else:
        view_result[tid] = 0


@pk.workunit
def equal_impl_1d_int32(tid: int,
                         view1: pk.View1D[pk.int32],
                         view2: pk.View1D[pk.int32],
                         view2_size: int,
                         view_result: pk.View1D[pk.uint8]):
    view2_idx: int = 0
    if view2_size == 1:
        view2_idx = 0
    else:
        view2_idx = tid
    if view1[tid] == view2[view2_idx]:
        view_result[tid] = 1
    else:
        view_result[tid] = 0


@pk.workunit
def equal_impl_1d_int64(tid: int,
                         view1: pk.View1D[pk.int64],
                         view2: pk.View1D[pk.int64],
                         view2_size: int,
                         view_result: pk.View1D[pk.uint8]):
    view2_idx: int = 0
    if view2_size == 1:
        view2_idx = 0
    else:
        view2_idx = tid
    if view1[tid] == view2[view2_idx]:
        view_result[tid] = 1
    else:
        view_result[tid] = 0


@pk.workunit
def isinf_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.uint8]):
    out[tid] = isinf(view[tid])


@pk.workunit
def isinf_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.uint8]):
    out[tid] = isinf(view[tid])


@pk.workunit
def isinf_impl_1d_int8(tid: int, view: pk.View1D[pk.int8], out: pk.View1D[pk.uint8]):
    out[tid] = isinf(view[tid])


@pk.workunit
def isinf_impl_1d_int64(tid: int, view: pk.View1D[pk.int64], out: pk.View1D[pk.uint8]):
    out[tid] = isinf(view[tid])


@pk.workunit
def isinf_impl_1d_int32(tid: int, view: pk.View1D[pk.int32], out: pk.View1D[pk.uint8]):
    out[tid] = isinf(view[tid])


@pk.workunit
def isinf_impl_1d_uint8(tid: int, view: pk.View1D[pk.uint8], out: pk.View1D[pk.uint8]):
    out[tid] = isinf(view[tid])


@pk.workunit
def isinf_impl_2d_uint8(tid: int, view: pk.View2D[pk.uint8], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isinf(view[tid][i]) # type: ignore


@pk.workunit
def isinf_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isinf(view[tid][i]) # type: ignore


@pk.workunit
def isinf_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isinf(view[tid][i]) # type: ignore


@pk.workunit
def isinf_impl_2d_int8(tid: int, view: pk.View2D[pk.int8], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isinf(view[tid][i]) # type: ignore


@pk.workunit
def isinf_impl_2d_int64(tid: int, view: pk.View2D[pk.int64], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isinf(view[tid][i]) # type: ignore


@pk.workunit
def isinf_impl_1d_uint16(tid: int, view: pk.View1D[pk.uint16], out: pk.View1D[pk.uint8]):
    out[tid] = isinf(view[tid])


@pk.workunit
def isinf_impl_1d_int16(tid: int, view: pk.View1D[pk.int16], out: pk.View1D[pk.uint8]):
    out[tid] = isinf(view[tid])


@pk.workunit
def isinf_impl_2d_int16(tid: int, view: pk.View2D[pk.int16], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isinf(view[tid][i]) # type: ignore


@pk.workunit
def isinf_impl_2d_int32(tid: int, view: pk.View2D[pk.int32], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isinf(view[tid][i]) # type: ignore


@pk.workunit
def isinf_impl_2d_uint16(tid: int, view: pk.View2D[pk.uint16], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isinf(view[tid][i]) # type: ignore


@pk.workunit
def isinf_impl_1d_uint32(tid: int, view: pk.View1D[pk.uint32], out: pk.View1D[pk.uint8]):
    out[tid] = isinf(view[tid])


@pk.workunit
def isinf_impl_2d_uint32(tid: int, view: pk.View2D[pk.uint32], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isinf(view[tid][i]) # type: ignore


@pk.workunit
def isinf_impl_1d_uint64(tid: int, view: pk.View1D[pk.uint64], out: pk.View1D[pk.uint8]):
    out[tid] = isinf(view[tid])


@pk.workunit
def isinf_impl_2d_uint64(tid: int, view: pk.View2D[pk.uint64], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = isinf(view[tid][i]) # type: ignore


@pk.workunit
def matmul_impl_1d_double(tid: int, acc: pk.Acc[pk.double], viewA: pk.View1D[pk.double], viewB: pk.View2D[pk.double]):
    acc += viewA[tid] * viewB[0][tid]


@pk.workunit
def matmul_impl_1d_float(tid: int, acc: pk.Acc[pk.float], viewA: pk.View1D[pk.float], viewB: pk.View2D[pk.float]):
    acc += viewA[tid] * viewB[0][tid]


@pk.workunit
def reciprocal_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = 1 / view[tid] # type: ignore


@pk.workunit
def reciprocal_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = 1 / view[tid] # type: ignore


@pk.workunit
def reciprocal_impl_2d_double(tid: int, view: pk.View2D[pk.double]):
    for i in range(view.extent(1)): # type: ignore
        view[tid][i] = 1 / view[tid][i] # type: ignore


@pk.workunit
def reciprocal_impl_2d_float(tid: int, view: pk.View2D[pk.float]):
    for i in range(view.extent(1)): # type: ignore
        view[tid][i] = 1 / view[tid][i] # type: ignore
