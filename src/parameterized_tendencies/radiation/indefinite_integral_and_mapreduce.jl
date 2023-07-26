# TDOO: Move this entire file to ClimaCore.

import ClimaCore.RecursiveApply: rzero, radd, rmul
import CUDA: @cuda

"""
    column_indefinite_integral!(ᶠintegral::Field, ᶜintegrand::Field)

Set `ᶠintegral(z) = ∫_0^z ᶜintegrand(z) dz`, where `ᶠintegral` is on cell
faces and `ᶜintegrand` is on cell centers.
"""
column_indefinite_integral!(ᶠintegral::Fields.Field, ᶜintegrand::Fields.Field) =
    column_indefinite_integral!(
        ClimaComms.device(ᶠintegral),
        ᶠintegral,
        ᶜintegrand,
    )

function column_indefinite_integral!(
    ::ClimaComms.CUDADevice,
    ᶠintegral::Fields.Field,
    ᶜintegrand::Fields.Field,
)
    Ni, Nj, _, _, Nh = size(Fields.field_values(ᶠintegral))
    nthreads, nblocks = Spaces._configure_threadblock(Ni * Nj * Nh)
    @cuda threads = nthreads blocks = nblocks column_indefinite_integral_kernel!(
        ᶠintegral,
        ᶜintegrand,
    )
end

function column_indefinite_integral_kernel!(
    ᶠintegral::Fields.FaceExtrudedFiniteDifferenceField,
    ᶜintegrand::Fields.CenterExtrudedFiniteDifferenceField,
)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(ᶜintegrand))
    if idx <= Ni * Nj * Nh
        i, j, h = Spaces._get_idx((Ni, Nj, Nh), idx)
        ᶠintegral_column = Spaces.column(ᶠintegral, i, j, h)
        ᶜintegrand_column = Spaces.column(ᶜintegrand, i, j, h)
        _column_indefinite_integral!(ᶠintegral_column, ᶜintegrand_column)
    end
    return nothing
end

column_indefinite_integral_kernel!(
    ᶠintegral::Fields.FaceFiniteDifferenceField,
    ᶜintegrand::Fields.CenterFiniteDifferenceField,
) = _column_indefinite_integral!(ᶠintegral, ᶜintegrand)

function column_indefinite_integral!(
    ::ClimaComms.AbstractCPUDevice,
    ᶠintegral::Fields.FaceExtrudedFiniteDifferenceField,
    ᶜintegrand::Fields.CenterExtrudedFiniteDifferenceField,
)
    Fields.bycolumn(axes(ᶜintegrand)) do colidx
        _column_indefinite_integral!(ᶠintegral[colidx], ᶜintegrand[colidx])
        nothing
    end
    return nothing
end

column_indefinite_integral!(
    ::ClimaComms.AbstractCPUDevice,
    ᶠintegral::Fields.FaceFiniteDifferenceField,
    ᶜintegrand::Fields.CenterFiniteDifferenceField,
) = _column_indefinite_integral!(ᶠintegral, ᶜintegrand)

function _column_indefinite_integral!(
    ᶠintegral::Fields.ColumnField,
    ᶜintegrand::Fields.ColumnField,
)
    face_space = axes(ᶠintegral)
    first_level = Operators.left_idx(face_space)
    last_level = Operators.right_idx(face_space)
    ᶜΔz = Fields.Δz_field(ᶜintegrand)
    @inbounds Fields.level(ᶠintegral, first_level)[] = rzero(eltype(ᶜintegrand))
    for level in (first_level + 1):last_level
        @inbounds Fields.level(ᶠintegral, level)[] = radd(
            Fields.level(ᶠintegral, level - 1)[],
            rmul(
                Fields.level(ᶜintegrand, level - half)[],
                Fields.level(ᶜΔz, level - half)[],
            ),
        )
    end
    return nothing
end

"""
    column_mapreduce!(fn, op, reduced_field::Field, fields::Field...)

Applies mapreduce along the vertical direction.
"""
column_mapreduce!(
    fn::F,
    op::O,
    reduced_field::Fields.Field,
    fields::Fields.Field...,
) where {F, O} = column_mapreduce!(
    ClimaComms.device(reduced_field),
    fn,
    op,
    reduced_field,
    fields...,
)

function column_mapreduce!(
    ::ClimaComms.CUDADevice,
    fn::F,
    op::O,
    reduced_field::Fields.Field,
    fields::Fields.Field...,
) where {F, O}
    Ni, Nj, _, _, Nh = size(Fields.field_values(reduced_field))
    nthreads, nblocks = Spaces._configure_threadblock(Ni * Nj * Nh)
    @cuda threads = nthreads blocks = nblocks column_mapreduce_kernel!(
        fn,
        op,
        reduced_field,
        fields...,
    )
end

function column_mapreduce_kernel!(
    fn::F,
    op::O,
    reduced_field::Fields.SpectralElementField,
    fields::Fields.ExtrudedFiniteDifferenceField...,
) where {F, O}
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    Ni, Nj, _, _, Nh = size(Fields.field_values(field))
    if idx <= Ni * Nj * Nh
        i, j, h = Spaces._get_idx((Ni, Nj, Nh), idx)
        column_reduced_field = Spaces.column(reduced_field, i, j, h)
        column_fields = map(field -> Spaces.column(field, i, j, h), fields)
        _column_mapreduce!(fn, op, column_reduced_field, column_fields...)
    end
    return nothing
end

column_mapreduce_kernel!(
    fn::F,
    op::O,
    reduced_field::Fields.PointField,
    fields::Fields.FiniteDifferenceField...,
) where {F, O} = _column_mapreduce!(fn, op, reduced_field, fields...)

function column_mapreduce!(
    ::ClimaComms.AbstractCPUDevice,
    fn::F,
    op::O,
    reduced_field::Fields.SpectralElementField,
    fields::Fields.ExtrudedFiniteDifferenceField...,
) where {F, O}
    Fields.bycolumn(axes(reduced_field)) do colidx
        column_fields = map(field -> field[colidx], fields)
        _column_mapreduce!(fn, op, reduced_field[colidx], column_fields...)
        nothing
    end
    return nothing
end

column_mapreduce!(
    ::ClimaComms.AbstractCPUDevice,
    fn::F,
    op::O,
    reduced_field::Fields.PointField,
    fields::Fields.FiniteDifferenceField...,
) where {F, O} = _column_mapreduce!(fn, op, reduced_field, fields...)

function _column_mapreduce!(
    fn::F,
    op::O,
    reduced_field::Fields.PointField,
    fields::Fields.ColumnField...,
) where {F, O}
    space = axes(fields[1])
    all(field -> axes(field) === space, fields[2:end]) ||
        error("All input fields for column_mapreduce must be on the same space")
    get_value(level) = field -> @inbounds Fields.level(field, level)[]
    first_level = Operators.left_idx(space)
    last_level = Operators.right_idx(space)
    first_level_values = map(get_value(first_level), fields)
    @inbounds reduced_field[] = fn(first_level_values...)
    for level in (first_level + 1):last_level
        level_values = map(get_value(level), fields)
        @inbounds reduced_field[] = op(reduced_field[], fn(level_values...))
    end
    return nothing
end
