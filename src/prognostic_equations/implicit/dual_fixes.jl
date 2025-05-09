# TODO: Move all of the following to ClimaCore.
# Method definitions that break precompilation have been commented out.

# Missing methods:

using ClimaCore: Adapt, Grids
Adapt.@adapt_structure Grids.ColumnGrid

Spaces.ncolumns(::Spaces.FiniteDifferenceSpace) = 1

# ClimaComms.device(fv::Fields.FieldVector) =
#     ClimaComms.device(first(Fields._values(fv)))
# ClimaComms.context(fv::Fields.FieldVector) =
#     ClimaComms.context(first(Fields._values(fv)))

# Bugfix for column views into Field broadcasts:

# column_style(::Type{S}) where {DS, S <: Fields.FieldStyle{DS}} =
#     Fields.FieldStyle{column_style(DS)}
# column_style(
#     ::Type{S},
# ) where {Nv, Ni, A, S <: DataLayouts.VIJFHStyle{Nv, Ni, A}} =
#     DataLayouts.VFStyle{Nv, A}
# column_style(::Type{S}) where {Ni, A, S <: DataLayouts.IJFHStyle{Ni, A}} =
#     DataLayouts.DataFStyle{A}
# Base.@propagate_inbounds function Fields.column(
#     bc::Base.Broadcast.Broadcasted{Style},
#     i,
#     j,
#     h,
# ) where {Style <: Fields.AbstractFieldStyle}
#     _args = Fields.column_args(bc.args, i, j, h)
#     _axes = Fields.column(axes(bc), i, j, h)
#     Base.Broadcast.Broadcasted{column_style(Style)}(bc.f, _args, _axes)
# end
# Base.@propagate_inbounds function Fields.column(
#     bc::DataLayouts.NonExtrudedBroadcasted{Style},
#     i,
#     j,
#     h,
# ) where {Style <: Fields.AbstractFieldStyle}
#     _args = Fields.column_args(bc.args, i, j, h)
#     _axes = Fields.column(axes(bc), i, j, h)
#     DataLayouts.NonExtrudedBroadcasted{column_style(Style)}(bc.f, _args, _axes)
# end

# Wrapping arrays in FieldVectors:

import ClimaCore.DataLayouts: parent_array_type, device_dispatch

parent_array_type(
    ::Type{<:Base.ReshapedArray{T, N, P, MI}},
) where {T, N, P, MI} = parent_array_type(P)
parent_array_type(
    ::Type{<:PermutedDimsArray{T, N, P, IP, A}},
) where {T, N, P, IP, A} = parent_array_type(A)

device_dispatch(x::PermutedDimsArray) = device_dispatch(parent(x))

# TODO: Reshape column_vectors from (Ni * Nj * Nh) × (Nv * Nf) to the transpose.
function column_vectors_to_field_vector(column_vectors, example_field_vector)
    example_fields = values(Fields._values(example_field_vector))
    example_column_fields = unrolled_map(first_column, example_fields)
    column_lengths = unrolled_map(length ∘ parent, example_column_fields)
    column_range_starts = unrolled_cumsum((1, column_lengths[1:(end - 1)]...))
    column_range_ends = column_range_starts .+ column_lengths .- 1
    column_ranges =
        unrolled_map(UnitRange, column_range_starts, column_range_ends)
    new_fields = unrolled_map(
        example_fields,
        column_ranges,
    ) do example_field, column_range
        new_data_layout = column_vectors_to_data_layout(
            view(column_vectors, :, column_range),
            Fields.field_values(example_field),
        )
        Fields.Field(new_data_layout, axes(example_field))
    end
    return Fields.FieldVector{eltype(column_vectors)}(new_fields)
end

import ClimaCore: DataLayouts
import ClimaCore.DataLayouts: replace_basetype, union_all, singleton
import ClimaCore.DataLayouts: type_params, farray_size, universal_size

# This is no longer needed, but it would still be good to fix.
# function array2data(array::AbstractArray, data::DataLayouts.AbstractData)
#     T = replace_basetype(eltype(parent(data)), eltype(array), eltype(data))
#     return union_all(singleton(data)){T, Base.tail(type_params(data))...}(
#         reshape(array, farray_size(data)...),
#     )
# end

function column_vectors_to_data_layout(array, data)
    T = replace_basetype(eltype(parent(data)), eltype(array), eltype(data))
    return union_all(singleton(data)){T, Base.tail(type_params(data))...}(
        reshaped_column_vectors(array, data),
    )
end

reshaped_column_vectors(array, data::DataLayouts.VF) =
    reshape(array, get_Nv(data), :)
reshaped_column_vectors(array, data::DataLayouts.IHF) =
    reshape(array, get_Ni(data), get_Nh(data), :)
reshaped_column_vectors(array, data::DataLayouts.IFH) =
    PermutedDimsArray(reshape(array, get_Ni(data), get_Nh(data), :), (1, 3, 2))
reshaped_column_vectors(array, data::DataLayouts.IJHF) =
    reshape(array, get_Ni(data), get_Nj(data), get_Nh(data), :)
reshaped_column_vectors(array, data::DataLayouts.IJFH) = PermutedDimsArray(
    reshape(array, get_Ni(data), get_Nj(data), get_Nh(data), :),
    (1, 2, 4, 3),
)
reshaped_column_vectors(array, data::DataLayouts.VIHF) = PermutedDimsArray(
    reshape(array, get_Ni(data), get_Nh(data), get_Nv(data), :),
    (3, 1, 2, 4),
)
reshaped_column_vectors(array, data::DataLayouts.VIFH) = PermutedDimsArray(
    reshape(array, get_Ni(data), get_Nh(data), get_Nv(data), :),
    (3, 1, 4, 2),
)
reshaped_column_vectors(array, data::DataLayouts.VIJHF) = PermutedDimsArray(
    reshape(array, get_Ni(data), get_Nj(data), get_Nh(data), get_Nv(data), :),
    (4, 1, 2, 3, 5),
)
reshaped_column_vectors(array, data::DataLayouts.VIJFH) = PermutedDimsArray(
    reshape(array, get_Ni(data), get_Nj(data), get_Nh(data), get_Nv(data), :),
    (4, 1, 2, 5, 3),
)
get_Ni(data) = universal_size(data)[1]
get_Nj(data) = universal_size(data)[2]
get_Nv(data) = universal_size(data)[4]
get_Nh(data) = universal_size(data)[5]

# Iterating over levels of scalars:

scalar_field_names(field_vector) =
    MatrixFields.filtered_names(field_vector) do x
        x isa Fields.Field && eltype(x) == eltype(field_vector)
    end

function scalar_level_iterator(field_vector)
    Iterators.flatmap(scalar_field_names(field_vector)) do name
        field = MatrixFields.get_field(field_vector, name)
        if field isa Fields.SpectralElementField
            (field,)
        else
            Iterators.map(1:Spaces.nlevels(axes(field))) do v
                Fields.level(field, v - 1 + Operators.left_idx(axes(field)))
            end
        end
    end
end

function scalar_field_index_ranges(field_vector)
    field_names = scalar_field_names(field_vector)
    last_level_indices = accumulate(field_names; init = 0) do index, name
        field = MatrixFields.get_field(field_vector, name)
        n_levels =
            field isa Fields.SpectralElementField ? 1 :
            Spaces.nlevels(axes(field))
        index + n_levels
    end
    first_level_indices = (1, (last_level_indices[1:(end - 1)] .+ 1)...)
    return map(UnitRange, first_level_indices, last_level_indices)
end

function column_iterator_indices(field)
    axes(field) isa Union{Spaces.PointSpace, Spaces.FiniteDifferenceSpace} &&
        return ((1, 1, 1),)
    horz_space = Spaces.horizontal_space(axes(field))
    qs = 1:Quadratures.degrees_of_freedom(Spaces.quadrature_style(horz_space))
    hs = Spaces.eachslabindex(horz_space)
    return horz_space isa Spaces.SpectralElementSpace1D ?
           Iterators.product(qs, hs) : Iterators.product(qs, qs, hs)
end
column_iterator_indices(field_vector::Fields.FieldVector) =
    column_iterator_indices(first(Fields._values(field_vector)))

column_iterator(iterable) =
    Iterators.map(column_iterator_indices(iterable)) do (indices...,)
        Fields.column(iterable, indices...)
    end
