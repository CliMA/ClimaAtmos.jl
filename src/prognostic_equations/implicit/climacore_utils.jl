# Fixing a bug in ClimaComms.context for FieldVectors:

# TODO: This must be moved into ClimaCore to avoid breaking precompilation.
# function ClimaComms.context(x::Fields.FieldVector)
#     values = Fields._values(x)
#     isempty(values) && error("Empty FieldVector has no device or context")
#     index = unrolled_findfirst(Base.Fix2(!isa, Fields.PointField), values)
#     return ClimaComms.context(values[isnothing(index) ? 1 : index])
# end

# Replacing numbers with Dual numbers in Y and p:

replace_parent_type(x::Fields.FieldVector, ::Type{T}) where {T} = similar(x, T)
replace_parent_type(x::Fields.Field, ::Type{T}) where {T} =
    Fields.Field(replace_parent_type(Fields.field_values(x), T), axes(x))
replace_parent_type(x::DataLayouts.AbstractData, ::Type{T}) where {T} =
    DataLayouts.replace_basetype(x, T)
replace_parent_type(x::Union{Tuple, NamedTuple}, ::Type{T}) where {T} =
    unrolled_map(Base.Fix2(replace_parent_type, T), x)

# Wrapping arrays in FieldVectors:

import ClimaCore: DataLayouts
import ClimaCore.DataLayouts: device_dispatch, parent_array_type
import ClimaCore.DataLayouts: replace_basetype, union_all, singleton
import ClimaCore.DataLayouts: type_params, farray_size, universal_size

# TODO: This must be moved into ClimaCore to avoid type piracy.
device_dispatch(x::PermutedDimsArray) = device_dispatch(parent(x))
parent_array_type(
    ::Type{<:Base.ReshapedArray{T, N, P, MI}},
) where {T, N, P, MI} = parent_array_type(P)
parent_array_type(
    ::Type{<:PermutedDimsArray{T, N, P, IP, A}},
) where {T, N, P, IP, A} = parent_array_type(A)

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

function column_vectors_to_field_vector(column_vectors, example_field_vector)
    example_field_vector_column = Fields.column(example_field_vector, 1, 1, 1)
    example_fields = values(Fields._values(example_field_vector))
    example_column_fields = values(Fields._values(example_field_vector_column))
    n_rows_in_fields = unrolled_map(length ∘ parent, example_column_fields)
    row_range_starts = unrolled_cumsum((1, n_rows_in_fields[1:(end - 1)]...))
    row_range_ends = row_range_starts .+ n_rows_in_fields .- 1
    row_ranges = unrolled_map(UnitRange, row_range_starts, row_range_ends)
    new_fields =
        unrolled_map(example_fields, row_ranges) do example_field, row_range
            new_data_layout = column_vectors_to_data_layout(
                view(column_vectors, row_range, :),
                Fields.field_values(example_field),
            )
            Fields.Field(new_data_layout, axes(example_field))
        end
    return Fields.FieldVector{eltype(column_vectors)}(
        NamedTuple{keys(Fields._values(example_field_vector))}(new_fields),
    )
end

# Reshape array with Nv×Nf rows and Ni×Nj×Nh columns to the specified layouts.
reshaped_column_vectors(array, data::DataLayouts.VF) =
    reshape(array, Nv(data), :)
reshaped_column_vectors(array, data::DataLayouts.IHF) =
    PermutedDimsArray(reshape(array, :, Ni(data), Nh(data)), (2, 3, 1))
reshaped_column_vectors(array, data::DataLayouts.IFH) =
    PermutedDimsArray(reshape(array, :, Ni(data), Nh(data)), (2, 1, 3))
reshaped_column_vectors(array, data::DataLayouts.IJHF) = PermutedDimsArray(
    reshape(array, :, Ni(data), Nj(data), Nh(data)),
    (2, 3, 4, 1),
)
reshaped_column_vectors(array, data::DataLayouts.IJFH) = PermutedDimsArray(
    reshape(array, :, Ni(data), Nj(data), Nh(data)),
    (2, 3, 1, 4),
)
reshaped_column_vectors(array, data::DataLayouts.VIHF) = PermutedDimsArray(
    reshape(array, Nv(data), :, Ni(data), Nh(data)),
    (1, 3, 4, 2),
)
reshaped_column_vectors(array, data::DataLayouts.VIFH) = PermutedDimsArray(
    reshape(array, Nv(data), :, Ni(data), Nh(data)),
    (1, 3, 2, 4),
)
reshaped_column_vectors(array, data::DataLayouts.VIJHF) = PermutedDimsArray(
    reshape(array, Nv(data), :, Ni(data), Nj(data), Nh(data)),
    (1, 3, 4, 5, 2),
)
reshaped_column_vectors(array, data::DataLayouts.VIJFH) = PermutedDimsArray(
    reshape(array, Nv(data), :, Ni(data), Nj(data), Nh(data)),
    (1, 3, 4, 2, 5),
)
Ni(data) = universal_size(data)[1]
Nj(data) = universal_size(data)[2]
Nv(data) = universal_size(data)[4]
Nh(data) = universal_size(data)[5]

# Lazily iterating over levels of scalars:

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

# Lazily iterating over columns:

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
