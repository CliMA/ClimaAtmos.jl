# TODO: Move all of the following to ClimaCore.jl

using ClimaCore: Adapt, Grids
Adapt.@adapt_structure Grids.ColumnGrid

# Fix broadcasting bug.
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

# Add missing methods.
# ClimaComms.device(fv::Fields.FieldVector) =
#     ClimaComms.device(first(Fields._values(fv)))
# ClimaComms.context(fv::Fields.FieldVector) =
#     ClimaComms.context(first(Fields._values(fv)))

# import ClimaCore.DataLayouts: array2data # Avoid breaking precompilation.
import ClimaCore.DataLayouts: replace_basetype, union_all, singleton
import ClimaCore.DataLayouts: type_params, farray_size
function array2data(array::AbstractArray, data::DataLayouts.AbstractData)
    T = replace_basetype(eltype(parent(data)), eltype(array), eltype(data))
    return union_all(singleton(data)){T, Base.tail(type_params(data))...}(
        reshape(array, farray_size(data)...),
    )
end

import ClimaCore.Fields: BlockArrays
import UnrolledUtilities: unrolled_map
function vector_to_fieldvector(vector, Y_prototype)
    block_vector = BlockArrays.BlockedArray(vector, axes(Y_prototype))
    field_views =
        unrolled_map(enumerate(Fields._values(Y_prototype))) do (index, value)
            vector_view =
                BlockArrays.viewblock(block_vector, BlockArrays.Block(index))
            data_layout_view =
                array2data(vector_view, Fields.field_values(value))
            Fields.Field(data_layout_view, axes(value))
        end
    return Fields.FieldVector{eltype(vector)}(field_views)
end

# import ClimaCore: DataLayouts

# # column_vectors are assumed to be (Nv * Nf) × (Ni * Nj * Nh)
# function column_vectors_to_data(column_vectors, data)
#     S = eltype(data)
#     T = replace_basetype(eltype(parent(data)), eltype(column_vectors), S)
#     return union_all(singleton(data)){T, Base.tail(type_params(data))...}(
#         reshaped_column_vectors(column_vectors, data),
#     )
# end
# Ni(data) = DataLayouts.universal_size(data)[1]
# Nj(data) = DataLayouts.universal_size(data)[2]
# Nv(data) = DataLayouts.universal_size(data)[4]
# Nh(data) = DataLayouts.get_Nh_dynamic(data)
# reshaped_column_vectors(array, data::DataLayouts.VF) =
#     reshape(array, Nv(data), :)
# reshaped_column_vectors(array, ::DataLayouts.IHF) =
#     reshape(array, Ni(data), Nh(data), :)
# reshaped_column_vectors(array, ::DataLayouts.IFH) = PermutedDimsArray(
#     reshape(array, Ni(data), Nh(data), :),
#     (1, 3, 2),
# )
# reshaped_column_vectors(array, ::DataLayouts.IJHF) =
#     reshape(array, Ni(data), Nj(data), Nh(data), :)
# reshaped_column_vectors(array, ::DataLayouts.IJFH) = PermutedDimsArray(
#     reshape(array, Ni(data), Nj(data), Nh(data), :),
#     (1, 2, 4, 3),
# )
# reshaped_column_vectors(array, ::DataLayouts.VIHF) = PermutedDimsArray(
#     reshape(array, Ni(data), Nh(data), Nv(data), :),
#     (3, 1, 2, 4),
# )
# reshaped_column_vectors(array, ::DataLayouts.VIFH) = PermutedDimsArray(
#     reshape(array, Ni(data), Nh(data), Nv(data), :),
#     (3, 1, 4, 2),
# )
# reshaped_column_vectors(array, ::DataLayouts.VIJHF) = PermutedDimsArray(
#     reshape(array, Ni(data), Nj(data), Nh(data), Nv(data), :),
#     (4, 1, 2, 3, 5),
# )
# reshaped_column_vectors(array, ::DataLayouts.VIJFH) = PermutedDimsArray(
#     reshape(array, Ni(data), Nj(data), Nh(data), Nv(data), :),
#     (4, 1, 2, 5, 3),
# )

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
