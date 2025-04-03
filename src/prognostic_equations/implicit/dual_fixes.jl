# # TODO: Move these to Thermodynamics.jl
TD.air_density(param_set, T, p, q) =
    TD.air_density(param_set, TD.promote_phase_partition(T, p, q)...)
TD.air_pressure(param_set, T, ρ, q) =
    TD.air_pressure(param_set, TD.promote_phase_partition(T, ρ, q)...)
TD.internal_energy(param_set, T, q) =
    TD.internal_energy(param_set, TD.promote_phase_partition(T, T, q)[2:end]...)
TD.internal_energy_sat(param_set, T, ρ, q_tot, type) =
    TD.internal_energy_sat(param_set, promote(T, ρ, q_tot)..., type)
TD.PhasePartition_equil(param_set, T, ρ, q_tot, type) =
    TD.PhasePartition_equil(param_set, promote(T, ρ, q_tot)..., type)
TD.PhasePartition_equil_given_p(param_set, T, p, q_tot, type) =
    TD.PhasePartition_equil_given_p(param_set, promote(T, p, q_tot)..., type)
# TD.specific_enthalpy_sat(param_set, T, ρ, q_tot, type) =
#     TD.specific_enthalpy_sat(param_set, promote(T, ρ, q_tot)..., type)
# TD.saturation_vapor_pressure(param_set, T, LH_0, Δcp) =
#     TD.saturation_vapor_pressure(param_set, promote(T, LH_0, Δcp)...)

# # TODO: Move these to SurfaceFluxes.jl
# import SurfaceFluxes as SF
# SF.Fluxes(int_values, surf_values, args...) =
#     SF.Fluxes(int_values, surf_values, promote(args...)...)
# SF.FluxesAndFrictionVelocity(int_values, surf_values, args...) =
#     SF.FluxesAndFrictionVelocity(int_values, surf_values, promote(args...)...)

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

filtered_names(f::F, x) where {F} = filtered_names_at_name(f, x, @name())
function filtered_names_at_name(f::F, x, name) where {F}
    field = MatrixFields.get_field(x, name)
    f(field) && return (name,)
    internal_names = MatrixFields.top_level_names(field)
    isempty(internal_names) && return ()
    tuples_of_names = MatrixFields.unrolled_map(internal_names) do internal_name
        Base.@_inline_meta
        child_name = MatrixFields.append_internal_name(name, internal_name)
        filtered_names_at_name(f, x, child_name)
    end
    return MatrixFields.unrolled_flatten(tuples_of_names)
end
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(filtered_names_at_name)
        m.recursion_relation = dont_limit
    end
end

scalar_field_names(fv) =
    filtered_names(x -> x isa Fields.Field && eltype(x) == eltype(fv), fv)

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
