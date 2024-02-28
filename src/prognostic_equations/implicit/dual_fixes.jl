# TODO: Move these to Thermodynamics.jl
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
TD.specific_enthalpy_sat(param_set, T, ρ, q_tot, type) =
    TD.specific_enthalpy_sat(param_set, promote(T, ρ, q_tot)..., type)
TD.saturation_vapor_pressure(param_set, T, LH_0, Δcp) =
    TD.saturation_vapor_pressure(param_set, promote(T, LH_0, Δcp)...)

# TODO: Move these to SurfaceFluxes.jl
import SurfaceFluxes as SF
SF.FluxesAndFrictionVelocity(int_values, surf_values, args...) =
    SF.FluxesAndFrictionVelocity(int_values, surf_values, promote(args...)...)

# TODO: Move this to ClimaCore.jl
ClimaComms.device(fv::Fields.FieldVector) =
    ClimaComms.device(first(Fields._values(fv)))

# Ensure that we can broadcast between FieldVectors and other BlockVectors, but
# without going through the type-unstable BlockArrays broadcasting code.
# TODO: Replace the FieldVector broadcasting code in ClimaCore.jl with this
struct NewFieldVectorStyle <: Base.Broadcast.AbstractArrayStyle{1} end
Base.Broadcast.BroadcastStyle(::Type{<:Fields.FieldVector}) =
    NewFieldVectorStyle()
Base.Broadcast.BroadcastStyle(
    fs::NewFieldVectorStyle,
    as::Base.Broadcast.DefaultArrayStyle{N},
) where {N} = N in (0, 1) ? fs : as # prevents an ambiguity with Base.Broadcast
Base.Broadcast.BroadcastStyle(
    fs::NewFieldVectorStyle,
    as::Base.Broadcast.AbstractArrayStyle{N},
) where {N} = N in (0, 1) ? fs : as
Base.@propagate_inbounds function BlockArrays.viewblock(
    fv::Fields.FieldVector,
    block::BlockArrays.Block{1},
)
    array = Fields.backing_array(Fields._values(fv)[block.n...])
    # vec(array) allocates; see https://github.com/JuliaLang/julia/issues/36313
    return Base.ReshapedArray(array, (length(array),), ())
end

first_field_vector(x::Fields.FieldVector, args...) = x
@inline first_field_vector(x::Base.Broadcast.Broadcasted, args...) =
    first_field_vector(x.args..., args...)
@inline first_field_vector(_, args...) = first_field_vector(args...)
check_broadcast_axes(fv, x::Base.Broadcast.Broadcasted) =
    check_broadcast_args_axes(fv, x.args...)
@inline check_broadcast_axes(fv, x) =
    isempty(axes(x)) || # scalar 
    axes(x) == axes(fv) || # AbstractBlockVector
    x isa AbstractVector && length(x) == length(fv) || # fallback
    error("Mismatched broadcast axes: $(axes(x)) != $(axes(fv))")
check_broadcast_args_axes(_) = true
@inline check_broadcast_args_axes(fv, arg, args...) =
    check_broadcast_axes(fv, arg) && check_broadcast_args_axes(fv, args...)

block_view(x, _, _) = x
Base.@propagate_inbounds block_view(
    x::BlockArrays.AbstractBlockVector,
    ::Val{index},
    _,
) where {index} = BlockArrays.viewblock(x, BlockArrays.Block(index))
Base.@propagate_inbounds block_view(x::AbstractVector, val, fv) =
    block_view(BlockArrays.PseudoBlockVector(x, axes(fv)), val, fv)
Base.@propagate_inbounds block_view(x::Base.Broadcast.Broadcasted, val, fv) =
    Base.Broadcast.broadcasted(x.f, block_view_args(val, fv, x.args...)...)
block_view_args(_, _) = ()
Base.@propagate_inbounds block_view_args(val, fv, arg, args...) =
    (block_view(arg, val, fv), block_view_args(val, fv, args...)...)

# Bypass the method for materialize! defined in BlockArrays, which results in
# type instabilities and allocations.
Base.materialize!(
    dest::Fields.FieldVector,
    bc::Base.Broadcast.Broadcasted{NewFieldVectorStyle},
) = materialize_field_vector!(dest, bc)
Base.materialize!(
    dest::AbstractVector,
    bc::Base.Broadcast.Broadcasted{NewFieldVectorStyle},
) = materialize_field_vector!(dest, bc)
Base.materialize!(
    dest::Fields.FieldVector,
    bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}},
) = materialize_field_vector!(dest, bc)
Base.materialize!(
    dest::Fields.FieldVector,
    bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{1}},
) = materialize_field_vector!(dest, bc)
Base.materialize!(
    dest::Fields.FieldVector,
    bc::Base.Broadcast.Broadcasted{<:BlockArrays.AbstractBlockStyle{1}},
) = materialize_field_vector!(dest, bc) # prevents an ambiguity with BlockArrays
function materialize_field_vector!(dest, bc)
    fv = first_field_vector(dest, bc)
    check_broadcast_args_axes(fv, dest, bc)
    block_index_vals = ntuple(Val, BlockArrays.blocksize(fv, 1))
    MatrixFields.unrolled_foreach(block_index_vals) do val
        dest_view = block_view(dest, val, fv)
        bc_view = block_view(bc, val, fv)
        @inbounds Base.materialize!(dest_view, bc_view)
    end
    return dest
end
Base.@propagate_inbounds function Base.getindex(
    fv::Fields.FieldVector,
    i::Integer,
)
    @warn "scalar indexing into FieldVector (this can be very slow)" maxlog = 1
    getindex(fv, BlockArrays.findblockindex(axes(fv, 1), i))
end
Base.@propagate_inbounds function Base.setindex!(
    fv::Fields.FieldVector,
    val,
    i::Integer,
)
    @warn "scalar indexing into FieldVector (this can be very slow)" maxlog = 1
    setindex!(fv, val, BlockArrays.findblockindex(axes(fv, 1), i))
end

# TODO: Move all of the following to ClimaCore.jl

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

function column_iterator(field_vector)
    example_field = field_vector.c # TODO: Generalize this.
    horz_space = Spaces.horizontal_space(axes(example_field))
    qs = 1:Quadratures.degrees_of_freedom(Spaces.quadrature_style(horz_space))
    hs = Spaces.eachslabindex(horz_space)
    return if Fields.field_values(example_field) isa DataLayouts.VIFH
        Iterators.map(Iterators.product(qs, hs)) do (i, h)
            field_vector[Fields.ColumnIndex((i,), h)]
        end
    else
        @assert Fields.field_values(example_field) isa DataLayouts.VIJFH
        Iterators.map(Iterators.product(qs, qs, hs)) do (i, j, h)
            field_vector[Fields.ColumnIndex((i, j), h)]
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
