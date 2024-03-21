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
import .SurfaceConditions: SF
SF.Fluxes(state_in, state_sfc, shf, lhf, z0m, z0b, gustiness) =
    SF.Fluxes(state_in, state_sfc, promote(shf, lhf, z0m, z0b, gustiness)...)
SF.FluxesAndFrictionVelocity(
    state_in,
    state_sfc,
    shf,
    lhf,
    ustar,
    z0m,
    z0b,
    gustiness,
) = SF.FluxesAndFrictionVelocity(
    state_in,
    state_sfc,
    promote(shf, lhf, ustar, z0m, z0b, gustiness)...,
)
SF.Coefficients(state_in, state_sfc, Cd, Ch, gustiness, beta) =
    SF.Coefficients(state_in, state_sfc, promote(Cd, Ch, gustiness, beta)...)
SF.ValuesOnly(state_in, state_sfc, z0m, z0b, gustiness, beta) =
    SF.ValuesOnly(state_in, state_sfc, promote(z0m, z0b, gustiness, beta)...)

# TODO: Move these to CloudMicrophysics.jl
import CloudMicrophysics.Microphysics1M as CM1
CM1.get_v0(
    (; C_drag, ρw, grav, r0)::CM1.CMP.Blk1MVelTypeRain{FT},
    ρ,
) where {FT} = sqrt(FT(8 / 3) / C_drag * (ρw / ρ - FT(1)) * grav * r0)
function CM1.lambda(
    pdf::Union{CM1.CMP.ParticlePDFIceRain{FT}, CM1.CMP.ParticlePDFSnow{FT}},
    mass::CM1.CMP.ParticleMass{FT},
    q,
    ρ,
) where {FT}
    # size distribution
    n0 = CM1.get_n0(pdf, q, ρ)
    # mass(size)
    (; r0, m0, me, Δm, χm) = mass

    return q > FT(0) ?
           (
        χm * m0 * n0 * CM1.SF.gamma(me + Δm + FT(1)) / ρ / q / r0^(me + Δm)
    )^FT(1 / (me + Δm + 1)) : FT(0)
end
CM1.terminal_velocity(
    (; pdf, mass)::Union{CM1.CMP.Rain{FT}, CM1.CMP.Snow{FT}},
    vel::Union{CM1.CMP.Blk1MVelTypeRain, CM1.CMP.Blk1MVelTypeSnow},
    ρ,
    q,
) where {FT} =
    if q > FT(0)
        # terminal_velocity(size)
        (; χv, ve, Δv) = vel
        v0 = CM1.get_v0(vel, ρ)
        # mass(size)
        (; r0, me, Δm, χm) = mass
        # size distrbution
        λ = CM1.lambda(pdf, mass, q, ρ)

        return χv *
               v0 *
               (λ * r0)^(-ve - Δv) *
               CM1.SF.gamma(me + ve + Δm + Δv + FT(1)) /
               CM1.SF.gamma(me + Δm + FT(1))
    else
        return FT(0)
    end

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

# Custom implementation of bycolumn for iterating over columns on both CPUs and
# GPUs (bycolumn does not support GPUs).
# TODO: Generalize this and move it to ClimaCore.jl
function column_iterator(field_vectors...)
    n_field_vectors = length(field_vectors)
    @assert n_field_vectors >= 1
    example_field = field_vectors[1].c

    horz_space = Spaces.horizontal_space(axes(example_field))
    qs = 1:Quadratures.degrees_of_freedom(Spaces.quadrature_style(horz_space))
    hs = Spaces.eachslabindex(horz_space)
    colidx_iterator = if Fields.field_values(example_field) isa DataLayouts.VIFH
        Iterators.map(Iterators.product(qs, hs)) do (i, h)
            Fields.ColumnIndex((i,), h)
        end
    else
        @assert Fields.field_values(example_field) isa DataLayouts.VIJFH
        Iterators.map(Iterators.product(qs, qs, hs)) do (i, j, h)
            Fields.ColumnIndex((i, j), h)
        end
    end
    return Iterators.map(colidx_iterator) do colidx
        field_vector_columns = map(field_vectors) do fv
            @assert all(in((:c, :f, :sfc)), propertynames(fv))
            has_sfc = hasproperty(fv, :sfc)
            Fields.FieldVector(;
                c = fv.c[colidx],
                f = fv.f[colidx],
                (has_sfc ? (; sfc = fv.sfc[colidx]) : (;))...,
            )
        end
        n_field_vectors == 1 ? field_vector_columns[1] : field_vector_columns
    end
end

# TODO: Move this to ClimaCore.jl
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
