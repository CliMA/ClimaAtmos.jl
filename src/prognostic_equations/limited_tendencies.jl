import ClimaCore: Limiters

"""
    ZhangShuPressureBound(R_d, cv_d, cv_v, T_0, e_int_v0, κ_min)

Concave lower bound [Pa] on the pressure of one node's conserved state `U`,
with all water taken as vapor and `off` the specific energy held unscaled.
"""
struct ZhangShuPressureBound{FT}
    R_d::FT
    cv_d::FT
    cv_v::FT
    T_0::FT
    e_int_v0::FT
    κ_min::FT
end
function ZhangShuPressureBound(params, ::Type{FT}) where {FT}
    R_d = FT(CAP.R_d(params))
    cv_d = FT(CAP.cv_d(params))
    cv_v = FT(CAP.cv_v(params))
    κ_min = min(R_d / cv_d, FT(CAP.R_v(params)) / cv_v)
    return ZhangShuPressureBound{FT}(
        R_d, cv_d, cv_v, FT(CAP.T_0(params)), FT(CAP.e_int_v0(params)), κ_min,
    )
end
@inline function (g::ZhangShuPressureBound)(U, off)
    ρq = haskey(U, :ρq_tot) ? U.ρq_tot : zero(U.ρ)
    ρe_int = U.ρe_tot - (U.ρu1^2 + U.ρu2^2) / (2 * U.ρ) - U.ρ * off
    ρcv_mT =
        ρe_int - ρq * g.e_int_v0 +
        (U.ρ - ρq) * (g.R_d + g.cv_d) * g.T_0 + ρq * g.cv_v * g.T_0
    return g.κ_min * ρcv_mT
end

"""
    ZhangShuAdmissibility(pressure_bound, ρ_min, p_min)

Concave admissibility constraint
`min(ρ − ρ_min, pressure_bound(U, off) − p_min) ≥ 0`.
"""
struct ZhangShuAdmissibility{G, FT}
    pressure_bound::G
    ρ_min::FT
    p_min::FT
end
@inline (g::ZhangShuAdmissibility)(U, off) =
    min(U.ρ - g.ρ_min, g.pressure_bound(U, off) - g.p_min)

"""
    ZhangShuLimiterCache(Y, params, FT, ρ_min, p_min)

Cache of `zhang_shu_limiter!`: the ClimaCore positivity limiters for the tracer
floors and the density [kg/m³] and pressure [Pa] floors, and the vapor density.
"""
struct ZhangShuLimiterCache{L1, L2, V}
    tracer_limiter::L1
    admissibility_limiter::L2
    ρq_vap::V
    ρq_vap_prev::V
end
Adapt.@adapt_structure ZhangShuLimiterCache

function ZhangShuLimiterCache(Y, params, ::Type{FT}, ρ_min, p_min) where {FT}
    water = hasproperty(Y.c, :ρq_tot)
    ρq_vap = water ? similar(Y.c.ρ) : nothing
    floored = zhang_shu_floor_states(Y, ρq_vap)
    tracer_limiter =
        isempty(floored) ? nothing :
        Limiters.PositivityLimiter(FT; floors = map(_ -> FT(0), floored))
    g = ZhangShuAdmissibility(ZhangShuPressureBound(params, FT), FT(ρ_min), FT(p_min))
    admissibility_limiter = Limiters.PositivityLimiter(FT; g, g_min = FT(0))
    ρq_vap_prev = water ? similar(Y.c.ρ) : nothing
    return ZhangShuLimiterCache(
        tracer_limiter,
        admissibility_limiter,
        ρq_vap,
        ρq_vap_prev,
    )
end

# The condensate species present in the grid-mean state `Y.c`.
condensate_names(Y) =
    filter(in((:ρq_lcl, :ρq_rai, :ρq_icl, :ρq_sno)), propertynames(Y.c))

"""
    zhang_shu_floor_states(Y, ρq_vap)

`NamedTuple` of the fields the Zhang–Shu floors act on: the non-water grid-mean
tracers and the vapor density.
"""
zhang_shu_floor_states(Y, ρq_vap) =
    isnothing(ρq_vap) ? zhang_shu_tracers(Y) :
    (;
        Base.structdiff(
            zhang_shu_tracers(Y),
            NamedTuple{(:ρq_tot, condensate_names(Y)...)},
        )...,
        ρq_vap,
    )

"""
    zhang_shu_limiter!(Y, p, lim)

Apply the Zhang–Shu (2010) positivity limiter to the grid-mean state in place,
keeping condensates, vapor, and tracers nonnegative and `ρ ≥ ρ_min`, `p ≥ p_min`
at every node while conserving element means. No-op when `lim` is `nothing`.
"""
zhang_shu_limiter!(Y, p, ::Nothing) = nothing
NVTX.@annotate function zhang_shu_limiter!(Y, p, lim::ZhangShuLimiterCache)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜT, ᶜK) = p.precomputed
    (; ᶜΦ) = p.core
    ᶜρu1 = p.scratch.ᶜtemp_scalar
    ᶜρu2 = p.scratch.ᶜtemp_scalar_2
    ᶜaux = p.scratch.ᶜtemp_scalar_3
    @. ᶜρu1 = Y.c.ρ * Geometry.UVVector(Y.c.uₕ).components.data.:1
    @. ᶜρu2 = Y.c.ρ * Geometry.UVVector(Y.c.uₕ).components.data.:2

    if !isnothing(lim.tracer_limiter)
        if !isnothing(lim.ρq_vap)
            # a negative condensate is diagnosed as zero, so zeroing it
            # changes nothing else
            lim.ρq_vap .= Y.c.ρq_tot
            for name in condensate_names(Y)
                ᶜρq = getproperty(Y.c, name)
                @. ᶜρq = max(zero(ᶜρq), ᶜρq)
                @. lim.ρq_vap -= ᶜρq
            end
            lim.ρq_vap_prev .= lim.ρq_vap
        end
        Limiters.apply_limiter!(
            zhang_shu_floor_states(Y, lim.ρq_vap),
            nothing,
            lim.tracer_limiter,
        )
        if !isnothing(lim.ρq_vap)
            # the vapor moved carries its mass and the element-mean energy
            # I_v + K + Φ, so every element mean stays exact
            ᶜδ = lim.ρq_vap_prev
            @. ᶜδ = lim.ρq_vap - ᶜδ
            @. Y.c.ρq_tot += ᶜδ
            @. Y.c.ρe_tot +=
                element_meanₕ(
                    TD.internal_energy_vapor(thermo_params, ᶜT) + ᶜK + ᶜΦ,
                ) * ᶜδ
            @. ᶜaux = element_meanₕ(ᶜρu1 / Y.c.ρ)
            @. ᶜρu1 += ᶜaux * ᶜδ
            @. ᶜaux = element_meanₕ(ᶜρu2 / Y.c.ρ)
            @. ᶜρu2 += ᶜaux * ᶜδ
            @. Y.c.ρ += ᶜδ
        end
        @. Y.c.uₕ = Geometry.Covariant12Vector(
            Geometry.UVVector(ᶜρu1 / Y.c.ρ, ᶜρu2 / Y.c.ρ),
        )
    end

    ᶜaux .= compute_kinetic(Y.c.uₕ, Y.f.u₃)
    @. ᶜaux += ᶜΦ - (ᶜρu1^2 + ᶜρu2^2) / (2 * Y.c.ρ^2)
    (; ρ, ρe_tot) = Y.c
    states = (; ρ, ρe_tot, ρu1 = ᶜρu1, ρu2 = ᶜρu2, zhang_shu_tracers(Y)...)
    Limiters.apply_limiter!(states, ᶜaux, lim.admissibility_limiter)
    @. Y.c.uₕ = Geometry.Covariant12Vector(
        Geometry.UVVector(ᶜρu1 / Y.c.ρ, ᶜρu2 / Y.c.ρ),
    )
    return nothing
end

"""
    zhang_shu_tracers(Y)

`NamedTuple` of the grid-mean tracer fields of `Y.c`.
"""
function zhang_shu_tracers(Y)
    names = filter(is_tracer_var, propertynames(Y.c))
    return NamedTuple{names}(map(name -> getproperty(Y.c, name), names))
end

"""
    _should_apply_limiter_to_tracer(ρχ_name, species) -> Bool

Return whether the vertical mass borrowing limiter applies to the tracer
`ρχ_name`.

# Arguments

  - `ρχ_name`: Tracer variable name, either a `Symbol` (e.g. `:ρq_tot`) or a
    `MatrixFields.FieldName`; it is only tested for membership in `species`.
  - `species`: Species selection. `nothing` selects all tracers; a `Tuple` selects
    only the tracers it lists, so the empty tuple selects none. Any other type is an
    error.

Called from `limiters_func!`.
"""
function _should_apply_limiter_to_tracer(ρχ_name, species)
    if isnothing(species)
        return true  # Apply to all tracers
    elseif species isa Tuple
        return ρχ_name in species
    else
        error("Invalid species configuration type: $(typeof(species))")
    end
end

"""
    limiters_func!(Y, p, t, ref_Y)

Apply the configured tracer limiters to the prognostic state `Y` in place.

Two tracer limiters may be active, in this order, each skipped when its entry
in `p.numerics` is `nothing`:

 1. **SEM quasimonotone limiter** (`sem_quasimonotone_limiter`): computes bounds
    from the reference state `ref_Y` and applies spectral-element limiting to every
    grid-mean tracer.
 2. **Vertical mass borrowing limiter** (`vertical_water_borrowing_limiter`):
    enforces nonnegativity by borrowing mass from vertical neighbours. It is
    constructed in the cache as `Limiters.VerticalMassBorrowingLimiter((FT(0),))`,
    so the threshold is zero for every tracer it touches. Because `apply_limiter!`
    takes no species argument, the selection in
    `p.numerics.vertical_water_borrowing_species` is applied here, by filtering the
    loop over tracers (see `_should_apply_limiter_to_tracer`). It operates on the
    specific tracer `χ = ρχ/ρ` held in scratch, then writes `ρχ` back.

Whenever a limiter changes `ρq_tot`, the induced increment `Δ(ρq_tot)` is measured
from the pre- and post-limited states and passed to
`enforce_mass_energy_consistency!`, which updates density and total energy; see the
"Microphysics" page of the docs (`docs/src/microphysics.md`).

# Arguments

  - `Y`: Current state vector, modified in place.
  - `p`: Cache; the limiter configuration is read from `p.numerics` and working
    fields from `p.scratch`.
  - `t`: Current simulation time; unused.
  - `ref_Y`: Reference state used to compute the quasimonotone bounds.

Returns `nothing`.
"""
NVTX.@annotate function limiters_func!(Y, p, t, ref_Y)
    (;
        sem_quasimonotone_limiter,
        vertical_water_borrowing_limiter,
        vertical_water_borrowing_species,
    ) =
        p.numerics

    # Apply general (SEM quasimonotone) limiter if configured.
    # When ρq_tot is limited, update ρ and ρe_tot for mass and energy consistency.
    if !isnothing(sem_quasimonotone_limiter)
        if hasproperty(Y.c, :ρq_tot)
            p.scratch.ᶜtemp_scalar_2 .= Y.c.ρq_tot
        end
        for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
            Limiters.compute_bounds!(
                sem_quasimonotone_limiter,
                ref_Y.c.:($ρχ_name),
                ref_Y.c.ρ,
            )
            Limiters.apply_limiter!(Y.c.:($ρχ_name), Y.c.ρ, sem_quasimonotone_limiter)
        end
        if hasproperty(Y.c, :ρq_tot)
            @. p.scratch.ᶜtemp_scalar_2 = Y.c.ρq_tot - p.scratch.ᶜtemp_scalar_2
            enforce_mass_energy_consistency!(Y, p, p.scratch.ᶜtemp_scalar_2)
        end
    end

    # Apply vertical water borrowing limiter if configured
    # Our state stores ρχ (tracer density). Store χ in scratch, apply limiter, then write ρχ back.
    # When ρq_tot is limited, update ρ and ρe_tot for mass and energy consistency.
    if !isnothing(vertical_water_borrowing_limiter)
        if _should_apply_limiter_to_tracer(
            @name(ρq_tot),
            vertical_water_borrowing_species,
        ) &&
           hasproperty(Y.c, :ρq_tot)
            p.scratch.ᶜtemp_scalar_2 .= Y.c.ρq_tot
        end
        ᶜχ = p.scratch.ᶜtemp_scalar
        for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
            if _should_apply_limiter_to_tracer(ρχ_name, vertical_water_borrowing_species)
                ρχ = getproperty(Y.c, ρχ_name)
                ᶜχ .= specific.(ρχ, Y.c.ρ)
                Limiters.apply_limiter!(ᶜχ, Y.c.ρ, vertical_water_borrowing_limiter)
                ρχ .= ᶜχ .* Y.c.ρ
            end
        end
        if _should_apply_limiter_to_tracer(
            @name(ρq_tot),
            vertical_water_borrowing_species,
        ) &&
           hasproperty(Y.c, :ρq_tot)
            @. p.scratch.ᶜtemp_scalar_2 = Y.c.ρq_tot - p.scratch.ᶜtemp_scalar_2
            enforce_mass_energy_consistency!(Y, p, p.scratch.ᶜtemp_scalar_2)
        end
    end
    return nothing
end
