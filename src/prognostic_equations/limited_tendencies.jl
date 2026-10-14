import ClimaCore: Limiters

"""
    ZhangShuPressureBound(R_d, cv_d, cv_v, T_0, e_int_v0, κ_min)

Constraint `g(U, off)` of the Zhang–Shu limiter: a lower bound on the pressure
of one node's conserved state `U = (; ρ, ρe_tot, ρu1, ρu2[, ρq_tot, ...])`,
with `off` the specific energy held unscaled (see `zhang_shu_limiter!`).

With all water taken as vapor, the Thermodynamics internal energy
`e_int = cv_m (T − T_0) + q e_int_v0 − (1 − q) R_d T_0` gives the thermal energy

    ρ cv_m T = ρe_int − ρq e_int_v0 + (ρ − ρq)(R_d + cv_d) T_0 + ρq cv_v T_0,

with `ρe_int = ρe_tot − |ρuₕ|²/(2ρ) − ρ off`. It is linear in `(ρ, ρe_tot, ρq)`
minus the convex `|ρuₕ|²/(2ρ)`, so concave in `U`, as the limiter's bisection
requires. Since `p = (R_m/cv_m) ρ cv_m T` and
`R_m/cv_m ≥ κ_min = min(R_d/cv_d, R_v/cv_v)`, `g = κ_min ρ cv_m T ≤ p` wherever
`T > 0`, so `g ≥ p_min` implies `p ≥ p_min`. Taking condensate as vapor only lowers the
estimate of `T` (by the latent heat of the condensate; phase differences in
heat capacity are neglected), which keeps the bound on the admissible side.
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
    zhang_shu_limiter!(Y, p, lim)

Apply the Zhang–Shu (2010) positivity limiter (a ClimaCore
`Limiters.PositivityLimiter`) to the grid-mean state, in place; a no-op when
`lim` is `nothing`.

The limiter acts on a conserved vector, so the vector-invariant state is
mapped to one: the prognostic densities (ρ, ρe_tot, and every grid-mean
tracer) are passed directly, the horizontal momentum is staged in scratch as
the orthonormal components ρu1/ρu2 and written back to `uₕ` after limiting,
and the face vertical velocity is left unscaled. The auxiliary field
`off = K − (u1² + u2²)/2 + Φ` carries the rest of the specific energy, with
`K` the model's kinetic energy (`compute_kinetic`), so that at θ = 1 the
constraint sees the same internal energy as the dynamics, including the
vertical-velocity and terrain cross terms.

In each element, ρ and each tracer are first scaled toward their own
WJ-weighted mean to meet `ρ ≥ ρ_min` and `ρχ ≥ 0`; the whole vector is then
scaled toward its mean by one θ to meet `p ≥ p_min`, with the pressure bound
from [`ZhangShuPressureBound`](@ref). Each element's means of ρ, ρe_tot, the
tracers, and the orthonormal ρuₕ are preserved exactly, so no mass-energy
consistency fixup is needed.

Called from `constrain_state!`, which the timestepper runs on the complete
stage value (after the explicit and implicit increments and the DSS). The
`update_constrain_state_every` cadence must therefore be `"stage"` or `"dss"`;
`AtmosSimulation` rejects `"step"`.
"""
zhang_shu_limiter!(Y, p, ::Nothing) = nothing
NVTX.@annotate function zhang_shu_limiter!(Y, p, lim::Limiters.PositivityLimiter)
    ᶜΦ = p.core.ᶜΦ
    ᶜρu1 = p.scratch.ᶜtemp_scalar
    ᶜρu2 = p.scratch.ᶜtemp_scalar_2
    ᶜoff = p.scratch.ᶜtemp_scalar_3

    @. ᶜρu1 = Y.c.ρ * Geometry.UVVector(Y.c.uₕ).components.data.:1
    @. ᶜρu2 = Y.c.ρ * Geometry.UVVector(Y.c.uₕ).components.data.:2
    ᶜoff .= compute_kinetic(Y.c.uₕ, Y.f.u₃)
    @. ᶜoff += ᶜΦ - (ᶜρu1^2 + ᶜρu2^2) / (2 * Y.c.ρ^2)

    (; ρ, ρe_tot) = Y.c
    states = (; ρ, ρe_tot, ρu1 = ᶜρu1, ρu2 = ᶜρu2, zhang_shu_tracers(Y)...)
    Limiters.apply_limiter!(states, ᶜoff, lim)

    @. Y.c.uₕ = Geometry.Covariant12Vector(
        Geometry.UVVector(ᶜρu1 / Y.c.ρ, ᶜρu2 / Y.c.ρ),
    )
    return nothing
end

"""
    zhang_shu_tracers(Y)

`NamedTuple` of the grid-mean tracer fields of `Y.c` (every name for which
`is_tracer_var` holds), which the Zhang–Shu limiter keeps nonnegative.
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
