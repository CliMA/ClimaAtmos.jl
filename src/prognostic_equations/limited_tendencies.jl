import ClimaCore: Limiters

"""
    zhang_shu_limiter_func!(Y, p, lim)

Apply the Zhang–Shu (2010) positivity limiter (a ClimaCore
`Limiters.PositivityLimiter`) to the grid-mean state, in place.

The limiter interface takes a conserved vector, so the vector-invariant state
is mapped to one: the prognostic densities (ρ, ρe_tot[, ρq_tot]) are passed
directly, the horizontal momentum is staged in scratch as the orthonormal
components ρuE/ρuN (single-valued at cubed-sphere panel edges) and written
back to `uₕ` after limiting, and the face vertical velocity is left unscaled,
its kinetic energy carried through the unscaled offset `off = w_c²/2 + Φ` (the
same convention as the FDDG flux-form drivers). Each element's WJ-weighted
means of ρ, ρe_tot, ρq_tot, and the orthonormal ρuₕ are preserved exactly.

The pressure floor is enforced on a dry ideal-gas proxy of the EOS,
`p = ρ R_d ((ρe/ρ − Kₕ − off)/cv_d + T_0)` — moisture corrections to R and cv
are irrelevant at floor magnitudes, and the proxy is a cheap closed form for
the per-node bisection in θ (a saturation adjustment there would be
prohibitive and is not needed for admissibility).

Called from `limiters_func!` when `p.numerics.zhang_shu_limiter` is set (the
`apply_zhang_shu_limiter` configuration option).
"""
NVTX.@annotate function zhang_shu_limiter_func!(Y, p, lim)
    (; params) = p
    R_d = CAP.R_d(params)
    cv_d = CAP.cv_d(params)
    T_0 = CAP.T_0(params)
    ᶜΦ = p.core.ᶜΦ
    ᶜρu1 = p.scratch.ᶜtemp_scalar
    ᶜρu2 = p.scratch.ᶜtemp_scalar_2
    ᶜρu3 = p.scratch.ᶜtemp_scalar_3
    ᶜoff = p.scratch.ᶜtemp_scalar_4
    ᶜinterp = Operators.InterpolateF2C()

    @. ᶜρu1 = Y.c.ρ * Geometry.UVVector(Y.c.uₕ).components.data.:1
    @. ᶜρu2 = Y.c.ρ * Geometry.UVVector(Y.c.uₕ).components.data.:2
    fill!(parent(ᶜρu3), 0)
    @. ᶜoff =
        Geometry.WVector(ᶜinterp(Y.f.u₃)).components.data.:1^2 / 2 + ᶜΦ

    pfn = let R_d = R_d, cv_d = cv_d, T_0 = T_0
        (ρ, ρe, ρu1, ρu2, ρu3, ρq, off) -> begin
            K_h = (ρu1^2 + ρu2^2 + ρu3^2) / (2 * ρ^2)
            ρ * R_d * ((ρe / ρ - K_h - off) / cv_d + T_0)
        end
    end
    states =
        hasproperty(Y.c, :ρq_tot) ?
        (Y.c.ρ, Y.c.ρe_tot, ᶜρu1, ᶜρu2, ᶜρu3, Y.c.ρq_tot) :
        (Y.c.ρ, Y.c.ρe_tot, ᶜρu1, ᶜρu2, ᶜρu3)
    Limiters.apply_positivity_limiter!(lim, pfn, states, ᶜoff)

    @. Y.c.uₕ = Geometry.Covariant12Vector(
        Geometry.UVVector(ᶜρu1 / Y.c.ρ, ᶜρu2 / Y.c.ρ),
    )
    return nothing
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

When `p.numerics.zhang_shu_limiter` is set (mutually exclusive with the
quasimonotone limiter), the Zhang–Shu positivity limiter is applied first;
see `zhang_shu_limiter_func!`. Then two tracer limiters may be active, in
this order, each skipped when its entry in `p.numerics` is `nothing`:

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
        zhang_shu_limiter,
        vertical_water_borrowing_limiter,
        vertical_water_borrowing_species,
    ) =
        p.numerics

    # Zhang–Shu positivity limiter (mutually exclusive with the quasimonotone
    # limiter; enforced at configuration time). Scales the whole conserved
    # vector by one θ, so no mass-energy consistency fixup is needed.
    if !isnothing(zhang_shu_limiter)
        zhang_shu_limiter_func!(Y, p, zhang_shu_limiter)
    end

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
