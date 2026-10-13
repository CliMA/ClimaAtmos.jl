import ClimaCore: Limiters

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

Two limiters may be active, in this order, each skipped when its entry in
`p.numerics` is `nothing`:

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
