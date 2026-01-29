import ClimaCore: Limiters

"""
    _should_apply_limiter_to_tracer(ρχ_name, species) -> Bool

Determine if a limiter should be applied to a specific tracer.

# Arguments
- `ρχ_name::Symbol`: Tracer variable name (e.g., `:ρq_tot`)
- `species`: Configuration — `nothing`: apply to all; `()`: apply to none;
  `Tuple{Symbol,...}`: apply only if `ρχ_name ∈ species`

# Returns
`true` if the limiter should be applied, `false` otherwise.
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

Apply tracer limiters to the prognostic state `Y` in place.

Supports two limiter types configured via `p.numerics`:

1. **SEM Quasimonotone Limiter** (`sem_quasimonotone_limiter`):
   Computes bounds from `ref_Y` and applies spectral element limiting.

2. **Vertical Mass Borrowing Limiter** (`vertical_water_borrowing_limiter`):
   Enforces strict nonnegativity by borrowing mass vertically.
   Species filtering via `vertical_water_borrowing_species`.

# Arguments
- `Y`: Current state vector (modified in place)
- `p`: Cache containing `p.numerics` limiter configuration
- `t`: Current simulation time
- `ref_Y`: Reference state for bounds computation
"""
NVTX.@annotate function limiters_func!(Y, p, t, ref_Y)
    (;
        sem_quasimonotone_limiter,
        vertical_water_borrowing_limiter,
        vertical_water_borrowing_species,
    ) =
        p.numerics

    # Apply general limiter if configured
    if !isnothing(sem_quasimonotone_limiter)
        for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
            Limiters.compute_bounds!(
                sem_quasimonotone_limiter,
                ref_Y.c.:($ρχ_name),
                ref_Y.c.ρ,
            )
            Limiters.apply_limiter!(Y.c.:($ρχ_name), Y.c.ρ, sem_quasimonotone_limiter)
        end
    end

    # Apply vertical water borrowing limiter if configured
    # Our state stores ρχ (tracer density). Store χ in scratch, apply limiter, then write ρχ back.
    # Also note: species filtering is done here, not passed to apply_limiter! (which doesn't support it)
    if !isnothing(vertical_water_borrowing_limiter)
        ᶜχ = p.scratch.ᶜtemp_scalar
        for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
            if _should_apply_limiter_to_tracer(ρχ_name, vertical_water_borrowing_species)
                ρχ = getproperty(Y.c, ρχ_name)
                ᶜχ .= specific.(ρχ, Y.c.ρ)
                Limiters.apply_limiter!(ᶜχ, Y.c.ρ, vertical_water_borrowing_limiter)
                ρχ .= ᶜχ .* Y.c.ρ
            end
        end
    end
    return nothing
end
