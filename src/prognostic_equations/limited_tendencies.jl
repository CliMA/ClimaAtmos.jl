import ClimaCore: Limiters

"""
    _should_apply_limiter_to_tracer(ρχ_name, species)

Helper function to determine if a limiter should be applied to a specific tracer based on
the species configuration.

Arguments:
- `ρχ_name`: Symbol name of the tracer variable (e.g., :ρq_tot)
- `species`: Species configuration — `nothing` (or config omitted/~): apply to all tracers;
  empty tuple `()`: apply to none; `Tuple{Symbol, ...}` (e.g. from config `["ρq_tot"]`): apply only if ρχ_name is in the tuple.

Returns:
- `true` if the limiter should be applied to this tracer, `false` otherwise
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

Applies tracer limiters to the prognostic state `Y` in place, using parameters `p`,
time `t`, and a reference state `ref_Y` for bounds determination.

This function iterates over all identified tracer variables within `Y.c`. If a limiter
is configured in `p.numerics.sem_quasimonotone_limiter`, it first calls `ClimaCore.Limiters.compute_bounds!`
to establish the permissible range for each tracer (based on `ref_Y.c.:(ρχ_name)`
and `ref_Y.c.ρ`). Subsequently, `ClimaCore.Limiters.apply_limiter!` is called to modify
`Y.c.:(ρχ_name)` (and implicitly `Y.c.ρ` if the limiter adjusts both to maintain the
specific quantity `ρχ/ρ` within bounds or if it only modifies `ρχ` based on `ρ`),
ensuring the tracer quantities adhere to these bounds.

For the VerticalMassBorrowingLimiter, if `p.numerics.vertical_water_borrowing_limiter`
is configured, it uses that limiter with `p.numerics.vertical_water_borrowing_species` for species selection.
The limiter enforces strict nonnegativity using a single threshold value (0.0) that applies uniformly
to all tracers for which the limiter is used. The limiter instance is created as
`Limiters.VerticalMassBorrowingLimiter((0.0,))` in the cache. The species configuration is used to
filter which tracers the limiter is applied to before calling `apply_limiter!` (since `apply_limiter!`
doesn't support a species keyword argument). If species is `nothing` (default), the limiter is applied
to all tracers. Otherwise, only tracers matching the specified tuple of names will have the limiter
applied. The tuple format for the threshold is required by ClimaCore's API (for GPU compatibility),
but only a single threshold value is used for all tracers.

Arguments:
- `Y`: The current state vector (`ClimaCore.Fields.FieldVector`), modified in place.
- `p`: A cache or parameters object, containing `p.numerics.sem_quasimonotone_limiter`, `p.numerics.vertical_water_borrowing_limiter`,
       and `p.numerics.vertical_water_borrowing_species`.
- `t`: The current simulation time (often unused by the limiter itself but part of a standard signature).
- `ref_Y`: A reference state vector (`ClimaCore.Fields.FieldVector`) used to compute the
           tracer bounds (e.g., ensuring positivity or monotonicity relative to this state).
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

"""
    triangle_inequality_limiter(force, allowed_source_amount, limit_neg=0)

Limits a `force` (or source term) based on maximum allowable limits using a
formula derived from the triangle inequality, as proposed by Horn (2012).

The formula used is: `L = F + M - sqrt(F² + M²)`, where `F` is the `force` and
`M` is the `allowed_source_amount`.

This limiter is designed to smoothly reduce the `force` as it approaches or
exceeds the `allowed_source_amount`, ensuring the result `L` satisfies `0 ≤ L ≤ M` if `F ≥ 0`
and `M > 0`. It also preserves `L ≤ F`. It's particularly useful for ensuring
that source terms (e.g., emissions, chemical production rates) do not become
unphysically large or lead to numerical instability, while being continuously
differentiable.


Due to numerical errors, `allowed_source_amount` can be negative.
This spurious behavior leads to returned `force` switching signs.
In this case the source and destination categories are swapped.
We try to limit this new force by the amount available in the old destination category.
If both source and destination are negative, we return zero, as there is nothing the limiter can do.

Arguments:
- `force`: The original force or source term value (positive or negative).
- `intended_source_amount`: The available mass in the source category (positive normally,
  but can be negative due to numerical errors).
- `limit_neg`: The available mass in the destination category for reverse conversions
  (defaults to 0). Only used when `intended_source_amount ≤ 0`.

Returns:
- The limited force value, guaranteed not to deplete more mass than available in either category.

Reference:
- Horn, M. (2012). "ASAMgpu V1.0 – a moist fully compressible atmospheric model using
    graphics processing units (GPUs)". Geoscientific Model Development,
    5, 345–353. https://doi.org/10.5194/gmd-5-345-2012
"""
function triangle_inequality_limiter(force, allowed_source_amount, limit_neg = 0)

    FT = eltype(force)

    if force < FT(0)
        return -triangle_inequality_limiter(-force, limit_neg, allowed_source_amount)
    end

    if allowed_source_amount >= FT(0)
        # If the intended source amount is positive, limit the force by the intended source amount.
        return force + allowed_source_amount - sqrt(force^2 + allowed_source_amount^2)
    elseif limit_neg > FT(0)
        # Due to numerics we might end up with negative numbers in the intended_source_amount.
        # This results in the tendecy switching signs. In that case we need to try to limit by the
        # amound available in the destination category.
        force1 = force + allowed_source_amount - sqrt(force^2 + allowed_source_amount^2)
        force2 = -force1 + limit_neg - sqrt(force1^2 + limit_neg^2)
        return -force2
    else
        # If both the source and destination are negatve, we set the force to zero
        # (There is nothing the limiter can do in this case.)
        return FT(0)
    end
end
