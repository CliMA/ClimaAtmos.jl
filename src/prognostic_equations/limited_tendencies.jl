"""
    limiters_func!(Y, p, t, ref_Y)

Applies tracer limiters to the prognostic state `Y` in place, using parameters `p`,
time `t`, and a reference state `ref_Y` for bounds determination.

This function iterates over all identified tracer variables within `Y.c`. If a limiter
is configured in `p.numerics.limiter`, it first calls `ClimaCore.Limiters.compute_bounds!`
to establish the permissible range for each tracer (based on `ref_Y.c.:(ρχ_name)`
and `ref_Y.c.ρ`). Subsequently, `ClimaCore.Limiters.apply_limiter!` is called to modify
`Y.c.:(ρχ_name)` (and implicitly `Y.c.ρ` if the limiter adjusts both to maintain the
specific quantity `ρχ/ρ` within bounds or if it only modifies `ρχ` based on `ρ`),
ensuring the tracer quantities adhere to these bounds.

Arguments:
- `Y`: The current state vector (`ClimaCore.Fields.FieldVector`), modified in place.
- `p`: A cache or parameters object, containing `p.numerics.limiter`.
- `t`: The current simulation time (often unused by the limiter itself but part of a standard signature).
- `ref_Y`: A reference state vector (`ClimaCore.Fields.FieldVector`) used to compute the
           tracer bounds (e.g., ensuring positivity or monotonicity relative to this state).
"""
NVTX.@annotate function limiters_func!(Y, p, t, ref_Y)
    (; limiter) = p.numerics
    if !isnothing(limiter)
        for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
            Limiters.compute_bounds!(limiter, ref_Y.c.:($ρχ_name), ref_Y.c.ρ)
            Limiters.apply_limiter!(Y.c.:($ρχ_name), Y.c.ρ, limiter)
        end
    end
    return nothing
end

"""
    triangle_inequality_limiter(force, limit)

Limits a `force` (or source term) based on a maximum allowable `limit` using a
formula derived from the triangle inequality, as proposed by Horn (2012).

The formula used is: `L = F + M - sqrt(F² + M²)`, where `F` is the `force` and
`M` is the `limit`.

This limiter is designed to smoothly reduce the `force` as it approaches or
exceeds the `limit`, ensuring the result `L` satisfies `0 ≤ L ≤ M` if `F ≥ 0`
and `M > 0`. It also preserves `L ≤ F`. It's particularly useful for ensuring
that source terms (e.g., emissions, chemical production rates) do not become
unphysically large or lead to numerical instability, while being continuously
differentiable.

Arguments:
- `force`: The original force or source term value.
- `limit`: The maximum permissible positive value for the limited force.

Returns:
- The limited force value.

Reference:
- Horn, M. (2012). "ASAMgpu V1.0 – a moist fully compressible atmospheric model using
    graphics processing units (GPUs)". Geoscientific Model Development,
    5, 345–353. https://doi.org/10.5194/gmd-5-345-2012
"""

function triangle_inequality_limiter(force, limit)
    FT = eltype(force)
    return force == FT(0) ? force : force + limit - sqrt(force^2 + limit^2)
end
