"""
    limiters_func!(Y, p, t, ref_Y)

Apply tracer limiters to the prognostic state `Y` in place, using parameters `p`,
time `t`, and a reference state `ref_Y` for bounds.

This function iterates over tracer variables in `Y.c` and applies the limiter
if one is specified in `p.numerics`.
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