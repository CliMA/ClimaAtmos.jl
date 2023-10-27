NVTX.@annotate function limited_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)
    tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    return nothing
end

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
