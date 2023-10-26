NVTX.@annotate function limited_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)
    tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    return nothing
end

function generate_limiters_func!(limiter::Nothing)
    limiters_func! = (Y, p, t, ref_Y) -> nothing
    return limiters_func!
end

function generate_limiters_func!(limiter)
    NVTX.@annotate function limiters_func!(Y, p, t, ref_Y)
        for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
            Limiters.compute_bounds!(limiter, ref_Y.c.:($ρχ_name), ref_Y.c.ρ)
            Limiters.apply_limiter!(Y.c.:($ρχ_name), Y.c.ρ, limiter)
        end
        return nothing
    end

    return limiters_func!
end
