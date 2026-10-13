NVTX.@annotate function subcol_model_callback!(integrator)
    Y = integrator.u
    p = integrator.p
    run_cosp_cloudsat!(Y, p, p.atmos.microphysics_model)

    return nothing
end

function default_model_callbacks(
    ::COSPModel;
    dt_subcol = "Inf",
    dt,
    t_start,
    t_end,
    checkpoint_frequency,
    kwargs...,
)
    time_to_seconds(dt_subcol) == Inf && return ()
    return scheduled_callback(
        subcol_model_callback!,
        dt_subcol,
        dt,
        t_start,
        t_end,
        checkpoint_frequency,
    )
end
