function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    zero_moment_microphysics_tendency!(Yₜ, Y, p, t)
    rrtmgp_model_tendency!(Yₜ, Y, p, t)
end
additional_callbacks = (PeriodicCallback(
    rrtmgp_model_callback!,
    FT(6 * 60 * 60); # update RRTMGPModel every 6 hours
    initial_affect = true,
),)
