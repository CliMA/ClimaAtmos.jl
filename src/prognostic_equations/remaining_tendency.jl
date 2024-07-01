
NVTX.@annotate function hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    prep_tracer_hyperdiffusion_tendency!(Yₜ_lim, Y, p, t)
    prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    if p.do_dss && !isnothing(p.atmos.hyperdiff)
        pairs = dss_hyperdiffusion_tendency_pairs(p)
        Spaces.weighted_dss!(pairs...)
    end
    apply_tracer_hyperdiffusion_tendency!(Yₜ_lim, Y, p, t)
    apply_hyperdiffusion_tendency!(Yₜ, Y, p, t)
end

NVTX.@annotate function remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    Yₜ .= zero(eltype(Yₜ))
    horizontal_tracer_advection_tendency!(Yₜ_lim, Y, p, t)
    fill_with_nans!(p)
    horizontal_advection_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    additional_tendency!(Yₜ, Y, p, t)
    return Yₜ
end

NVTX.@annotate function additional_tendency!(Yₜ, Y, p, t)
    viscous_sponge_tendency!(Yₜ, Y, p, t, p.atmos.viscous_sponge)

    # Vertical tendencies
    rayleigh_sponge_tendency!(Yₜ, Y, p, t, p.atmos.rayleigh_sponge)
    forcing_tendency!(Yₜ, Y, p, t, p.atmos.forcing_type)
    subsidence_tendency!(Yₜ, Y, p, t, p.atmos.subsidence)
    edmf_coriolis_tendency!(Yₜ, Y, p, t, p.atmos.edmf_coriolis)
    large_scale_advection_tendency!(Yₜ, Y, p, t, p.atmos.ls_adv)
    external_forcing_tendency!(Yₜ, Y, p, t, p.atmos.external_forcing)

    if p.atmos.sgs_adv_mode == Explicit()
        edmfx_sgs_vertical_advection_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.turbconv_model,
        )
    end

    if p.atmos.diff_mode == Explicit()
        vertical_diffusion_boundary_layer_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.vert_diff,
        )
        edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    radiation_tendency!(Yₜ, Y, p, t, p.atmos.radiation_mode)
    edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    edmfx_nh_pressure_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    edmfx_filter_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    edmfx_tke_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    # Non-equilibrium cloud formation
    cloud_condensate_tendency!(Yₜ, p, p.atmos.moisture_model)
    edmfx_precipitation_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
        p.atmos.precip_model,
    )
    precipitation_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.precip_model,
        p.atmos.turbconv_model,
    )

    # NOTE: Precipitation tendencies should be applied before calling this function,
    # because precipitation cache is used in this function
    surface_temp_tendency!(Yₜ, Y, p, t, p.atmos.surface_model)

    # NOTE: All ρa tendencies should be applied before calling this function
    pressure_work_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    
    horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, p.atmos.smagorinsky_lilly)
    vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, p.atmos.smagorinsky_lilly)

    # NOTE: This will zero out all momentum tendencies in the edmfx advection test
    # please DO NOT add additional velocity tendencies after this function
    # Other SGS tendencies
    zero_velocity_tendency!(Yₜ, Y, p, t)

    # TODO: make bycolumn-able
    non_orographic_gravity_wave_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.non_orographic_gravity_wave,
    )
    orographic_gravity_wave_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.orographic_gravity_wave,
    )
    # NOTE: This will zero out all tendencies
    # please DO NOT add additional tendencies after this function
    zero_tendency!(Yₜ, Y, p, t, p.atmos.tendency_model, p.atmos.turbconv_model)
end
