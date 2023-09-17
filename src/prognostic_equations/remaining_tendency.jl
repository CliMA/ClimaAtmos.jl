
NVTX.@annotate function remaining_tendency!(Yₜ, Y, p, t)
    fill_with_nans!(p)
    Yₜ .= zero(eltype(Yₜ))
    set_precomputed_quantities!(Y, p, t)
    NVTX.@range "horizontal" color = colorant"orange" begin
        horizontal_advection_tendency!(Yₜ, Y, p, t)
        hyperdiffusion_tendency!(Yₜ, Y, p, t)
    end
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    additional_tendency!(Yₜ, Y, p, t)
    dss!(Yₜ, p, t)
    return Yₜ
end

NVTX.@annotate function additional_tendency!(Yₜ, Y, p, t)
    viscous_sponge_tendency!(Yₜ, Y, p, t, p.atmos.viscous_sponge)
    surface_temp_tendency!(Yₜ, Y, p, t, p.atmos.surface_model)

    # Vertical tendencies
    Fields.bycolumn(axes(Y.c)) do colidx
        rayleigh_sponge_tendency!(Yₜ, Y, p, t, colidx, p.atmos.rayleigh_sponge)
        forcing_tendency!(Yₜ, Y, p, t, colidx, p.forcing_type)
        subsidence_tendency!(Yₜ, Y, p, t, colidx, p.subsidence)
        edmf_coriolis_tendency!(Yₜ, Y, p, t, colidx, p.edmf_coriolis)
        large_scale_advection_tendency!(Yₜ, Y, p, t, colidx, p.ls_adv)

        (; vert_diff) = p.atmos
        vertical_diffusion_boundary_layer_tendency!(
            Yₜ,
            Y,
            p,
            t,
            colidx,
            vert_diff,
        )

        radiation_tendency!(Yₜ, Y, p, t, colidx, p.radiation_model)
        edmfx_entr_detr_tendency!(Yₜ, Y, p, t, colidx, p.turbconv_model)
        edmfx_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, p.atmos.turbconv_model)
        edmfx_nh_pressure_tendency!(Yₜ, Y, p, t, colidx, p.turbconv_model)
        edmfx_tke_tendency!(Yₜ, Y, p, t, colidx, p.turbconv_model)
        explicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, p.turbconv_model)
        precipitation_tendency!(Yₜ, Y, p, t, colidx, p.precip_model)
        # NOTE: All ρa tendencies should be applied before calling this function
        pressure_work_tendency!(Yₜ, Y, p, t, colidx, p.atmos.turbconv_model)

        # NOTE: This will zero out all monmentum tendencies in the edmfx advection test
        # please DO NOT add additional velocity tendencies after this function
        zero_velocity_tendency!(Yₜ, Y, p, t, colidx)
    end
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
end
