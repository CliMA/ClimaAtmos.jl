
NVTX.@annotate function hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    prep_tracer_hyperdiffusion_tendency!(Yₜ_lim, Y, p, t)
    prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    if do_dss(axes(Y.c)) && !isnothing(p.atmos.hyperdiff)
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
    questionable_vertical_advection_tendency!(Yₜ, Y, p, t)
    additional_tendency!(Yₜ, Y, p, t)
    return Yₜ
end

import LazyBroadcast: @lazy
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

function z_coordinate_fields(space::Spaces.AbstractSpace)
    ᶜz = Fields.coordinate_field(Spaces.center_space(space)).z
    ᶠz = Fields.coordinate_field(Spaces.face_space(space)).z
    return (; ᶜz, ᶠz)
end


NVTX.@annotate function additional_tendency!(Yₜ, Y, p, t)

    (; ᶜh_tot, ᶜspecific) = p.precomputed
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Yₜ.f.u₃
    ᶜρ = Y.c.ρ
    (; forcing_type, moisture_model, rayleigh_sponge, viscous_sponge) = p.atmos
    (; edmf_coriolis) = p.atmos
    (; params) = p
    (; ᶜp, sfc_conditions) = p.precomputed

    vst_uₕ = viscous_sponge_tendency_uₕ(ᶜuₕ, viscous_sponge)
    vst_u₃ = viscous_sponge_tendency_u₃(ᶠu₃, viscous_sponge)
    vst_ρe_tot = viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, viscous_sponge)
    rst_uₕ = rayleigh_sponge_tendency_uₕ(ᶜuₕ, rayleigh_sponge)
    hs_args = (ᶜuₕ, ᶜp, params, sfc_conditions.ts, moisture_model, forcing_type)
    hs_tendency_uₕ = held_suarez_forcing_tendency_uₕ(hs_args...)
    hs_tendency_ρe_tot = held_suarez_forcing_tendency_ρe_tot(ᶜρ, hs_args...)
    edmf_cor_tend_uₕ = edmf_coriolis_tendency_uₕ(ᶜuₕ, edmf_coriolis)

    # TODO: fuse, once we fix
    #       https://github.com/CliMA/ClimaCore.jl/issues/2165
    @. Yₜ.c.uₕ += vst_uₕ
    @. Yₜ.c.uₕ += rst_uₕ
    @. Yₜ.f.u₃.components.data.:1 += vst_u₃
    @. Yₜ.c.ρe_tot += vst_ρe_tot

    # TODO: can we write this out explicitly?
    if viscous_sponge isa ViscousSponge
        for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
            χ_name == :e_tot && continue
            vst_tracer = viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, viscous_sponge)
            @. ᶜρχₜ += vst_tracer
            @. Yₜ.c.ρ += vst_tracer
        end
    end

    # Held Suarez tendencies
    @. Yₜ.c.uₕ += hs_tendency_uₕ
    @. Yₜ.c.ρe_tot += hs_tendency_ρe_tot

    subsidence_tendency!(Yₜ, Y, p, t, p.atmos.subsidence)

    @. Yₜ.c.uₕ += edmf_cor_tend_uₕ

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
    cloud_condensate_tendency!(
        Yₜ,
        p,
        p.atmos.moisture_model,
        p.atmos.precip_model,
    )
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
        p.atmos.moisture_model,
        p.atmos.precip_model,
        p.atmos.turbconv_model,
    )

    # NOTE: Precipitation tendencies should be applied before calling this function,
    # because precipitation cache is used in this function
    surface_temp_tendency!(Yₜ, Y, p, t, p.atmos.surface_model)

    # NOTE: All ρa tendencies should be applied before calling this function
    pressure_work_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    sl = p.atmos.smagorinsky_lilly
    horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl)
    vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl)

    # NOTE: This will zero out all momentum tendencies in the edmfx advection test
    # please DO NOT add additional velocity tendencies after this function
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
