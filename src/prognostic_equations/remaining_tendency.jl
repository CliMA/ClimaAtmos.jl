
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

function prognostic_nt(::Val{names}; tends...) where {names}
    nt_ordered = NamedTuple{names}(rzero(values(tends)))
    nt_values = NamedTuple{keys(tends)}(values(tends))
    return merge(nt_ordered, nt_values)
end

function ᶜremaining_tendency(ᶜY, ᶠY, p, t)
    names = propertynames(ᶜY)
    tends = (;
        ᶜremaining_tendency_ρ(ᶜY, ᶠY, p, t)...,
        ᶜremaining_tendency_uₕ(ᶜY, ᶠY, p, t)...,
        ᶜremaining_tendency_ρe_tot(ᶜY, ᶠY, p, t)...,
        ᶜremaining_tendency_ρq_tot(ᶜY, ᶠY, p, t)...,
        ᶜremaining_tendency_ρq_liq(ᶜY, ᶠY, p, t)...,
        ᶜremaining_tendency_ρq_ice(ᶜY, ᶠY, p, t)...,
        ᶜremaining_tendency_ρq_rai(ᶜY, ᶠY, p, t)...,
        ᶜremaining_tendency_ρq_sno(ᶜY, ᶠY, p, t)...,
        ᶜremaining_tendency_sgs⁰(ᶜY, ᶠY, p, t)...,
        ᶜremaining_tendency_sgsʲs(ᶜY, ᶠY, p, t)...,
    )
    return lazy.(prognostic_nt.(Val(names); tends...))
end
function ᶠremaining_tendency(ᶜY, ᶠY, p, t)
    names = propertynames(ᶠY)
    tends = (;
        ᶠremaining_tendency_u₃(ᶜY, ᶠY, p, t)...,
        ᶠremaining_tendency_sgsʲs(ᶜY, ᶠY, p, t)...,
    )
    return lazy.(prognostic_nt.(Val(names); tends...))
end
using ClimaCore.RecursiveApply: rzero
function ᶜremaining_tendency_ρ(ᶜY, ᶠY, p, t)
    :ρ in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρ))
    return (; ρ = ∑tendencies)
end
function ᶜremaining_tendency_uₕ(ᶜY, ᶠY, p, t)
    :uₕ in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.uₕ))
    return (; uₕ = ∑tendencies)
end
function ᶜremaining_tendency_ρe_tot(ᶜY, ᶠY, p, t)
    :ρe_tot in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρe_tot))
    return (; ρe_tot = ∑tendencies)
end
function ᶜremaining_tendency_ρq_tot(ᶜY, ᶠY, p, t)
    :ρq_tot in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρq_tot))
    return (; ρq_tot = ∑tendencies)
end
function ᶜremaining_tendency_ρq_liq(ᶜY, ᶠY, p, t)
    :ρq_liq in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρq_liq))
    return (; ρq_liq = ∑tendencies)
end
function ᶜremaining_tendency_ρq_ice(ᶜY, ᶠY, p, t)
    :ρq_ice in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρq_ice))
    return (; ρq_ice = ∑tendencies)
end
function ᶜremaining_tendency_ρq_rai(ᶜY, ᶠY, p, t)
    :ρq_rai in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρq_rai))
    return (; ρq_rai = ∑tendencies)
end
function ᶜremaining_tendency_ρq_sno(ᶜY, ᶠY, p, t)
    :ρq_sno in propertynames(ᶜY) || return ()
    ∑tendencies = zero(eltype(ᶜY.ρq_sno))
    return (; ρq_sno = ∑tendencies)
end
function ᶜremaining_tendency_sgsʲs(ᶜY, ᶠY, p, t)
    :sgsʲs in propertynames(ᶜY) || return ()
    ∑tendencies = rzero(eltype(ᶜY.sgsʲs))
    return (; sgsʲs = ∑tendencies)
end
function ᶜremaining_tendency_sgs⁰(ᶜY, ᶠY, p, t)
    :sgs⁰ in propertynames(ᶜY) || return ()
    ∑tendencies = rzero(eltype(ᶜY.sgs⁰))
    return (; sgs⁰ = ∑tendencies)
end
function ᶠremaining_tendency_u₃(ᶜY, ᶠY, p, t)
    :u₃ in propertynames(ᶠY) || return ()
    ∑tendencies = zero(eltype(ᶠY.u₃))
    return (; u₃ = ∑tendencies)
end
function ᶠremaining_tendency_sgsʲs(ᶜY, ᶠY, p, t)
    :sgsʲs in propertynames(ᶠY) || return ()
    ∑tendencies = rzero(eltype(ᶠY.sgsʲs))
    return (; sgsʲs = ∑tendencies)
end


NVTX.@annotate function remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    device = ClimaComms.device(axes(Y.c))
    (localmem_lg, localmem_state) = if device isa ClimaComms.CUDADevice
        Val(false), Val(true)
    else
        Val(false), Val(false)
    end
    p_kernel = (;)
    if :sfc in propertynames(Y) # columnwise! does not yet handle .sfc
        parent(Yₜ.sfc) .= zero(Spaces.undertype(axes(Y.c)))
    end
    Operators.columnwise!(
        device,
        ᶜremaining_tendency,
        ᶠremaining_tendency,
        Yₜ.c,
        Yₜ.f,
        Y.c,
        Y.f,
        p_kernel,
        t,
        localmem_lg,
        localmem_state,
    )
    horizontal_tracer_advection_tendency!(Yₜ_lim, Y, p, t)
    fill_with_nans!(p)
    horizontal_advection_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    vertical_advection_of_water_tendency!(Yₜ, Y, p, t)
    additional_tendency!(Yₜ, Y, p, t)
    return Yₜ
end

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
    ᶠu₃ = Y.f.u₃
    ᶜρ = Y.c.ρ
    (; forcing_type, moisture_model, rayleigh_sponge, viscous_sponge) = p.atmos
    (; ls_adv, edmf_coriolis) = p.atmos
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜp, sfc_conditions, ᶜts) = p.precomputed

    vst_uₕ = viscous_sponge_tendency_uₕ(ᶜuₕ, viscous_sponge)
    vst_u₃ = viscous_sponge_tendency_u₃(ᶠu₃, viscous_sponge)
    vst_ρe_tot = viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, viscous_sponge)
    rst_uₕ = rayleigh_sponge_tendency_uₕ(ᶜuₕ, rayleigh_sponge)
    hs_args = (ᶜuₕ, ᶜp, params, sfc_conditions.ts, moisture_model, forcing_type)
    hs_tendency_uₕ = held_suarez_forcing_tendency_uₕ(hs_args...)
    hs_tendency_ρe_tot = held_suarez_forcing_tendency_ρe_tot(ᶜρ, hs_args...)
    edmf_cor_tend_uₕ = edmf_coriolis_tendency_uₕ(ᶜuₕ, edmf_coriolis)
    lsa_args = (ᶜρ, thermo_params, ᶜts, t, ls_adv)
    bc_lsa_tend_ρe_tot = large_scale_advection_tendency_ρe_tot(lsa_args...)

    # TODO: fuse, once we fix
    #       https://github.com/CliMA/ClimaCore.jl/issues/2165
    @. Yₜ.c.uₕ += vst_uₕ
    @. Yₜ.c.uₕ += rst_uₕ
    @. Yₜ.f.u₃.components.data.:1 += vst_u₃
    @. Yₜ.c.ρe_tot += vst_ρe_tot

    # TODO: can we write this out explicitly?
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        vst_tracer = viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, viscous_sponge)
        @. ᶜρχₜ += vst_tracer
        @. Yₜ.c.ρ += vst_tracer
    end

    # Held Suarez tendencies
    @. Yₜ.c.uₕ += hs_tendency_uₕ
    @. Yₜ.c.ρe_tot += hs_tendency_ρe_tot

    subsidence_tendency!(Yₜ, Y, p, t, p.atmos.subsidence)

    @. Yₜ.c.ρe_tot += bc_lsa_tend_ρe_tot
    if moisture_model isa AbstractMoistModel
        bc_lsa_tend_ρq_tot = large_scale_advection_tendency_ρq_tot(lsa_args...)
        @. Yₜ.c.ρq_tot += bc_lsa_tend_ρq_tot
    end

    @. Yₜ.c.uₕ += edmf_cor_tend_uₕ

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

    surface_flux_tendency!(Yₜ, Y, p, t)

    radiation_tendency!(Yₜ, Y, p, t, p.atmos.radiation_mode)
    if p.atmos.sgs_entr_detr_mode == Explicit()
        edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    if p.atmos.sgs_mf_mode == Explicit()
        edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    if p.atmos.sgs_nh_pressure_mode == Explicit()
        edmfx_nh_pressure_drag_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    edmfx_filter_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    edmfx_tke_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    if p.atmos.noneq_cloud_formation_mode == Explicit()
        cloud_condensate_tendency!(
            Yₜ,
            Y,
            p,
            p.atmos.moisture_model,
            p.atmos.precip_model,
        )
    end

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
end
