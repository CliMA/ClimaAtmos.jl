#####
##### Implicit tendencies
#####

import ClimaCore
import ClimaCore: Fields, Geometry

NVTX.@annotate function implicit_tendency!(Yₜ, Y, p, t)
    fill_with_nans!(p)
    Yₜ .= zero(eltype(Yₜ))
    implicit_vertical_advection_tendency!(Yₜ, Y, p, t)

    if p.atmos.microphysics_tendency_timestepping == Implicit()
        microphysics_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.microphysics_model,
            p.atmos.turbconv_model,
        )
        # Surface water/energy deposition from precipitation (implicit path).
        # The explicit counterpart is called from remaining_tendency!.
        surface_precipitation_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.surface.temperature,
            p.atmos.microphysics_model,
        )
    end

    edmfx_sgs_vertical_advection_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
    )

    if p.atmos.diff_mode == Implicit()
        vertical_diffusion_boundary_layer_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.vertical_diffusion,
        )
        edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    edmfx_vertical_diffusion_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    edmfx_boundary_condition_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    # NOTE: All ρa tendencies should be applied before calling this function
    pressure_work_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    sgs_u₃_implicit_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    sgs_ρa_implicit_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    # NOTE: This will zero out all momentum tendencies in the edmfx advection test
    # DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)

    return nothing
end

# TODO: All of these should use dtγ instead of dt, but dtγ is not available in
# the implicit tendency function. Since dt >= dtγ, we can safely use dt for now.

function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:none})
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠu³ * ᶠinterp(ᶜχ))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:first_order})
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠupwind1(ᶠu³, ᶜχ))))
end
@static if pkgversion(ClimaCore) ≥ v"0.14.22"
    function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:vanleer_limiter})
        ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
        ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
        return @. lazy(
            -(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠlin_vanleer(ᶠu³, ᶜχ, dt))),
        )
    end
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:third_order})
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠupwind3(ᶠu³, ᶜχ))))
end

vertical_advection(ᶠu³, ᶜχ, ::Val{:none}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))

function implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; microphysics_model, turbconv_model, rayleigh_sponge) = p.atmos
    (; params, dt) = p
    n = n_mass_flux_subdomains(turbconv_model)
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶠu³, ᶜp, ᶜh_tot, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)

    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠu³)

    # Central vertical advection of active tracers (e_tot and q_tot)
    vtt = vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    @. Yₜ.c.ρe_tot += vtt
    if !(microphysics_model isa DryModel)
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        vtt = vertical_transport(Y.c.ρ, ᶠu³, ᶜq_tot, dt, Val(:none))
        @. Yₜ.c.ρq_tot += vtt
    end

    # Vertical advection of passive tracers with the mean flow
    # is done in the explicit tendency.
    # Here we add the vertical advection with precipitation terminal velocity
    # using downward biasing and free outflow bottom boundary condition
    if microphysics_model isa NonEquilibriumMicrophysics
        (; ᶜwₗ, ᶜwᵢ) = p.precomputed
        @. Yₜ.c.ρq_lcl -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                Geometry.WVector(-(ᶜwₗ)) * specific(Y.c.ρq_lcl, Y.c.ρ),
            ),
        )
        @. Yₜ.c.ρq_icl -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                Geometry.WVector(-(ᶜwᵢ)) * specific(Y.c.ρq_icl, Y.c.ρ),
            ),
        )
    end
    if microphysics_model isa
       NonEquilibriumMicrophysics1M
        (; ᶜwᵣ, ᶜwₛ) = p.precomputed
        @. Yₜ.c.ρq_rai -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                Geometry.WVector(-(ᶜwᵣ)) * specific(Y.c.ρq_rai, Y.c.ρ),
            ),
        )
        @. Yₜ.c.ρq_sno -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                Geometry.WVector(-(ᶜwₛ)) * specific(Y.c.ρq_sno, Y.c.ρ),
            ),
        )
    end
    if microphysics_model isa
       NonEquilibriumMicrophysics2M
        (; ᶜwₙₗ, ᶜwₙᵣ, ᶜwᵣ, ᶜwₛ) = p.precomputed
        @. Yₜ.c.ρn_lcl -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                Geometry.WVector(-(ᶜwₙₗ)) * specific(Y.c.ρn_lcl, Y.c.ρ),
            ),
        )
        @. Yₜ.c.ρn_rai -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                Geometry.WVector(-(ᶜwₙᵣ)) * specific(Y.c.ρn_rai, Y.c.ρ),
            ),
        )
        @. Yₜ.c.ρq_rai -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                Geometry.WVector(-(ᶜwᵣ)) * specific(Y.c.ρq_rai, Y.c.ρ),
            ),
        )
        @. Yₜ.c.ρq_sno -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                Geometry.WVector(-(ᶜwₛ)) * specific(Y.c.ρq_sno, Y.c.ρ),
            ),
        )
    end
    if microphysics_model isa NonEquilibriumMicrophysics2MP3
        (; ρ, ρn_ice, ρq_rim, ρb_rim) = Y.c
        ᶜwnᵢ = @. lazy(Geometry.WVector(p.precomputed.ᶜwnᵢ))
        ᶜwᵢ = @. lazy(Geometry.WVector(p.precomputed.ᶜwᵢ))
        ᶠρ = @. lazy(ᶠinterp(ρ * ᶜJ) / ᶠJ)

        # Note: `ρq_icl` is handled above, in `microphysics_model isa NonEquilibriumMicrophysics`
        @. Yₜ.c.ρn_ice -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(- ᶜwnᵢ * specific(ρn_ice, ρ)))
        @. Yₜ.c.ρq_rim -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(- ᶜwᵢ * specific(ρq_rim, ρ)))
        @. Yₜ.c.ρb_rim -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(- ᶜwᵢ * specific(ρb_rim, ρ)))
    end

    vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    # This is equivalent to grad_v(Φ) + grad_v(p) / ρ
    ᶜΦ_r = @. lazy(phi_r(thermo_params, ᶜp))
    ᶜθ_v = p.scratch.ᶜtemp_scalar
    @. ᶜθ_v = theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
    ᶜθ_vr = @. lazy(theta_vr(thermo_params, ᶜp))
    ᶜΠ = @. lazy(TD.exner_given_pressure(thermo_params, ᶜp))
    @. Yₜ.f.u₃ -= ᶠgradᵥ_ᶜΦ - ᶠgradᵥ(ᶜΦ_r) +
                  cp_d * (ᶠinterp(ᶜθ_v - ᶜθ_vr)) * ᶠgradᵥ(ᶜΠ)

    rst_u₃ = rayleigh_sponge_tendency_u₃(Y.f.u₃, rayleigh_sponge)
    @. Yₜ.f.u₃ += rst_u₃
    return nothing
end
