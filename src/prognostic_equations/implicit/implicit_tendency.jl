#####
##### Implicit tendencies
#####

import ClimaCore
import ClimaCore: Fields, Geometry

NVTX.@annotate function implicit_tendency!(Yₜ, Y, p, t)
    fill_with_nans!(p)
    Yₜ .= zero(eltype(Yₜ))
    implicit_vertical_advection_tendency!(Yₜ, Y, p, t)

    if p.atmos.noneq_cloud_formation_mode == Implicit()
        cloud_condensate_tendency!(
            Yₜ,
            Y,
            p,
            p.atmos.moisture_model,
            p.atmos.microphysics_model,
            p.atmos.turbconv_model,
        )
    end

    if p.atmos.sgs_adv_mode == Implicit()
        edmfx_sgs_vertical_advection_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.turbconv_model,
        )
    end

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


    if p.atmos.sgs_entr_detr_mode == Implicit()
        edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    if p.atmos.sgs_mf_mode == Implicit()
        edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    if p.atmos.sgs_nh_pressure_mode == Implicit()
        edmfx_nh_pressure_drag_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    if p.atmos.sgs_vertdiff_mode == Implicit()
        edmfx_vertical_diffusion_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    # NOTE: All ρa tendencies should be applied before calling this function
    pressure_work_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    # NOTE: This will zero out all momentum tendencies in the edmfx advection test
    # DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)

    return nothing
end

# TODO: All of these should use dtγ instead of dt, but dtγ is not available in
# the implicit tendency function. Since dt >= dtγ, we can safely use dt for now.
# TODO: Can we rewrite ᶠfct_boris_book and ᶠfct_zalesak so that their broadcast
# expressions are less convoluted?

function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:none})
    ᶠρ = face_density(ᶜρ)
    return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠu³ * ᶠinterp(ᶜχ))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:first_order})
    ᶠρ = face_density(ᶜρ)
    return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠupwind1(ᶠu³, ᶜχ))))
end
@static if pkgversion(ClimaCore) ≥ v"0.14.22"
    function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:vanleer_limiter})
        ᶠρ = face_density(ᶜρ)
        return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠlin_vanleer(ᶠu³, ᶜχ, dt))))
    end
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:third_order})
    ᶠρ = face_density(ᶜρ)
    return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠupwind3(ᶠu³, ᶜχ))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:boris_book})
    ᶠρ = face_density(ᶜρ)
    ᶠu³χ = @. lazy(ᶠupwind1(ᶠu³, ᶜχ))
    return @. lazy(
        -(ᶜadvdivᵥ(
            ᶠρ * (
                ᶠu³χ + ᶠfct_boris_book(
                    ᶠupwind3(ᶠu³, ᶜχ) - ᶠu³χ, ᶜχ / dt - ᶜadvdivᵥ(ᶠρ * ᶠu³χ) / ᶜρ,
                )
            ),
        )),
    )
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:zalesak})
    ᶠρ = face_density(ᶜρ)
    ᶠu³χ = @. lazy(ᶠupwind1(ᶠu³, ᶜχ))
    return @. lazy(
        -(ᶜadvdivᵥ(
            ᶠρ * (
                ᶠu³χ + ᶠfct_zalesak(
                    ᶠupwind3(ᶠu³, ᶜχ) - ᶠu³χ, ᶜχ / dt, ᶜχ / dt - ᶜadvdivᵥ(ᶠρ * ᶠu³χ) / ᶜρ,
                )
            ),
        )),
    )
end
function vertical_transport_sedimentation(ᶜρ, ᶜw, ᶜχ)
    ᶠρ = face_density(ᶜρ)
    return @. lazy(-(ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(WVec(-(ᶜw)) * ᶜχ))))
end

vertical_advection(ᶠu³, ᶜχ, ::Val{:none}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))

function implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; moisture_model, turbconv_model, rayleigh_sponge, microphysics_model) = p.atmos
    (; params, dt) = p
    n = n_mass_flux_subdomains(turbconv_model)
    ᶜρ = Y.c.ρ
    ᶠρ = face_density(ᶜρ)
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶠu³, ᶜp, ᶜts) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)
    e_tot = @. lazy(specific(Y.c.ρe_tot, ᶜρ))
    ᶜh_tot = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, e_tot))

    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠρ * ᶠu³)

    # Central vertical advection of active tracers (e_tot and q_tot)
    vtt = vertical_transport(ᶜρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    @. Yₜ.c.ρe_tot += vtt
    if !(moisture_model isa DryModel)
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, ᶜρ))
        vtt = vertical_transport(ᶜρ, ᶠu³, ᶜq_tot, dt, Val(:none))
        @. Yₜ.c.ρq_tot += vtt
    end

    # Vertical advection of passive tracers with the mean flow
    # is done in the explicit tendency.
    # Here we add the vertical advection with precipitation terminal velocity
    # using downward biasing and free outflow bottom boundary condition
    if moisture_model isa NonEquilMoistModel
        (; ᶜwₗ, ᶜwᵢ) = p.precomputed
        q_liq = @. lazy(specific(Y.c.ρq_liq, ᶜρ))
        q_ice = @. lazy(specific(Y.c.ρq_ice, ᶜρ))
        @. Yₜ.c.ρq_liq -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwₗ) * q_liq))
        @. Yₜ.c.ρq_ice -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwᵢ) * q_ice))
    end
    if microphysics_model isa Microphysics1Moment
        (; ᶜwᵣ, ᶜwₛ) = p.precomputed
        q_rai = @. lazy(specific(Y.c.ρq_rai, ᶜρ))
        q_sno = @. lazy(specific(Y.c.ρq_sno, ᶜρ))
        @. Yₜ.c.ρq_rai -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwᵣ) * q_rai))
        @. Yₜ.c.ρq_sno -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwₛ) * q_sno))
    end
    if microphysics_model isa Microphysics2Moment
        (; ᶜwₙₗ, ᶜwₙᵣ, ᶜwᵣ, ᶜwₛ) = p.precomputed
        n_liq = @. lazy(specific(Y.c.ρn_liq, ᶜρ))
        n_rai = @. lazy(specific(Y.c.ρn_rai, ᶜρ))
        q_rai = @. lazy(specific(Y.c.ρq_rai, ᶜρ))
        q_sno = @. lazy(specific(Y.c.ρq_sno, ᶜρ))
        @. Yₜ.c.ρn_liq -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwₙₗ) * n_liq))
        @. Yₜ.c.ρn_rai -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwₙᵣ) * n_rai))
        @. Yₜ.c.ρq_rai -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwᵣ) * q_rai))
        @. Yₜ.c.ρq_sno -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwₛ) * q_sno))
    end
    if microphysics_model isa Microphysics2MomentP3
        (; ᶜwnᵢ, ᶜwᵢ) = p.precomputed
        ᶜρ = Y.c.ρ
        ᶠρ = face_density(ᶜρ)

        # Note: `ρq_ice` is handled above, in `moisture_model isa NonEquilMoistModel`
        n_ice = @. lazy(specific(Y.c.ρn_ice, ᶜρ))
        q_rim = @. lazy(specific(Y.c.ρq_rim, ᶜρ))
        b_rim = @. lazy(specific(Y.c.ρb_rim, ᶜρ))
        @. Yₜ.c.ρn_ice -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwnᵢ) * n_ice))
        @. Yₜ.c.ρq_rim -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwᵢ) * q_rim))
        @. Yₜ.c.ρb_rim -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-WVec(ᶜwᵢ) * b_rim))
    end

    # TODO - decide if this needs to be explicit or implicit
    #vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    # This is equivalent to grad_v(Φ) + grad_v(p) / ρ
    ᶜΦ_r = @. lazy(phi_r(thermo_params, ᶜts))
    ᶜθ_v = @. lazy(theta_v(thermo_params, ᶜts))
    ᶜθ_vr = @. lazy(theta_vr(thermo_params, ᶜts))
    ᶜΠ = @. lazy(dry_exner_function(thermo_params, ᶜts))
    @. Yₜ.f.u₃ -= ᶠgradᵥ_ᶜΦ - ᶠgradᵥ(ᶜΦ_r) + cp_d * (ᶠinterp(ᶜθ_v - ᶜθ_vr)) * ᶠgradᵥ(ᶜΠ)

    if rayleigh_sponge isa RayleighSponge
        ᶠz = Fields.coordinate_field(Y.f).z
        zmax = z_max(axes(Y.f))
        rs = rayleigh_sponge
        @. Yₜ.f.u₃ -= β_rayleigh_w(rs, ᶠz, zmax) * Y.f.u₃
        if turbconv_model isa PrognosticEDMFX
            for j in 1:n
                @. Yₜ.f.sgsʲs.:($$j).u₃ -= β_rayleigh_w(rs, ᶠz, zmax) * Y.f.sgsʲs.:($$j).u₃
            end
        end
    end
    return nothing
end
