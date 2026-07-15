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
    ᶠρ = ᶠface_density(ᶜρ)
    return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠu³ * ᶠinterp(ᶜχ))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:first_order})
    ᶠρ = ᶠface_density(ᶜρ)
    return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠupwind1(ᶠu³, ᶜχ))))
end
@static if pkgversion(ClimaCore) ≥ v"0.14.22"
    function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:vanleer_limiter})
        ᶠρ = ᶠface_density(ᶜρ)
        return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠlin_vanleer(ᶠu³, ᶜχ, dt))))
    end
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:third_order})
    ᶠρ = ᶠface_density(ᶜρ)
    return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠupwind3(ᶠu³, ᶜχ))))
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
    ᶜρ, ᶠρ = Y.c.ρ, ᶠface_density(Y.c.ρ)
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶠu³, ᶜp, ᶜh_tot, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)

    # Mass advection with zero flux through the top and bottom
    # boundaries (ᶜadvdivᵥ). The state filter in
    # set_implicit_precomputed_quantities! also sets ᶠu³ to 0 at both
    # boundaries, and the ρ row of the manual Jacobian is built from
    # ᶜadvdivᵥ_matrix(), so using ᶜadvdivᵥ here keeps the residual, the
    # boundary conditions, and the Jacobian consistent.
    @. Yₜ.c.ρ -= ᶜadvdivᵥ(ᶠρ * ᶠu³)

    # Central vertical advection of active tracers (ρe_tot and ρq_tot).
    # The upwind correction is applied post-Newton via `T_post_imp!`
    # (see `correct_implicit_advection_tendency!`), so that the upwind
    # direction is taken with respect to the Newton-solved velocity rather
    # than the initial guess.
    vtt = vertical_transport(ᶜρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    @. Yₜ.c.ρe_tot += vtt
    if !(microphysics_model isa DryModel)
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, ᶜρ))
        vtt = vertical_transport(ᶜρ, ᶠu³, ᶜq_tot, dt, Val(:none))
        @. Yₜ.c.ρq_tot += vtt
    end

    # Vertical advection of passive tracers with the mean flow
    # is done in the explicit tendency.
    # Here we add the vertical advection with precipitation terminal velocity
    # using downward biasing and free outflow bottom boundary condition
    if microphysics_model isa NonEquilibriumMicrophysics
        (; ᶜwₗ, ᶜwᵢ) = p.precomputed
        @. Yₜ.c.ρq_lcl -=
            ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(WVec(-(ᶜwₗ)) * specific(Y.c.ρq_lcl, ᶜρ)))
        @. Yₜ.c.ρq_icl -=
            ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(WVec(-(ᶜwᵢ)) * specific(Y.c.ρq_icl, ᶜρ)))
    end
    if microphysics_model isa
       NonEquilibriumMicrophysics1M
        (; ᶜwᵣ, ᶜwₛ) = p.precomputed
        @. Yₜ.c.ρq_rai -=
            ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(WVec(-(ᶜwᵣ)) * specific(Y.c.ρq_rai, ᶜρ)))
        @. Yₜ.c.ρq_sno -=
            ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(WVec(-(ᶜwₛ)) * specific(Y.c.ρq_sno, ᶜρ)))
    end
    if microphysics_model isa
       NonEquilibriumMicrophysics2M
        (; ᶜwₙₗ, ᶜwₙᵣ, ᶜwᵣ, ᶜwₛ) = p.precomputed
        @. Yₜ.c.ρn_lcl -=
            ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(WVec(-(ᶜwₙₗ)) * specific(Y.c.ρn_lcl, ᶜρ)))
        @. Yₜ.c.ρn_rai -=
            ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(WVec(-(ᶜwₙᵣ)) * specific(Y.c.ρn_rai, ᶜρ)))
        @. Yₜ.c.ρq_rai -=
            ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(WVec(-(ᶜwᵣ)) * specific(Y.c.ρq_rai, ᶜρ)))
        @. Yₜ.c.ρq_sno -=
            ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(WVec(-(ᶜwₛ)) * specific(Y.c.ρq_sno, ᶜρ)))
    end
    if microphysics_model isa NonEquilibriumMicrophysics2MP3
        (; ρn_ice, ρq_rim, ρb_rim) = Y.c
        ᶜwnᵢ = @. lazy(WVec(p.precomputed.ᶜwnᵢ))
        ᶜwᵢ = @. lazy(WVec(p.precomputed.ᶜwᵢ))

        # Note: `ρq_icl` is handled above, in `microphysics_model isa NonEquilibriumMicrophysics`
        @. Yₜ.c.ρn_ice -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(- ᶜwnᵢ * specific(ρn_ice, ᶜρ)))
        @. Yₜ.c.ρq_rim -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(- ᶜwᵢ * specific(ρq_rim, ᶜρ)))
        @. Yₜ.c.ρb_rim -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(- ᶜwᵢ * specific(ρb_rim, ᶜρ)))
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

"""
    correct_implicit_advection_tendency!(Yₜ, Y, p, t)

Post-Newton upwind correction to the central-differenced implicit vertical
advection of `ρe_tot` and `ρq_tot` in
[`implicit_vertical_advection_tendency!`](@ref). Called by ClimaTimeSteppers
as the `T_post_imp!` hook on `ClimaODEFunction`: evaluated at the
Newton-solved stage state `U*` and applied as `U ← U* + dtγ · Yₜ`.

Writes `vtt_upwind - vtt_central` for `ρe_tot` (and `ρq_tot` when
available). All other fields of `Yₜ` are zero.

Evaluating the correction *after* Newton — rather than folding it into
the implicit tendency — means the upwind direction is taken with respect
to the Newton-solved velocity, avoiding the "wrong-cell" upwinding that
occurs when the sign of `ᶠu³` flips between the initial guess and the
Newton solution (a real concern with `max_iters = 1`).
"""
NVTX.@annotate function correct_implicit_advection_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    (; microphysics_model) = p.atmos
    (; energy_q_tot_upwinding) = p.atmos.numerics
    (; dt) = p
    (; ᶠu³, ᶜh_tot) = p.precomputed

    vtt_up = vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, energy_q_tot_upwinding)
    vtt_c = vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    @. Yₜ.c.ρe_tot = vtt_up - vtt_c
    if !(microphysics_model isa DryModel)
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        vtt_up = vertical_transport(Y.c.ρ, ᶠu³, ᶜq_tot, dt, energy_q_tot_upwinding)
        vtt_c = vertical_transport(Y.c.ρ, ᶠu³, ᶜq_tot, dt, Val(:none))
        @. Yₜ.c.ρq_tot = vtt_up - vtt_c
    end
    return nothing
end
