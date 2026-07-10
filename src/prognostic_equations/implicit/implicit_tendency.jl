#####
##### Implicit tendencies
#####

import ClimaCore
import ClimaCore: Fields, Geometry

"""
    implicit_tendency!(Yₜ, Y, p, t)

Compute the implicitly treated tendency of the state `Y` at time `t`,
overwriting `Yₜ`.

This is the stiff part of the IMEX splitting, containing the terms that are
fast relative to the timestep: vertical advection and the vertical momentum
equation (`implicit_vertical_advection_tendency!`), EDMFX subdomain vertical
advection, entrainment/detrainment, mass flux, boundary conditions, pressure
work, and the cached updraft `u₃`/`ρa` stage tendencies. Vertical diffusion is
included only when `p.atmos.diff_mode == Implicit()`, and microphysics sources
(plus surface precipitation deposition) only when
`p.atmos.microphysics_tendency_timestepping == Implicit()`. All non-EDMFX
configurations dispatch the SGS tendencies to no-ops.

The linearization of this tendency is the Jacobian ``∂Yₜ/∂Y`` computed by the
[`JacobianAlgorithm`](@ref) subtypes; the residual it enters contains only the
central (`Val(:none)`) vertical transport of `ρe_tot` and `ρq_tot` — the
upwind–central difference is applied after the Newton solve by
`correct_implicit_advection_tendency!`.

Reads the precomputed quantities set by `set_implicit_precomputed_quantities!`
(`ᶠu³`, `ᶜh_tot`, sedimentation and precipitation terminal velocities, and the
EDMFX subdomain states). Returns `nothing`.

See [Implicit Solver](@ref) for the IMEX formulation.
"""
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
        vertical_smagorinsky_lilly_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.smagorinsky_lilly,
        )
        edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

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

"""
    vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, upwinding)

Return a lazy center-space broadcast of the flux-form vertical transport
tendency `-∂ᵥ(ρ u³ χ)`, with zero-flux boundary conditions imposed by
`ᶜadvdivᵥ`.

# Arguments

  - `ᶜρ`: Center-space density [kg/m³].
  - `ᶠu³`: Face-space contravariant vertical velocity [1/s].
  - `ᶜχ`: Center-space specific (per-mass) transported quantity.
  - `dt`: Timestep, used only by the van Leer limiter [s].
  - `upwinding`: Reconstruction of `χ` on faces; one of `Val(:none)` (central
    interpolation), `Val(:first_order)`, `Val(:third_order)`, or
    `Val(:vanleer_limiter)`.
"""
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:none})
    ᶠρ = ᶠface_density(ᶜρ)
    return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠu³ * ᶠinterp(ᶜχ))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:first_order})
    ᶠρ = ᶠface_density(ᶜρ)
    return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠupwind1(ᶠu³, ᶜχ))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:vanleer_limiter})
    ᶠρ = ᶠface_density(ᶜρ)
    return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠlin_vanleer(ᶠu³, ᶜχ, dt))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:third_order})
    ᶠρ = ᶠface_density(ᶜρ)
    return @. lazy(-(ᶜadvdivᵥ(ᶠρ * ᶠupwind3(ᶠu³, ᶜχ))))
end

"""
    vertical_advection(ᶠu³, ᶜχ, upwinding)

Return a lazy center-space broadcast of the advective-form vertical advection
tendency `-u³ ∂ᵥχ`, computed as the difference `-(∂ᵥ(u³ χ) - χ ∂ᵥu³)` so that
the same divergence operator (and reconstruction) is used as in
`vertical_transport`. `upwinding` is `Val(:none)`, `Val(:first_order)`,
or `Val(:third_order)`.
"""
vertical_advection(ᶠu³, ᶜχ, ::Val{:none}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))

"""
    implicit_vertical_advection_tendency!(Yₜ, Y, p, t)

Add the implicit vertical-advection and vertical-momentum tendencies to `Yₜ`.

Adds, in order:

  - The mass flux divergence `-∂ᵥ(ρ u³)` to `Yₜ.c.ρ`, with zero flux through the
    top and bottom boundaries (consistent with the state filter that zeroes `ᶠu³`
    there and with the `ᶜadvdivᵥ_matrix()` used in the manual Jacobian).
  - Central (`Val(:none)`) vertical transport of `h_tot` and `q_tot` to
    `Yₜ.c.ρe_tot` and `Yₜ.c.ρq_tot`; the upwind correction is applied post-Newton
    by `correct_implicit_advection_tendency!`.
  - Vertical transport of the non-equilibrium microphysics tracers with their
    terminal velocities, using downward (`ᶠtop_bias`) biasing and a free-outflow
    bottom boundary (`ᶜprecipdivᵥ`), plus the water sedimentation contributions to
    `ρ`, `ρe_tot`, and `ρq_tot` from `vertical_advection_of_water_tendency!`.
  - The Exner-form pressure gradient and buoyancy tendency
    `-(∂ᵥΦ - ∂ᵥΦ_r + cp_d (θ_v - θ_vr) ∂ᵥΠ)` and the Rayleigh sponge tendency to
    `Yₜ.f.u₃`.

Vertical advection of passive tracers by the mean flow is treated explicitly.
Returns `nothing`.
"""
function implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; microphysics_model, rayleigh_sponge) = p.atmos
    (; params, dt) = p
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
            ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(WVec(-(ᶜwₗ)) * specific(Y.c.ρq_lcl, ᶜρ)))
        @. Yₜ.c.ρq_icl -=
            ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(WVec(-(ᶜwᵢ)) * specific(Y.c.ρq_icl, ᶜρ)))
    end
    if microphysics_model isa
       NonEquilibriumMicrophysics1M
        (; ᶜwᵣ, ᶜwₛ) = p.precomputed
        @. Yₜ.c.ρq_rai -=
            ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(WVec(-(ᶜwᵣ)) * specific(Y.c.ρq_rai, ᶜρ)))
        @. Yₜ.c.ρq_sno -=
            ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(WVec(-(ᶜwₛ)) * specific(Y.c.ρq_sno, ᶜρ)))
    end
    if microphysics_model isa
       NonEquilibriumMicrophysics2M
        (; ᶜwₙₗ, ᶜwₙᵣ, ᶜwᵣ, ᶜwₛ) = p.precomputed
        @. Yₜ.c.ρn_lcl -=
            ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(WVec(-(ᶜwₙₗ)) * specific(Y.c.ρn_lcl, ᶜρ)))
        @. Yₜ.c.ρn_rai -=
            ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(WVec(-(ᶜwₙᵣ)) * specific(Y.c.ρn_rai, ᶜρ)))
        @. Yₜ.c.ρq_rai -=
            ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(WVec(-(ᶜwᵣ)) * specific(Y.c.ρq_rai, ᶜρ)))
        @. Yₜ.c.ρq_sno -=
            ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(WVec(-(ᶜwₛ)) * specific(Y.c.ρq_sno, ᶜρ)))
    end
    if microphysics_model isa NonEquilibriumMicrophysics2MP3
        (; ρn_ice, ρq_rim, ρb_rim) = Y.c
        ᶜwnᵢ = @. lazy(WVec(p.precomputed.ᶜwnᵢ))
        ᶜwᵢ = @. lazy(WVec(p.precomputed.ᶜwᵢ))

        # Note: `ρq_icl` is handled above, in `microphysics_model isa NonEquilibriumMicrophysics`
        @. Yₜ.c.ρn_ice -= ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(- ᶜwnᵢ * specific(ρn_ice, ᶜρ)))
        @. Yₜ.c.ρq_rim -= ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(- ᶜwᵢ * specific(ρq_rim, ᶜρ)))
        @. Yₜ.c.ρb_rim -= ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(- ᶜwᵢ * specific(ρb_rim, ᶜρ)))
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

Apply the post-Newton upwind correction to the central-differenced implicit
vertical advection of `ρe_tot` and `ρq_tot` in
`implicit_vertical_advection_tendency!`. Called by ClimaTimeSteppers as the
`T_post_imp!` hook on `ClimaODEFunction`: evaluated at the Newton-solved stage
state `U*` and applied as `U ← U* + dtγ · Yₜ`. The hook is wired only when
`energy_q_tot_upwinding` is not `Val(:none)`, since the correction is then
identically zero.

Overwrites `Yₜ` with `vtt_upwind - vtt_central` for `ρe_tot` (and `ρq_tot`
when available), where the upwind scheme is given by the
`energy_q_tot_upwinding` numerics option. All other fields of `Yₜ` are zero.
Returns `nothing`.

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
