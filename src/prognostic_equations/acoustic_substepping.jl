#####
##### Horizontal acoustic tendency (fast modes for acoustic substepping)
#####

"""
    horizontal_acoustic_tendency!(Yₜ, Y, p, t)

Compute the horizontal acoustic (sound-wave) contributions to the grid-mean
prognostic tendencies: the horizontal mass-flux divergence on `ρ`, the
horizontal total-enthalpy-flux divergence on `ρe_tot`, and the horizontal
pressure-gradient (split θᵥ-Exner form) on `uₕ`.

The horizontal acoustic subset of `horizontal_dynamics_tendency!`, sub-cycled by
the acoustic-substepping timestepper.
"""
NVTX.@annotate function horizontal_acoustic_tendency!(Yₜ, Y, p, t)
    (; ᶜu, ᶜp, ᶜT, ᶜh_tot, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = thermo_params.cp_d

    @. Yₜ.c.ρ -= split_divₕ(Y.c.ρ * ᶜu, 1)
    @. Yₜ.c.ρe_tot -= split_divₕ(Y.c.ρ * ᶜu, ᶜh_tot)

    ᶜθ_v = p.scratch.ᶜtemp_scalar
    @. ᶜθ_v = theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
    ᶜθ_vr = @. lazy(theta_vr(thermo_params, ᶜp))
    ᶜΠ = @. lazy(TD.exner_given_pressure(thermo_params, ᶜp))
    ᶜθ_v_diff = @. lazy(ᶜθ_v - ᶜθ_vr)
    # symmetric split-form pressure gradient in θᵥ′ = θᵥ − θᵥ,ref:
    # ½ cp_d (θᵥ′ ∇Π + ∇(θᵥ′ Π) − Π ∇θᵥ′)
    @. Yₜ.c.uₕ -= C12(
        cp_d *
        (
            ᶜθ_v_diff * gradₕ(ᶜΠ) + gradₕ(ᶜθ_v_diff * ᶜΠ) - ᶜΠ * gradₕ(ᶜθ_v_diff)
        ) / 2,
    )
    return nothing
end

"""
    kinetic_energy_gradient_tendency!(Yₜ, Y, p, t)

Compute the kinetic-energy-gradient contributions to the grid-mean momentum
tendencies: the horizontal gradient of `K + Φ − Φ_r` on `uₕ` and the vertical
gradient of `K` on `u₃`.

The advective subset of the grid-mean momentum tendency that carries an
acoustic-frequency signal. Re-evaluated inside the acoustic sub-cycle rather than
held in the frozen slow forcing when the timestepper uses `FastKineticEnergy`.
The `uₕ` term matches `horizontal_dynamics_tendency!` and the `u₃` term matches
`explicit_vertical_advection_tendency!`.
"""
NVTX.@annotate function kinetic_energy_gradient_tendency!(Yₜ, Y, p, t)
    (; ᶜK, ᶜp) = p.precomputed
    (; ᶜΦ) = p.core
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜΦ_r = @. lazy(phi_r(thermo_params, ᶜp))
    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜK + ᶜΦ - ᶜΦ_r))
    @. Yₜ.f.u₃ -= ᶠgradᵥ(ᶜK)
    return nothing
end

"""
    grid_mean_acoustic_tendency!(Yₜ, Y, p, t)

Compute the vertical grid-mean acoustic (sound-wave) and vertical-transport contributions to the grid-mean
prognostic tendencies: the vertical mass-flux divergence on `ρ`, the vertical advection of total
enthalpy on `ρe_tot` and (when moist) total specific humidity on `ρq_tot`, the vertical pressure-gradient
and gravity (split θᵥ-Exner form) on `u₃`, and the Rayleigh sponge on `u₃`.

The vertical grid-mean acoustic subset of `implicit_vertical_advection_tendency!`, duplicated rather than
extracted so the default implicit tendency is unchanged. The inner operator of the inner/outer implicit
split of acoustic substepping; the `ρ`/`ρe_tot`/`ρq_tot` vertical transport is kept together to preserve
free-stream consistency.
"""
function grid_mean_acoustic_tendency!(Yₜ, Y, p, t)
    (; microphysics_model, rayleigh_sponge) = p.atmos
    (; energy_q_tot_upwinding) = p.atmos.numerics
    (; params, dt) = p
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶠu³, ᶜp, ᶜh_tot, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)

    @. Yₜ.c.ρ -= ᶜadvdivᵥ(ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠu³)

    vtt = vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, energy_q_tot_upwinding)
    @. Yₜ.c.ρe_tot += vtt
    if !(microphysics_model isa DryModel)
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        vtt = vertical_transport(Y.c.ρ, ᶠu³, ᶜq_tot, dt, energy_q_tot_upwinding)
        @. Yₜ.c.ρq_tot += vtt
    end

    ᶜΦ_r = @. lazy(phi_r(thermo_params, ᶜp))
    ᶜθ_v = p.scratch.ᶜtemp_scalar
    @. ᶜθ_v = theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
    ᶜθ_vr = @. lazy(theta_vr(thermo_params, ᶜp))
    ᶜΠ = @. lazy(TD.exner_given_pressure(thermo_params, ᶜp))
    @. Yₜ.f.u₃ -=
        ᶠgradᵥ_ᶜΦ - ᶠgradᵥ(ᶜΦ_r) + cp_d * (ᶠinterp(ᶜθ_v - ᶜθ_vr)) * ᶠgradᵥ(ᶜΠ)

    rst_u₃ = rayleigh_sponge_tendency_u₃(Y.f.u₃, rayleigh_sponge)
    @. Yₜ.f.u₃ += rst_u₃
    return nothing
end
