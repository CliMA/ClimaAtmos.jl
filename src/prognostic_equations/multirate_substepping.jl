#####
##### Multirate horizontal substepping for sound/gravity waves
#####

import ClimaCore: Fields, Geometry, Spaces

"""
    set_fast_precomputed_quantities!(Y_fast, p, t)

Minimal thermodynamic update for fast horizontal substeps. Stripped-down
version of `set_implicit_precomputed_quantities!` that only updates the
quantities needed by `horizontal_dynamics_tendency!`.

The fast substeps only modify `Y_fast.c.ρ`, `Y_fast.c.ρe_tot`, and
`Y_fast.c.uₕ`. Vertical velocity (`Y_fast.f.u₃`) and moisture quantities
are unchanged during fast substeps.
"""
function set_fast_precomputed_quantities!(Y_fast, p, t)
    (; ᶜΦ) = p.core
    (; ᶜu, ᶠu³, ᶜK, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno, ᶜh_tot, ᶜp) =
        p.precomputed
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3
    thermo_params = CAP.thermodynamics_params(p.params)
    microphysics_model = p.atmos.microphysics_model

    # 1. Velocity: uₕ changed, u₃ unchanged
    @. ᶠuₕ³ = $compute_ᶠuₕ³(Y_fast.c.uₕ, Y_fast.c.ρ)
    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, Y_fast.f.u₃, Y_fast.c.uₕ, ᶠuₕ³)

    # 2. Internal energy (ρ and ρe_tot changed, Φ static)
    ᶜe_int = @. lazy(specific(Y_fast.c.ρe_tot, Y_fast.c.ρ) - ᶜK - ᶜΦ)

    # 3. Temperature from internal energy
    if microphysics_model isa EquilibriumMicrophysics0M
        # Saturation adjustment: condensate partition depends on T
        @. ᶜq_tot_safe = max(0, specific(Y_fast.c.ρq_tot, Y_fast.c.ρ))
        (; ᶜsa_result) = p.precomputed
        @. ᶜsa_result = saturation_adjustment_tuple(
            thermo_params,
            TD.ρe(),
            Y_fast.c.ρ,
            ᶜe_int,
            ᶜq_tot_safe,
        )
        @. ᶜT = ᶜsa_result.T
        @. ᶜq_liq_rai = ᶜsa_result.q_liq
        @. ᶜq_ice_sno = ᶜsa_result.q_ice
    else  # DryModel or NonEquilibriumMicrophysics
        # q values don't change during fast substeps, just recompute T
        @. ᶜT = max(
            CAP.T_min_sgs(p.params),
            TD.air_temperature(
                thermo_params,
                ᶜe_int,
                ᶜq_tot_safe,
                ᶜq_liq_rai,
                ᶜq_ice_sno,
            ),
        )
    end

    # 4. Pressure and enthalpy
    ᶜe_tot = @. lazy(specific(Y_fast.c.ρe_tot, Y_fast.c.ρ))
    @. ᶜh_tot = TD.total_enthalpy(
        thermo_params,
        ᶜe_tot,
        ᶜT,
        ᶜq_tot_safe,
        ᶜq_liq_rai,
        ᶜq_ice_sno,
    )
    @. ᶜp = TD.air_pressure(
        thermo_params,
        ᶜT,
        Y_fast.c.ρ,
        ᶜq_tot_safe,
        ᶜq_liq_rai,
        ᶜq_ice_sno,
    )

    # Exner (Π) and virtual potential temperature (θ_v) are computed lazily
    # inside horizontal_dynamics_tendency! — no need to store them here.
    return nothing
end

"""
    slow_remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)

Same as `remaining_tendency!` but skips `horizontal_dynamics_tendency!`,
which is handled separately by fast substeps.
"""
NVTX.@annotate function slow_remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    Yₜ .= zero(eltype(Yₜ))
    horizontal_tracer_advection_tendency!(Yₜ_lim, Y, p, t)
    fill_with_nans!(p)
    # horizontal_dynamics_tendency! is handled by fast substeps
    hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    additional_tendency!(Yₜ, Y, p, t)
    return Yₜ
end

"""
    substepped_remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)

Multirate explicit tendency that substeps the fast horizontal terms
(sound/gravity waves) while evaluating slow terms once per slow step.

Algorithm:
1. Copy Y → Y_fast, then do N fast substeps of `horizontal_dynamics_tendency!`
   (each substep updates thermodynamic quantities from the evolving state).
2. Compute slow tendency into Yₜ (everything except horizontal dynamics).
3. Add linearized fast contribution: `Yₜ += (Y_fast - Y) / dt_slow`.
"""
NVTX.@annotate function substepped_remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    dt_fast = p.atmos.numerics.dt_fast::Float64
    dt_slow = p.dt
    n_substeps = floor(Int, dt_slow / dt_fast)
    @assert n_substeps >= 2 "dt_fast=$dt_fast with dt_slow=$dt_slow gives $n_substeps substeps (need ≥ 2)"

    # 1. Fast substeps (reuse Yₜ as fast tendency buffer)
    (; Y_fast) = p.scratch
    Y_fast .= Y

    for i in 1:n_substeps
        set_fast_precomputed_quantities!(Y_fast, p, t)
        Yₜ .= zero(eltype(Yₜ))
        horizontal_dynamics_tendency!(Yₜ, Y_fast, p, t)
        @. Y_fast += dt_fast * Yₜ
        # DSS: project shared element-boundary nodes for horizontal continuity
        if do_dss(axes(Y_fast.c))
            Spaces.weighted_dss!(
                Y_fast.c => p.ghost_buffer.c,
                Y_fast.f => p.ghost_buffer.f,
            )
        end
    end

    # 2. Slow tendency (overwrites Yₜ)
    slow_remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)

    # 3. Add linearized fast contribution
    @. Yₜ += (Y_fast - Y) / dt_slow

    return Yₜ
end
