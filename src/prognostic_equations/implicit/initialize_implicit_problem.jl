#####
##### Initialize implicit problem
#####

import ClimaCore
import ClimaCore: Fields, Spaces

"""
    has_precip_stage_accumulator(p)

Return `true` when `p.scratch` holds an active stage-weighted surface
precipitation accumulator (`n_imp_stages > 0`).
"""
@inline has_precip_stage_accumulator(p) =
    hasproperty(p.scratch, :precip_stage_idx) &&
    hasproperty(p.scratch, :n_imp_stages) &&
    p.scratch.n_imp_stages > 0

"""
    reset_precip_stage_accumulator!(p)

Mark the start of an implicit stage for the stage-weighted precipitation
accumulator: set `p.scratch.precip_stage_pending`, so that the next `cache!`
call (at the Newton-solved stage value) samples the stage, and reset
`p.scratch.precip_stage_idx` to `0` when it has reached `n_imp_stages`, i.e. at
the first implicit stage of a new timestep. Called only from
`initialize_implicit_stage_problem!`, which ClimaTimeSteppers invokes only for
stages with a non-zero implicit diagonal. A no-op when the accumulator is not
active in `p.scratch`.
"""
@inline function reset_precip_stage_accumulator!(p)
    if has_precip_stage_accumulator(p)
        if p.scratch.precip_stage_idx[] >= p.scratch.n_imp_stages
            p.scratch.precip_stage_idx[] = 0
        end
        p.scratch.precip_stage_pending[] = true
    end
    return nothing
end

"""
    initialize_implicit_stage_problem!(Y, p, dtγ)

Prepare the state `Y` for an IMEX implicit stage with step fraction `dtγ`,
registered as the `initialize_imp!` hook of `ClimaODEFunction`. Resets the
implicit-stage index for the stage-weighted precipitation surface-flux
accumulator at the first implicit stage of a timestep, and is otherwise a no-op
unless `p.atmos.turbconv_model isa PrognosticEDMFX`.

For `PrognosticEDMFX`:

 0. Refresh the `Y`-dependent precomputed quantities
    (`set_implicit_precomputed_quantities_core!`), so that the updraft density
    `ρʲ`, and hence the buoyancy and the area bound below, are those of the
    stage-entering state rather than of the last `cache!` call.

 1. Clamp `Y.c.sgsʲs.:(j).ρa` to `[0, a_max ρʲ]` so the analytic `u₃` solve and
    Newton start from a feasible draft area.

 2. Solve `u₃` analytically (`solve_sgs_u₃_implicit_stage_analytic!`) from the
    stage-entering velocity and the precomputed buoyancy, entrainment, and drag
    inputs to warm-start the Newton solve.
"""
NVTX.@annotate function initialize_implicit_stage_problem!(Y, p, dtγ)
    reset_precip_stage_accumulator!(p)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing

    (; turbconv_model) = p.atmos
    (; ᶜρʲs) = p.precomputed
    n = n_mass_flux_subdomains(turbconv_model)
    FT = eltype(p.params)
    # Cast dtγ to the parameter float type so broadcasts stay type-stable
    # (ClimaTimeSteppers passes dtγ as Float64 even when the state is Float32).
    dtγ = FT(dtγ)
    zero_FT = zero(FT)
    # The warm start below solves for u₃ with the buoyancy of the stage-entering
    # state; `t` is unused by the core refresh.
    set_implicit_precomputed_quantities_core!(Y, p, zero_FT)
    a_max = CAP.max_area(CAP.turbconv_params(p.params))
    for j in 1:n
        @. Y.c.sgsʲs.:($$j).ρa =
            clamp(Y.c.sgsʲs.:($$j).ρa, zero_FT, a_max * ᶜρʲs.:($$j))
    end
    solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)
    return nothing
end

"""
    u₃_linear_sink_rate(entr_nonvel_rate, turb_entr, a⁰_inv)

Rate [1/s] of the linear sink of the updraft vertical velocity from the
velocity-independent entrainment rates, `(entr_nonvel_rate + turb_entr) / a⁰`,
where the factor `1/a⁰` comes from eliminating the environment velocity with
`w⁰ − w ≈ −w / a⁰`. Shared by `sgs_u₃_implicit_tendency!`,
`solve_sgs_u₃_implicit_stage_analytic!`, and `update_sgs_u₃_and_ρa_jacobian!`.
"""
@inline u₃_linear_sink_rate(entr_nonvel_rate, turb_entr, a⁰_inv) =
    (entr_nonvel_rate + turb_entr) * a⁰_inv

"""
    u₃_quadratic_sink_rate(
        entr_vel_scale, aʲ, a⁰, a⁰_inv, nh_pressure, α_d, scale_height, a_min, a_max,
    )

Coefficient [1/m] of the quadratic sink `w |w|` of the updraft vertical velocity:
the velocity-proportional entrainment `entr_vel_scale / a⁰²` plus, when
`nh_pressure` is set, the form drag `pressure_drag_coefficient(...) / a⁰`.
Shared by `sgs_u₃_implicit_tendency!`, `solve_sgs_u₃_implicit_stage_analytic!`,
and `update_sgs_u₃_and_ρa_jacobian!`.
"""
@inline function u₃_quadratic_sink_rate(
    entr_vel_scale, aʲ, a⁰, a⁰_inv, nh_pressure, α_d, scale_height, a_min, a_max,
)
    drag = ifelse(
        nh_pressure,
        pressure_drag_coefficient(α_d, scale_height, aʲ, a⁰, a_min, a_max),
        zero(a⁰),
    )
    return (entr_vel_scale * a⁰_inv + drag) * a⁰_inv
end

"""
    ρa_net_entr_detr_rate(
        entr_vel_scale, w, entr_nonvel_rate, lower_limiter_factor,
        detr_buoy_coeff, buoy_inv_time_scale,
    )

Net entrainment minus detrainment rate [1/s] of the updraft `ρa`, without the
mass-flux-divergence detrainment (which enters through
`ρa_mass_flux_detr_prefactor`): velocity-scaled and velocity-independent
entrainment minus the limited buoyancy detrainment. Shared by
`sgs_ρa_implicit_tendency!` and `update_sgs_u₃_and_ρa_jacobian!`.
"""
@inline ρa_net_entr_detr_rate(
    entr_vel_scale, w, entr_nonvel_rate, lower_limiter_factor,
    detr_buoy_coeff, buoy_inv_time_scale,
) =
    entr_vel_scale * w + entr_nonvel_rate -
    lower_limiter_factor * detr_buoy_coeff * buoy_inv_time_scale

"""
    ρa_mass_flux_detr_prefactor(
        div_ρa_flux, upper_limiter_factor, lower_limiter_factor,
        detr_massflux_vertdiv_coeff,
    )

Prefactor on the updraft mass-flux divergence in the `ρa` equation. Where the
flux converges (`div_ρa_flux < 0`), the fraction
`1 − upper_limiter_factor (1 − lower_limiter_factor detr_massflux_vertdiv_coeff)`
of the convergence is detrained, so all of it is detrained at the area limit
(`upper_limiter_factor = 0`); a diverging flux is applied in full. Shared by
`sgs_ρa_implicit_tendency!` and `update_sgs_u₃_and_ρa_jacobian!`.
"""
@inline ρa_mass_flux_detr_prefactor(
    div_ρa_flux, upper_limiter_factor, lower_limiter_factor,
    detr_massflux_vertdiv_coeff,
) = ifelse(
    div_ρa_flux < 0,
    upper_limiter_factor *
    (1 - lower_limiter_factor * detr_massflux_vertdiv_coeff),
    one(upper_limiter_factor),
)

"""
    ρa_bounds_restoring_rate(ρa, ρʲ, a_max)
    ∂ρa_bounds_restoring_rate_∂ρa(ρa, ρʲ, a_max)

The numerical bounds-restoring term `min(ρa, 0) + max(ρa − ρʲ a_max, 0)`
subtracted from the `ρa` tendency, which relaxes `ρa` back into `[0, ρʲ a_max]`
at 1 s⁻¹, and the derivative the Jacobian uses for it (1 outside the open
interval `(0, ρʲ a_max)`, 0 inside). Shared by `sgs_ρa_implicit_tendency!` and
`update_sgs_u₃_and_ρa_jacobian!`.

The derivative is 1 on the bounds as well, where the tendency's derivative is
discontinuous. `initialize_implicit_stage_problem!` clamps `ρa` onto a bound
when the stage target lies outside it, and the derivative there must already
see the restoring term: with a derivative of 0 the first Newton step ignores
the restoring term and moves `ρa` almost all the way to the out-of-bounds
target, where with 1 it moves `ρa` by the target's excess divided by
`1 + dtγ`.
"""
@inline ρa_bounds_restoring_rate(ρa, ρʲ, a_max) =
    min(ρa, zero(ρa)) + max(ρa - ρʲ * a_max, zero(ρa))
@inline ∂ρa_bounds_restoring_rate_∂ρa(ρa, ρʲ, a_max) =
    ifelse(ρa <= 0, one(ρa), zero(ρa)) + ifelse(ρa >= ρʲ * a_max, one(ρa), zero(ρa))

"""
    sgs_u₃_implicit_tendency!(Yₜ, Y, p, t, turbconv_model)

Accumulate the SGS/updraft `u₃` ODE right-hand side into `Yₜ.f.sgsʲs.:(j).u₃`.

For `PrognosticEDMFX`, evaluates the buoyancy, entrainment damping, pressure
drag, and Rayleigh sponge terms at the current Newton iterate `Y`. Because
`implicit_tendency!` zeros `Yₜ` before calling this function, the assignment is
equivalent to accumulation. The tendency is assigned (not accumulated) so any
earlier updraft `u₃` contribution is discarded.

For all other `turbconv_model` subtypes this is a no-op. Mutates `Yₜ` and
returns `nothing`.
"""
sgs_u₃_implicit_tendency!(Yₜ, Y, p, t, _) = nothing

function sgs_u₃_implicit_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    (; params) = p
    (; rayleigh_sponge) = p.atmos
    (; ᶜρʲs, ᶠKᵥʲs, ᶜρ_ref_excess) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (;
        ᶜturb_entrʲs,
        ᶜentr_vel_scaleʲs,
        ᶜentr_nonvel_rateʲs,
    ) = p.precomputed
    FT = eltype(p.params)
    (zero_FT, one_FT) = (zero(FT), one(FT))

    turbconv_params = CAP.turbconv_params(params)
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
    a_min = CAP.min_area(turbconv_params)
    a_max = CAP.max_area(turbconv_params)
    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
    nh_pressure = p.atmos.edmfx_model.nh_pressure
    # Empty tuple when no sponge so that foreach_column gets a consistent type.
    (sponge, ᶠz) =
        isnothing(rayleigh_sponge) ? ((), ()) :
        (
            (rayleigh_sponge, Spaces.z_max(axes(Y.f))),
            Fields.coordinate_field(Y.f).z,
        )
    ᶠdz = Fields.Δz_field(axes(Y.f))
    n = n_mass_flux_subdomains(turbconv_model)

    for j in 1:n
        rates = (
            ᶜentr_vel_scaleʲs.:($j),
            ᶜentr_nonvel_rateʲs.:($j), ᶜturb_entrʲs.:($j),
        )
        DataLayouts.foreach_column(
            Yₜ.f.sgsʲs.:($j).u₃, Y.c.sgsʲs.:($j).ρa, Y.c.sgsʲs,
            Y.f.sgsʲs.:($j).u₃, ᶜρʲs.:($j), ᶜρʲs, Y.c.ρ, ᶜρ_ref_excess,
            ᶠgradᵥ_ᶜΦ, ᶠdz, ᶠz, rates, ᶠKᵥʲs.:($j),
        ) do ᶠu₃ₜʲ, ᶜρaʲ, ᶜsgsʲs, ᶠu₃ʲ, ᶜρʲ, ᶜρʲs, ᶜρ, ᶜρ_ref_excess,
        ᶠgradᵥ_ᶜΦ, ᶠdz, ᶠz, rates, ᶠKᵥʲ
            @inline
            (ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ, ᶜturb_entrʲ) = rates
            # Clamp a⁰ ∈ [1−a_max, 1] to guard out-of-range ρa during Newton.
            ᶜa⁰ = @. clamp(a⁰(ᶜsgsʲs, ᶜρʲs, turbconv_model), 1 - a_max, one_FT)
            ᶜa⁰_inv = @. one_FT / ᶜa⁰
            ᶜlinear_rate = @. u₃_linear_sink_rate(
                ᶜentr_nonvel_rateʲ, ᶜturb_entrʲ, ᶜa⁰_inv,
            )
            ᶜquadratic_rate = @. u₃_quadratic_sink_rate(
                ᶜentr_vel_scaleʲ, draft_area(ᶜρaʲ, ᶜρʲ), ᶜa⁰, ᶜa⁰_inv,
                nh_pressure, α_d, scale_height, a_min, a_max,
            )
            ᶠu₃_data = @. ᶠu₃ʲ.components.data.:1
            # KE advection ∂(w²/2)/∂z: sign-based upwinding — from below for
            # upward flow (u₃ ≥ 0), from above for downward flow.
            ᶜKE_grad = @. ᶜgradᵥ(ᶠKᵥʲ)
            ᶠKE_adv_b = @. ᶠbottom_bias(ᶜKE_grad)
            ᶠKE_adv_t = @. ᶠtop_bias(ᶜKE_grad)
            ᶠKE_adv_b_data = ᶠKE_adv_b.components.data.:1
            ᶠKE_adv_t_data = ᶠKE_adv_t.components.data.:1
            ᶠKE_adv_data =
                @. ifelse(ᶠu₃_data >= zero_FT, ᶠKE_adv_b_data, ᶠKE_adv_t_data)
            # RHS = buoyancy + KE advection + linear + quadratic damping.
            # Linear fixer: −min(u₃, 0) damps negative velocity at 1 s⁻¹.
            # Patankar barrier lives only in the Jacobian (virtual drag
            # −B/max(u₃, ε) when B < 0, including u₃ ≤ 0), keeping the tendency
            # physical.
            # Buoyancy against the reference density ρ + Δρ (`ᶜρ_ref_excess`,
            # frozen at its last explicit update), as in `ᶜρ_diffʲs`.
            ᶠbuoyancy = @. -(1 - α_b) *
               ᶠinterp((ᶜρʲ - ᶜρ - ᶜρ_ref_excess) / ᶜρʲ) *
               ᶠgradᵥ_ᶜΦ.components.data.:1
            ᶠtend = @. (
                ᶠbuoyancy -
                ᶠKE_adv_data -
                ᶠinterp(ᶜlinear_rate) * ᶠu₃_data -
                ᶠinterp(ᶜquadratic_rate) / ᶠdz * ᶠu₃_data * abs(ᶠu₃_data) -
                min(ᶠu₃_data, zero_FT)
            )
            if !isempty(sponge)
                (sponge_model, zmax) = sponge
                @. ᶠtend -= β_rayleigh_u₃(sponge_model, ᶠz, zmax) * ᶠu₃_data
            end
            # cache_imp! resets u₃ = 0 at both boundary faces before every Newton
            # residual evaluation (set_velocity_at_surface! / set_velocity_at_top!),
            # and the analytic warm start gives 0 at the bottom face via column_accumulate!.
            # Zero the boundary face tendencies here so the Newton residual is zero there.
            ᶠset_zero_bc = Operators.SetBoundaryOperator(
                bottom = Operators.SetValue(zero_FT),
                top = Operators.SetValue(zero_FT),
            )
            @. ᶠu₃ₜʲ = C3(ᶠset_zero_bc(ᶠtend))
        end
    end
    return nothing
end

"""
    sgs_ρa_implicit_tendency!(Yₜ, Y, p, t, turbconv_model)

Accumulate the updraft area-weighted density `ρa` ODE right-hand side into
`Yₜ.c.sgsʲs.:(j).ρa`.

For `PrognosticEDMFX`, evaluates the upwind mass-flux divergence (with the
mass-flux-divergence detrainment prefactor), the net entrainment minus
detrainment, the microphysics mass source, and the volumetric surface mass-flux
source at the current Newton iterate `Y`. Because `implicit_tendency!` zeros
`Yₜ` before calling this function, the assignment is equivalent to accumulation.

For all other `turbconv_model` subtypes this is a no-op. Mutates `Yₜ` and
returns `nothing`.
"""
sgs_ρa_implicit_tendency!(Yₜ, Y, p, t, _) = nothing

function sgs_ρa_implicit_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    (; ᶜρ_diffʲs, ᶜρʲs, ᶠu³ʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜentr_vel_scaleʲs, ᶜentr_nonvel_rateʲs) = p.precomputed
    FT = eltype(p.params)
    (zero_FT, one_FT) = (zero(FT), one(FT))

    turbconv_params = CAP.turbconv_params(p.params)
    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶠdz = Fields.Δz_field(axes(Y.f))
    a_max = CAP.max_area(turbconv_params)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    entr_detr_buoy_inv_tau_max = CAP.entr_detr_buoy_inv_tau_max(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)

    nonequilibrium_mp =
        p.atmos.microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
    implicit_0m_mp =
        p.atmos.microphysics_model isa EquilibriumMicrophysics0M
    dtγ = p.dtγ

    ᶜmass_source_rate = p.scratch.ᶜtemp_scalar_5
    n = n_mass_flux_subdomains(turbconv_model)

    for j in 1:n
        nonequilibrium_mp &&
            updraft_sedimentation_rate!(ᶜmass_source_rate, Y, p, j, dtγ)
        ᶜr_mass =
            nonequilibrium_mp ? ᶜmass_source_rate :
            implicit_0m_mp ? p.precomputed.ᶜmp_tendencyʲs.:($j).dq_tot_dt :
            ()
        rates = (ᶜentr_vel_scaleʲs.:($j), ᶜentr_nonvel_rateʲs.:($j))
        DataLayouts.foreach_column(
            Yₜ.c.sgsʲs.:($j).ρa, Y.c.sgsʲs.:($j).ρa, ᶠu³ʲs.:($j),
            ᶜρʲs.:($j), ᶜρ_diffʲs.:($j), rates, ᶜr_mass,
            p.precomputed.sfc_mass_flux_sourceʲs.:($j), ᶠgradᵥ_ᶜΦ, ᶜdz,
            ᶠdz,
        ) do ᶜρaₜʲ, ᶜρaʲ, ᶠu³ʲ, ᶜρʲ, ᶜρ_diffʲ, rates, ᶜr_mass_column,
        mass_flux_source, ᶠgradᵥ_ᶜΦ, ᶜdz, ᶠdz
            @inline
            (ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ) = rates
            ᶜr_massʲ = ᶜr_mass_column === () ? zero_FT : ᶜr_mass_column
            ᶠlg = Fields.local_geometry_field(axes(ᶠu³ʲ))
            # Use the precomputed u³ʲ (CT3) so that the mass-flux divergence
            # is consistent with the Jacobian, which also uses ᶠu³ʲs.
            ᶠu³_data = @. ᶠu³ʲ.components.data.:1
            ᶠw = @. ᶠu³_data * ᶠdz
            # Draft area and area limiters evaluated at the current iterate.
            ᶜaʲ = @. draft_area(ᶜρaʲ, ᶜρʲ)
            ᶜlower_limiter_factor =
                @. lower_area_limiter_factor(ᶜaʲ, turbconv_params)
            ᶜupper_limiter_factor =
                @. upper_area_limiter_factor(ᶜaʲ, turbconv_params)
            # First-order sign-based upwinded ρa flux (CT3 face field): upwinds
            # from below when u³ʲ > 0, from above when u³ʲ < 0.
            ᶠρa_flux = @. ᶠupwind1(ᶠu³ʲ, ᶜρaʲ)
            # Mass-flux-divergence detrainment prefactor: applied when the
            # flux is convergent (divergence < 0).  At the area limit
            # (upper_limiter → 0) all converging mass is detrained.
            ᶜone_minus_prefactor = @. ρa_mass_flux_detr_prefactor(
                ᶜdivᵥ(ᶠρa_flux), ᶜupper_limiter_factor, ᶜlower_limiter_factor,
                detr_massflux_vertdiv_coeff,
            )
            # Buoyancy-driven detrainment rate (negative buoyancy detrain).
            ᶜbuoy_inv_time_scale = @. ᶜinterp(
                detr_buoy_inv_time_scale(
                    ᶠw,
                    vertical_buoyancy_acceleration(
                        ᶠinterp(ᶜρ_diffʲ),
                        ᶠgradᵥ_ᶜΦ,
                        ᶠlg,
                    ),
                    entr_detr_buoy_inv_tau_max,
                ),
            )
            # (ε − δ) without the mass-flux-divergence detrainment piece.
            ᶜexplicit_entr_minus_detr = @. ρa_net_entr_detr_rate(
                ᶜentr_vel_scaleʲ, ᶜinterp(ᶠw), ᶜentr_nonvel_rateʲ,
                ᶜlower_limiter_factor, detr_buoy_coeff, ᶜbuoy_inv_time_scale,
            )
            # Volumetric surface mass-flux source (non-zero at first cell only).
            ᶜone_first_cell =
                Operators.SetBoundaryOperator(
                    bottom = Operators.SetValue(one_FT),
                )
            ᶜsource = @. ᶜone_first_cell(zero_FT * ᶜdz) * mass_flux_source
            # ρa ODE RHS: (ε−δ)·ρa + r_mass·ρa − prefactor·∇·(ρaʲ·u₃) + source.
            @. ᶜρaₜʲ =
                (ᶜexplicit_entr_minus_detr + ᶜr_massʲ) * ᶜρaʲ -
                ᶜone_minus_prefactor * ᶜdivᵥ(ᶠρa_flux) +
                ᶜsource
            # Bounds restoring: 1 s⁻¹ drag back to [0, ρʲ·a_max] when outside.
            # Not entrainment/detrainment — a numerical stabilizer analogous to
            # the negative-velocity drag on u₃.
            @. ᶜρaₜʲ -= ρa_bounds_restoring_rate(ᶜρaʲ, ᶜρʲ, a_max)
        end
    end
    return nothing
end

"""
    solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)

Solve the IMEX/ARK implicit-stage equation for the SGS/updraft vertical
velocity `u₃` analytically, overwriting `Y.f.sgsʲs.:(j).u₃` in each EDMFX
mass-flux subdomain `j`.

The result is used as the Newton predictor (warm start) for the subsequent
Newton solve. Returns early for any `turbconv_model` other than
`PrognosticEDMFX`.

The underlying evolution equation for the physical vertical velocity `w` is

    ∂w/∂t + ∂(w²/2)/∂z = (1 − α_b) b + E (w₀ − w) − a⁰ d_c (w − w₀) |w − w₀|,

with `d_c` the coefficient of `pressure_drag_coefficient` and `a⁰ = ρ̂⁰/ρ` the
environment weight of the pairwise form drag, which, at an IMEX/ARK stage,
becomes an algebraic equation for the new stage value. The environment
velocity is eliminated with the mass-flux constraint,

    w₀ − w ≈ −w / a⁰,

so that each velocity difference brings one factor of `1/a⁰`: the
velocity-independent entrainment rates (background and turbulent) contribute
the linear sink
`E w / a⁰`; the velocity-proportional entrainment rate, which scales with the
relative velocity `entr_vel_scale · |w − w₀|`, contributes the quadratic sink
`entr_vel_scale w² / a⁰²`; and the drag contributes the quadratic sink
`d_c w² / a⁰`, its weight `a⁰` cancelling one of the two factors. After
rearrangement, the stage equation for face `i` reduces to

    a u₃² + b u₃ + c − (u₃[i−1] / Δz)² / 2 = 0,

where the prognostic variable is the covariant component `u₃ = w · Δz`, so the
whole equation carries one factor of `Δz` relative to the equation for `w`.
The coefficients `(a, b, c)` collect the implicit stage term `(w − w_old)/dtγ`,
entrainment, nonhydrostatic pressure drag (only when `edmfx_model.nh_pressure`
is `true`), the Rayleigh sponge damping rate (only when a sponge is
configured), buoyancy reduced by the buoyancy-pressure coefficient
`(1 − α_b)`, and the local part of the vertical advection of kinetic energy.

The equation is swept upward with `Operators.column_accumulate!`, which couples
each face to the one below through `u₃[i−1]`, and each face takes the `+` root
of the quadratic. Clamping the constant term with `min(0, ⋅)` keeps the
discriminant non-negative and the root non-negative, so the solve can never
produce a downward updraft velocity. Since `a > 0`, the sweep takes the root of
the equation divided by `a`,

    u₃ = −B + √(B² − min(0, C − K u₃[i−1]²)),

with `B = b / 2a`, `C = c / a`, and `K = 1 / (2a Δz²)`, so that it reads three
values per face and divides by none. The quadratic sinks contribute `S / Δz` to
`a`, with `S` the face value of their rates, and advection contributes
`1 / 2Δz²`, so `2a Δz² = 1 + 2 S Δz`, and `K = 1 / (1 + 2 S Δz)` and
`1 / 2a = Δz² K` take one division per face. The coefficients of each draft
are computed in one column loop, which writes `B` and `K` to `p.scratch.ᶠtemp_scalar` and
`ᶠtemp_scalar_2` and overwrites `u₃_old` in `u₃` with `C`; the sweep reads `C`
at each face before overwriting it with the solution.
"""
NVTX.@annotate function solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)

    p.atmos.turbconv_model isa PrognosticEDMFX || return

    (; params) = p
    (; turbconv_model, rayleigh_sponge) = p.atmos
    (; ᶜρʲs, ᶜρ_ref_excess) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (;
        ᶜturb_entrʲs,
        ᶜentr_vel_scaleʲs,
        ᶜentr_nonvel_rateʲs,
    ) = p.precomputed
    FT = eltype(p.params)
    (zero_FT, one_FT) = (zero(FT), one(FT))
    inv_dtγ = 1 / dtγ

    turbconv_params = CAP.turbconv_params(params)
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
    a_min = CAP.min_area(turbconv_params)
    a_max = CAP.max_area(turbconv_params)
    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
    nh_pressure = p.atmos.edmfx_model.nh_pressure
    (sponge, ᶠz) =
        isnothing(rayleigh_sponge) ? ((), ()) :
        ((rayleigh_sponge, Spaces.z_max(axes(Y.f))), Fields.coordinate_field(Y.f).z)

    ᶠdz = Fields.Δz_field(axes(Y.f))

    # Face coefficients of the quadratic equation divided by a,
    #     u₃² + 2B u₃ + C − K u₃[i−1]² = 0.
    ᶠB = p.scratch.ᶠtemp_scalar
    ᶠK = p.scratch.ᶠtemp_scalar_2

    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        rates = (
            ᶜentr_vel_scaleʲs.:($j),
            ᶜentr_nonvel_rateʲs.:($j), ᶜturb_entrʲs.:($j),
        )
        DataLayouts.foreach_column(
            ᶠB, ᶠK, Y.c.sgsʲs.:($j).ρa, ᶜρʲs.:($j), Y.c.sgsʲs, ᶜρʲs,
            Y.f.sgsʲs.:($j).u₃, rates, Y.c.ρ, ᶜρ_ref_excess, ᶠgradᵥ_ᶜΦ, ᶠdz, ᶠz,
        ) do ᶠB, ᶠK, ᶜρaʲ, ᶜρʲ, ᶜsgsʲs, ᶜρʲs, ᶠu₃ʲ, rates, ᶜρ, ᶜρ_ref_excess,
        ᶠgradᵥ_ᶜΦ,
        ᶠdz, ᶠz
            @inline
            (ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ, ᶜturb_entrʲ) = rates
            # Approximation factor used in w₀ - w ≈ -(1 / a⁰) w (single-updraft
            # case). Guard against an out-of-range stage-entering ρa.
            # Clamp a⁰ ∈ [1−a_max, 1].
            ᶜa⁰ = @. clamp(a⁰(ᶜsgsʲs, ᶜρʲs, turbconv_model), 1 - a_max, one_FT)
            ᶜa⁰_inv = @. one_FT / ᶜa⁰

            # Implicit entrainment E (w₀ - w) ≈ -E w / a⁰. The velocity-dependent
            # rate scales with the relative velocity, `entr_vel_scale · |w - w₀|`
            # ≈ `entr_vel_scale · w / a⁰`, so it contributes a quadratic sink (∝ w²)
            # with two factors of 1/a⁰; the velocity-independent rates contribute a
            # linear sink (∝ w) with one. The ρa residual evaluates the same rate
            # with w₀ ≈ 0 (`compute_entrainment`). `entr_nonvel_rate` is included in
            # the linear sink, with the assumption that the change in b/w is slow.
            # Implicit NH pressure drag a⁰ d_c (w - w₀)|w - w₀| ≈ d_c w² / a⁰
            # contributes a quadratic sink: two factors of 1/a⁰ from the velocity
            # differences, one cancelled by the environment weight a⁰ of the
            # pairwise form. The clamp inside the coefficient is idempotent. (The
            # flag is tested at each point, so that the loop body has one type.)
            ᶜquadratic_rate = @. u₃_quadratic_sink_rate(
                ᶜentr_vel_scaleʲ, draft_area(ᶜρaʲ, ᶜρʲ), ᶜa⁰, ᶜa⁰_inv,
                nh_pressure, α_d, scale_height, a_min, a_max,
            )
            ᶜlinear_rate = @. u₃_linear_sink_rate(
                ᶜentr_nonvel_rateʲ, ᶜturb_entrʲ, ᶜa⁰_inv,
            )

            # Start from the implicit stage term (w - wᵢ) / (γ Δt). The optional
            # Rayleigh sponge adds extra linear damping near the top.
            ᶠb_entr = @. inv_dtγ + ᶠinterp(ᶜlinear_rate)
            ᶠb = if isempty(sponge)
                ᶠb_entr
            else
                (sponge_model, zmax) = sponge
                @. ᶠb_entr + β_rayleigh_u₃(sponge_model, ᶠz, zmax)
            end

            # Implicit advection adds the local w² term 1 / 2Δz² to a and
            # couples each face to the previously solved face through w_prev².
            ᶠK_value = @. one_FT / (1 + 2 * ᶠinterp(ᶜquadratic_rate) * ᶠdz)
            ᶠinv_2a = @. ᶠdz^2 * ᶠK_value
            @. ᶠB = ᶠb * ᶠinv_2a
            @. ᶠK = ᶠK_value
            # `C` overwrites the stage-initial velocity `u₃_old` that it is
            # computed from.
            ᶠC = ᶠu₃ʲ.components.data.:1
            @. ᶠC =
                2 *
                (
                    -(ᶠC / dtγ) +
                    (1 - α_b) * ᶠinterp((ᶜρʲ - ᶜρ - ᶜρ_ref_excess) / ᶜρʲ) *
                    ᶠgradᵥ_ᶜΦ.components.data.:1
                ) * ᶠinv_2a
        end

        ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
        input = @. lazy(tuple(ᶠB, ᶠK, ᶠu₃ʲ.components.data.:1))
        Operators.column_accumulate!(
            ᶠu₃ʲ,
            input;
            init = C3(zero_FT),
        ) do u₃_prev_face, (B_face, K_face, C_face)
            return C3(
                -B_face +
                sqrt(B_face * B_face - min(0, C_face - K_face * u₃_prev_face[1]^2)),
            )
        end
    end

end
