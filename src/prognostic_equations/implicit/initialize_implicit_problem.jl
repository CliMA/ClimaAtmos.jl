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
    `ρʲ`, and hence the area bound below, is that of the stage-entering state
    rather than of the last `cache!` call.

 1. Clamp `Y.c.sgsʲs.:(j).ρa` to `[0, a_max ρʲ]` so the analytic `u₃` solve and
    Newton start from a feasible draft area.

 2. Copy the clamped stage-entering drafts `Y.c.sgsʲs` to
    `p.precomputed.ᶜsgsʲs_stage_snapshot`.

 3. Solve `u₃` analytically (`predict_sgs_u₃!`) from the stage-entering
    velocity, the buoyancy `ᶜρ_diffʲs` of the last `cache!` call (the refreshed
    `ρʲ` carries the extrapolation error of the stage), and the precomputed
    entrainment and drag inputs, then impose the impenetrable velocity on the
    boundary faces (`set_velocity_at_surface!`, `set_velocity_at_top!`), where
    the sweep gives zero or a swept value.

 4. Transport `ρa` (`predict_sgs_ρa_advection!`) and then the draft scalars
    (`predict_sgs_tracer_advection!`) up the column with that velocity, with
    the detrainment of converging mass flux and the surface sources. Newton
    transmits a signal one or two cells up per iteration through the
    lower-bidiagonal advection blocks; the sweeps carry it through the whole
    column, and leave the local terms to Newton.

 5. Shift the grid-mean `ρe_tot`, `ρq_tot`, and tracer densities by the updraft
    increment, so that the environment keeps its stage-entering values
    (`warm_start_grid_mean_tracers!`).

All of this only sets the initial guess: the Newton residual is unchanged.
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
    # `t` is unused by the core refresh.
    set_implicit_precomputed_quantities_core!(Y, p, zero_FT)
    a_max = CAP.max_area(CAP.turbconv_params(p.params))
    for j in 1:n
        @. Y.c.sgsʲs.:($$j).ρa =
            clamp(Y.c.sgsʲs.:($$j).ρa, zero_FT, a_max * ᶜρʲs.:($$j))
    end
    ᶜsnapshot = p.precomputed.ᶜsgsʲs_stage_snapshot
    @. ᶜsnapshot = Y.c.sgsʲs
    predict_sgs_u₃!(Y, p, dtγ)
    if !(p.atmos.prescribed_flow isa PrescribedFlow)
        ᶠuₕ³ = p.scratch.ᶠtemp_CT3
        @. ᶠuₕ³ = $compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ)
        set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)
        set_velocity_at_top!(Y, ᶠuₕ³, turbconv_model)
    end
    predict_sgs_ρa_advection!(Y, p, dtγ)
    predict_sgs_tracer_advection!(Y, p, dtγ)
    warm_start_grid_mean_tracers!(Y, p)
    return nothing
end

"""
    u₃_linear_sink_rate(entr_nonvel_rate, turb_entr, a⁰_inv)

Rate [1/s] of the linear sink of the updraft vertical velocity from the
velocity-independent entrainment rates, `(entr_nonvel_rate + turb_entr) / a⁰`,
where the factor `1/a⁰` comes from eliminating the environment velocity with
`w⁰ − w ≈ −w / a⁰`. Shared by `sgs_u₃_implicit_tendency!`,
`predict_sgs_u₃!`, and `update_sgs_u₃_and_ρa_jacobian!`.
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
Shared by `sgs_u₃_implicit_tendency!`, `predict_sgs_u₃!`,
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
`ρa_mass_flux_detr_prefactor`): `|w|`-scaled and velocity-independent
entrainment minus the limited buoyancy detrainment. Shared by
`sgs_ρa_implicit_tendency!` and `update_sgs_u₃_and_ρa_jacobian!`.
"""
@inline ρa_net_entr_detr_rate(
    entr_vel_scale, w, entr_nonvel_rate, lower_limiter_factor,
    detr_buoy_coeff, buoy_inv_time_scale,
) =
    entr_vel_scale * abs(w) + entr_nonvel_rate -
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
            # cache_imp! imposes the impenetrable u₃ (u³ = 0) on both boundary
            # faces before every Newton residual evaluation
            # (set_velocity_at_surface! / set_velocity_at_top!), and
            # `initialize_implicit_stage_problem!` imposes it on the guess.
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
            Y.f.sgsʲs.:($j).u₃,
        ) do ᶜρaₜʲ, ᶜρaʲ, ᶠu³ʲ, ᶜρʲ, ᶜρ_diffʲ, rates, ᶜr_mass_column,
        mass_flux_source, ᶠgradᵥ_ᶜΦ, ᶜdz, ᶠu₃ʲ
            @inline
            (ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ) = rates
            ᶜr_massʲ = ᶜr_mass_column === () ? zero_FT : ᶜr_mass_column
            ᶠlg = Fields.local_geometry_field(axes(ᶠu³ʲ))
            # Physical vertical velocity of the updraft, from its covariant
            # component; the contravariant ᶠu³ʲ, used for the mass flux below,
            # also carries the horizontal wind across terrain-following levels.
            ᶠw = @. get_physical_w(ᶠu₃ʲ, ᶠlg)
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
                ᶜadvdivᵥ(ᶠρa_flux), ᶜupper_limiter_factor, ᶜlower_limiter_factor,
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
                ᶜone_minus_prefactor * ᶜadvdivᵥ(ᶠρa_flux) +
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
    predict_sgs_u₃!(Y, p, dtγ)

Solve the IMEX/ARK implicit-stage equation for the SGS/updraft vertical
velocity `u₃` analytically, overwriting `Y.f.sgsʲs.:(j).u₃` in each EDMFX
mass-flux subdomain `j`.

The result is used as the Newton predictor (warm start) for the subsequent
Newton solve. The buoyancy is read from `p.precomputed.ᶜρ_diffʲs`, set at the
last full cache update, rather than from the draft density `ᶜρʲs` refreshed at
the stage-entering state, whose extrapolation error would otherwise drive the
predicted velocity; the Newton residual keeps the refreshed buoyancy. Returns
early for any `turbconv_model` other than `PrognosticEDMFX`.

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
NVTX.@annotate function predict_sgs_u₃!(Y, p, dtγ)

    p.atmos.turbconv_model isa PrognosticEDMFX || return

    (; params) = p
    (; turbconv_model, rayleigh_sponge) = p.atmos
    (; ᶜρʲs, ᶜρ_diffʲs) = p.precomputed
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
            Y.f.sgsʲs.:($j).u₃, rates, ᶜρ_diffʲs.:($j), ᶠgradᵥ_ᶜΦ, ᶠdz, ᶠz,
        ) do ᶠB, ᶠK, ᶜρaʲ, ᶜρʲ, ᶜsgsʲs, ᶜρʲs, ᶠu₃ʲ, rates, ᶜρ_diffʲ,
        ᶠgradᵥ_ᶜΦ, ᶠdz, ᶠz
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
            # computed from. The buoyancy reads the cached `ᶜρ_diffʲs` of the
            # last full cache update, not the draft density `ᶜρʲs` refreshed
            # at the stage-entering state, which carries the extrapolation
            # error of the stage; the Newton residual uses the refreshed one.
            ᶠC = ᶠu₃ʲ.components.data.:1
            @. ᶠC =
                2 *
                (
                    -(ᶠC / dtγ) +
                    (1 - α_b) * ᶠinterp(ᶜρ_diffʲ) *
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

"""
    predict_sgs_ρa_advection!(Y, p, dtγ)

Advance the updraft area-weighted density `ρa` of the Newton initial guess by
the upward transport of the implicit stage, overwriting `Y.c.sgsʲs.:(j).ρa`.

The predictor keeps the terms of the `ρa` residual (`sgs_ρa_implicit_tendency!`)
that carry a signal up the column, which a few Newton iterations with a
lower-bidiagonal coupling transmit only one or two cells per iteration: the
first-order upwind flux divergence with the velocity of the analytic `u₃` solve,
the detrainment of converging mass flux (`ρa_mass_flux_detr_prefactor`), and the
surface mass-flux source of the first cell. The local entrainment, buoyancy
detrainment, mass source of sedimentation, and bounds restoring are left to
Newton. With `u³ = max(0, ᶠuₕ³ + u₃)` on the faces and `f` the detrainment
prefactor, backward Euler reads

```
ρa[i] (1/dtγ + f[i] J⁺u³⁺ / J[i])
    = ρa_old[i]/dtγ + δ_{i,1} S_sfc + f[i] J⁻u³⁻ / J[i] ρa[i−1],
```

which is swept bottom-to-top with `Operators.column_accumulate!`, with no
inflow through the surface. The prefactor and the area limiters inside it are
evaluated once, at the stage-entering area and the predicted velocity. Each cell
is clamped to `[0, ρʲ a_max]`. Writes `p.scratch.ᶜtemp_scalar`,
`ᶜtemp_scalar_2`, and `ᶠtemp_CT3`.
"""
NVTX.@annotate function predict_sgs_ρa_advection!(Y, p, dtγ)
    (; turbconv_model) = p.atmos
    (; ᶜρʲs, sfc_mass_flux_sourceʲs) = p.precomputed
    FT = eltype(p.params)
    (zero_FT, one_FT) = (zero(FT), one(FT))
    turbconv_params = CAP.turbconv_params(p.params)
    a_max = CAP.max_area(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)
    ᶜoffset = p.scratch.ᶜtemp_scalar
    ᶜfactor = p.scratch.ᶜtemp_scalar_2
    ᶠu³ʲ = p.scratch.ᶠtemp_CT3
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        # The contravariant velocity of the analytic `u₃` solve.
        @. ᶠu³ʲ = $compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ) + CT3(Y.f.sgsʲs.:($$j).u₃)
        DataLayouts.foreach_column(
            ᶜoffset, ᶜfactor, Y.c.sgsʲs.:($j).ρa, ᶠu³ʲ, ᶜρʲs.:($j),
            sfc_mass_flux_sourceʲs.:($j),
        ) do ᶜoffset, ᶜfactor, ᶜρaʲ, ᶠu³ʲ, ᶜρʲ, mass_flux_source
            @inline
            ᶜJ = Fields.local_geometry_field(axes(ᶜρaʲ)).J
            ᶠJ = Fields.local_geometry_field(axes(ᶠu³ʲ)).J
            # The upward face fluxes per unit ρa, J u³ / J at the cell.
            ᶠJu³⁺ = @. lazy(ᶠJ * max(zero_FT, ᶠu³ʲ.components.data.:1))
            ᶜaʲ = @. lazy(draft_area(ᶜρaʲ, ᶜρʲ))
            ᶜprefactor = @. lazy(
                ρa_mass_flux_detr_prefactor(
                    ᶜadvdivᵥ(ᶠupwind1(ᶠu³ʲ, ᶜρaʲ)),
                    upper_area_limiter_factor(ᶜaʲ, turbconv_params),
                    lower_area_limiter_factor(ᶜaʲ, turbconv_params),
                    detr_massflux_vertdiv_coeff,
                ),
            )
            ᶜzero_first_cell =
                Operators.SetBoundaryOperator(bottom = Operators.SetValue(zero_FT))
            ᶜone_first_cell =
                Operators.SetBoundaryOperator(bottom = Operators.SetValue(one_FT))
            # `ᶜfactor` holds the denominator until the offset is computed.
            @. ᶜfactor = 1 / dtγ + ᶜprefactor * ᶜtop_bias(ᶠJu³⁺) / ᶜJ
            @. ᶜoffset =
                (
                    ᶜρaʲ / dtγ +
                    ᶜone_first_cell(zero_FT * ᶜJ) * mass_flux_source
                ) / ᶜfactor
            @. ᶜfactor =
                ᶜzero_first_cell(ᶜprefactor * ᶜbottom_bias(ᶠJu³⁺) / ᶜJ) / ᶜfactor
        end
        input = @. lazy(tuple(ᶜoffset, ᶜfactor, ᶜρʲs.:($$j)))
        Operators.column_accumulate!(
            Y.c.sgsʲs.:($j).ρa,
            input;
            init = zero_FT,
        ) do ρa_prev, (offset, factor, ρʲ)
            return clamp(offset + factor * ρa_prev, zero(ρa_prev), a_max * ρʲ)
        end
    end
    return nothing
end

"""
    predict_sgs_tracer_advection!(Y, p, dtγ)

Advance the updraft specific scalars of the Newton initial guess (`mse`,
`q_tot`, and every SGS tracer) by the upward transport of the implicit stage,
overwriting them in `Y.c.sgsʲs.:(j)`; call after `predict_sgs_ρa_advection!`.

The predictor keeps the advective-form first-order upwind transport of
`edmfx_sgs_vertical_advection_tendency!` with the velocity of the analytic `u₃`
solve, and the relaxation of `mse` and `q_tot` toward their buoyant surface
values in the first cell (`edmfx_boundary_condition_tendency!`), at the rate
`r_sfc` of the predicted area. Entrainment, the pressure work, sedimentation,
and microphysics are left to Newton. With `α = J⁻ max(0, u³⁻) / J` from the
bottom face (zero in the first cell, where nothing flows through the surface),
backward Euler gives the convex combination

    χ[i] = (1 − a[i]) χ_old[i] + a[i] χ[i−1],    a = α dtγ / (1 + α dtγ),

after the first cell is relaxed as
`χ[1] = (χ_old[1] + dtγ r_sfc χ_sfc) / (1 + dtγ r_sfc)`. The stage-entering
values of every scalar other than `mse` are floored at zero, so the predicted
values are non-negative. Writes `p.scratch.ᶜtemp_scalar` and `ᶠtemp_CT3`.
"""
NVTX.@annotate function predict_sgs_tracer_advection!(Y, p, dtγ)
    (; turbconv_model) = p.atmos
    (; ᶜρʲs, sfc_mass_flux_sourceʲs, sfc_mse_buoyantʲs, sfc_q_tot_buoyantʲs) =
        p.precomputed
    FT = eltype(p.params)
    zero_FT = zero(FT)
    a_min = CAP.min_area(CAP.turbconv_params(p.params))
    ᶜa = p.scratch.ᶜtemp_scalar
    ᶠu³ʲ = p.scratch.ᶠtemp_CT3
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜzero_first_cell =
        Operators.SetBoundaryOperator(bottom = Operators.SetValue(zero_FT))
    χ_names = sgs_tracer_names(Y)
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        ᶜsgsʲ = Y.c.sgsʲs.:($j)
        # The contravariant velocity of the analytic `u₃` solve.
        @. ᶠu³ʲ = $compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ) + CT3(Y.f.sgsʲs.:($$j).u₃)
        ᶠu³_data = ᶠu³ʲ.components.data.:1
        @. ᶜa = ᶜzero_first_cell(
            dtγ * ᶜbottom_bias(ᶠJ * max(zero_FT, ᶠu³_data)) / ᶜJ,
        )
        @. ᶜa = ᶜa / (1 + ᶜa)

        # Relax the first-cell `mse` and `q_tot` toward their surface values,
        # with the point loop of `edmfx_boundary_condition_tendency!`.
        DataLayouts.foreach_point(
            Fields.field_values(Fields.level(ᶜsgsʲ, 1)),
            Fields.field_values(Fields.level(ᶜρʲs.:($j), 1)),
            Fields.field_values(Fields.level(sfc_mass_flux_sourceʲs.:($j), 1)),
            Fields.field_values(Fields.level(sfc_mse_buoyantʲs.:($j), 1)),
            Fields.field_values(Fields.level(sfc_q_tot_buoyantʲs.:($j), 1)),
        ) do sgsʲ, ρʲ, mass_flux_source, mse_sfc, q_tot_sfc
            dtγ_r_sfc = @inline @. lazy(
                dtγ * edmfx_sfc_relaxation_rate(mass_flux_source, sgsʲ.ρa, ρʲ, a_min),
            )
            @. sgsʲ.mse = (sgsʲ.mse + dtγ_r_sfc * mse_sfc) / (1 + dtγ_r_sfc)
            @. sgsʲ.q_tot =
                (max(zero_FT, sgsʲ.q_tot) + dtγ_r_sfc * q_tot_sfc) / (1 + dtγ_r_sfc)
        end

        sweep!(ᶜχ, floor) =
            Operators.column_accumulate!(
                ᶜχ,
                (@. lazy(tuple(ᶜχ, ᶜa)));
                init = zero_FT,
            ) do χ_prev, (χ_old, a)
                return (1 - a) * (floor ? max(zero(χ_old), χ_old) : χ_old) +
                       a * χ_prev
            end
        sweep!(ᶜsgsʲ.mse, false)
        sweep!(ᶜsgsʲ.q_tot, true)
        MatrixFields.unrolled_foreach(χ_names) do χ_name
            sweep!(MatrixFields.get_field(ᶜsgsʲ, χ_name), true)
        end
    end
    return nothing
end

"""
    warm_start_grid_mean_tracers!(Y, p)

Move the grid-mean `ρe_tot`, `ρq_tot`, and the grid-mean density `ρχ` of every
SGS tracer `χ` of the Newton initial guess by the updraft increment of the
predictor, so that each environment specific value (and `mse⁰`) keeps its
stage-entering value:

    Δρχ     = (ρa⁰_new − ρa⁰_old) max(0, χ⁰_old)
            + Σⱼ (ρaʲ_new χʲ_new − ρaʲ_old χʲ_old),
    Δρe_tot = (ρa⁰_new − ρa⁰_old) mse⁰_old
            + Σⱼ (ρaʲ_new mseʲ_new − ρaʲ_old mseʲ_old),

with the `*_old` values diagnosed from `Y.c` and the stage-entering draft state
in `p.precomputed.ᶜsgsʲs_stage_snapshot`, and the energy decomposition
`ρe_tot = ρa⁰ mse⁰ + Σⱼ ρaʲ mseʲ + (ρK − p)` at fixed `ρK − p`. Without this
update, the environment of the guess absorbs the change of the updraft
content. The grid-mean `ρ` is unchanged. Tracers without a grid-mean
counterpart are skipped. Writes `p.scratch.ᶜtemp_scalar`.
"""
function warm_start_grid_mean_tracers!(Y, p)
    (; turbconv_model) = p.atmos
    (; ᶜh_tot, ᶜK) = p.precomputed
    ᶜsnapshot = p.precomputed.ᶜsgsʲs_stage_snapshot
    ᶜχ⁰_old = p.scratch.ᶜtemp_scalar
    zero_FT = zero(eltype(p.params))
    ᶜρa⁰_old = @. lazy(ρa⁰(Y.c.ρ, ᶜsnapshot, turbconv_model))
    ᶜΔρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model) - ᶜρa⁰_old)

    χ_names = unrolled_filter(
        χ_name -> MatrixFields.has_field(Y.c, get_ρχ_name(χ_name)),
        (@name(q_tot), sgs_tracer_names(Y)...),
    )
    MatrixFields.unrolled_foreach(χ_names) do χ_name
        ᶜρχ = MatrixFields.get_field(Y.c, get_ρχ_name(χ_name))
        contrib(sgsʲ) = sgsʲ.ρa * MatrixFields.get_field(sgsʲ, χ_name)
        @. ᶜχ⁰_old = max(
            zero_FT,
            specific_env_value(
                χ_name, ᶜρχ, Y.c.ρ, ᶜsnapshot, ᶜρa⁰_old, turbconv_model,
            ),
        )
        @. ᶜρχ +=
            ᶜΔρa⁰ * ᶜχ⁰_old + draft_sum(contrib, Y.c.sgsʲs) -
            draft_sum(contrib, ᶜsnapshot)
    end

    contrib_mse(sgsʲ) = sgsʲ.ρa * sgsʲ.mse
    @. ᶜχ⁰_old =
        specific_env_mse(Y.c.ρ, ᶜsnapshot, ᶜh_tot, ᶜK, ᶜρa⁰_old, turbconv_model)
    @. Y.c.ρe_tot +=
        ᶜΔρa⁰ * ᶜχ⁰_old + draft_sum(contrib_mse, Y.c.sgsʲs) -
        draft_sum(contrib_mse, ᶜsnapshot)
    return nothing
end
