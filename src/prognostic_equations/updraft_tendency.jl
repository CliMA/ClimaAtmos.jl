#####
##### Implicit tendencies of the updraft vertical velocity and area
#####

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
        z, z_sfc, c_R, R_max,
    )

Coefficient [1/m] of the quadratic sink `w |w|` of the updraft vertical velocity:
the velocity-proportional entrainment `entr_vel_scale / a⁰²` plus, when
`nh_pressure` is set, the form drag `pressure_drag_coefficient(...) / a⁰`, whose
updraft radius has the height-growing floor of `plume_radius` at height `z`
above the surface height `z_sfc`.
Shared by `sgs_u₃_implicit_tendency!`, `predict_sgs_u₃!`,
and `update_sgs_u₃_and_ρa_jacobian!`.
"""
@inline function u₃_quadratic_sink_rate(
    entr_vel_scale, aʲ, a⁰, a⁰_inv, nh_pressure, α_d, scale_height, a_min, a_max,
    z, z_sfc, c_R, R_max,
)
    drag = ifelse(
        nh_pressure,
        pressure_drag_coefficient(
            α_d, scale_height, aʲ, a⁰, a_min, a_max, z, z_sfc, c_R, R_max,
        ),
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
at 1 s⁻¹ (at `1/dt` under `prescribed_flow`, which evaluates it explicitly), and
the derivative the Jacobian uses for it (1 outside the open
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
    c_R = CAP.pressure_drag_plume_radius_slope(turbconv_params)
    R_max = CAP.pressure_drag_plume_radius_max(turbconv_params)
    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
    nh_pressure = p.atmos.edmfx_model.nh_pressure
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    # Rate [1/s] of the bounds-restoring terms: 1 s⁻¹, which the implicit solve
    # integrates stably, or 1/dt under `prescribed_flow`, where every
    # evaluation is explicit and forward Euler then removes the excess in one
    # step instead of overshooting.
    restoring_rate =
        p.atmos.prescribed_flow isa PrescribedFlow ? one_FT / p.dt : one_FT
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
            ᶠgradᵥ_ᶜΦ, ᶠdz, ᶠz, rates, ᶠKᵥʲs.:($j), ᶜz, z_sfc,
        ) do ᶠu₃ₜʲ, ᶜρaʲ, ᶜsgsʲs, ᶠu₃ʲ, ᶜρʲ, ᶜρʲs, ᶜρ, ᶜρ_ref_excess,
        ᶠgradᵥ_ᶜΦ, ᶠdz, ᶠz, rates, ᶠKᵥʲ, ᶜz, z_sfc
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
                nh_pressure, α_d, scale_height, a_min, a_max, ᶜz, z_sfc, c_R, R_max,
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
            # Linear fixer: −min(u₃, 0) damps negative velocity at
            # `restoring_rate`.
            # Patankar barrier lives only in the Jacobian (virtual drag
            # −B/max(u₃, w_floor Δz) when B < 0, including u₃ ≤ 0), keeping the
            # tendency physical.
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
                restoring_rate * min(ᶠu₃_data, zero_FT)
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
    # Rate [1/s] of the bounds-restoring terms: 1 s⁻¹, which the implicit solve
    # integrates stably, or 1/dt under `prescribed_flow`, where every
    # evaluation is explicit and forward Euler then removes the excess in one
    # step instead of overshooting.
    restoring_rate =
        p.atmos.prescribed_flow isa PrescribedFlow ? one_FT / p.dt : one_FT
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
            Y.f.sgsʲs.:($j).u₃, Y.c.uₕ, Y.c.ρ,
        ) do ᶜρaₜʲ, ᶜρaʲ, ᶠu³ʲ, ᶜρʲ, ᶜρ_diffʲ, rates, ᶜr_mass_column,
        mass_flux_source, ᶠgradᵥ_ᶜΦ, ᶜdz, ᶠu₃ʲ, ᶜuₕ, ᶜρ
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
            # First-order upwind ρa flux (CT3 face field), with the draft's
            # vertical velocity (always from below) and the horizontal wind's
            # component across sloping levels upwinded separately.
            ᶠuₕ³ = compute_ᶠuₕ³(ᶜuₕ, ᶜρ)
            ᶠρa_flux = updraft_ρa_flux(ᶠu₃ʲ, ᶠuₕ³, ᶜρaʲ)
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
            # Bounds restoring: drag back to [0, ρʲ·a_max] when outside, at
            # `restoring_rate`. Not entrainment/detrainment — a numerical
            # stabilizer analogous to the negative-velocity drag on u₃.
            @. ᶜρaₜʲ -= restoring_rate * ρa_bounds_restoring_rate(ᶜρaʲ, ᶜρʲ, a_max)
        end
    end
    return nothing
end
