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

 1. Refresh the `Y`-dependent precomputed quantities
    (`set_implicit_precomputed_quantities_core!`), so that the updraft density
    `ρʲ` (the draft area and its bound `ρʲ a_max` in the sweeps) and the
    grid-mean `h_tot` and `K` (the environment `mse`) are those of the
    stage-entering state rather than of the last `cache!` call.

 2. Copy the stage-entering drafts `Y.c.sgsʲs` to
    `p.precomputed.ᶜsgsʲs_stage_snapshot`.

 3. Solve `u₃` analytically (`predict_sgs_u₃!`) from the stage-entering
    velocity, the buoyancy `ᶜρ_diffʲs` of the last `cache!` call (the refreshed
    `ρʲ` carries the extrapolation error of the stage), and the precomputed
    entrainment and drag inputs, then impose the impenetrable velocity on the
    boundary faces (`set_velocity_at_surface!`, `set_velocity_at_top!`), where
    the sweep gives zero or a swept value.

 4. Transport `ρa` (`predict_sgs_ρa!`) and then the draft scalars
    (`predict_sgs_tracers!`) up the column with that velocity, with
    entrainment and detrainment, the surface sources, and the cached updraft
    microphysics, and then apply the implicit updraft microphysics to the
    transported draft condensate. Newton transmits a signal one or two cells up
    per iteration through the lower-bidiagonal advection blocks; the sweeps
    carry it through the whole column. The terms the sweeps leave out (the
    pressure work, sedimentation, and the bounds restoring of `ρa`) and the
    buoyancy at the guess are left to Newton.

 5. Shift the grid-mean `ρe_tot`, `ρq_tot`, and tracer densities by the updraft
    increment, so that the environment keeps its stage-entering values
    (`warm_start_grid_mean_tracers!`).

All of this only sets the initial guess: the Newton residual is unchanged.
"""
NVTX.@annotate function initialize_implicit_stage_problem!(Y, p, dtγ)
    reset_precip_stage_accumulator!(p)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing

    (; turbconv_model) = p.atmos
    FT = eltype(p.params)
    # Cast dtγ to the parameter float type so broadcasts stay type-stable
    # (ClimaTimeSteppers passes dtγ as Float64 even when the state is Float32).
    dtγ = FT(dtγ)
    zero_FT = zero(FT)
    # `t` is unused by the core refresh.
    set_implicit_precomputed_quantities_core!(Y, p, zero_FT)
    ᶜsnapshot = p.precomputed.ᶜsgsʲs_stage_snapshot
    @. ᶜsnapshot = Y.c.sgsʲs
    predict_sgs_u₃!(Y, p, dtγ)
    if !(p.atmos.prescribed_flow isa PrescribedFlow)
        ᶠuₕ³ = p.scratch.ᶠtemp_CT3
        @. ᶠuₕ³ = $compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ)
        set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)
        set_velocity_at_top!(Y, ᶠuₕ³, turbconv_model)
    end
    predict_sgs_ρa!(Y, p, dtγ)
    predict_sgs_tracers!(Y, p, dtγ)
    warm_start_grid_mean_tracers!(Y, p)
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

    a u₃² + b u₃ + c − (u₃[i−1] / Δz[i−1])² / 2 = 0,

where the prognostic variable is the covariant component `u₃ = w · Δz`, so the
whole equation carries one factor of `Δz` relative to the equation for `w`, and
the inflow kinetic energy `w[i−1]² / 2` converts `u₃[i−1]` with the spacing
`Δz[i−1]` of its own face.
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

with `B = b / 2a`, `C = c / a`, and `K = 1 / (2a Δz[i−1]²)`, so that it reads
three values per face and divides by none. The quadratic sinks contribute
`S / Δz` to `a`, with `S` the face value of their rates, and advection
contributes `1 / 2Δz²`, so `2a Δz² = 1 + 2 S Δz`; with `K₀ = 1 / (1 + 2 S Δz)`,
`1 / 2a = Δz² K₀` and `K = K₀ (Δz / Δz[i−1])²`. The coefficients of each draft
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
    c_R = CAP.pressure_drag_plume_radius_slope(turbconv_params)
    R_max = CAP.pressure_drag_plume_radius_max(turbconv_params)
    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
    nh_pressure = p.atmos.edmfx_model.nh_pressure
    (sponge, ᶠz) =
        isnothing(rayleigh_sponge) ? ((), ()) :
        ((rayleigh_sponge, Spaces.z_max(axes(Y.f))), Fields.coordinate_field(Y.f).z)
    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)

    ᶠdz = Fields.Δz_field(axes(Y.f))
    # Spacing of the face below; the bottom face has none, and its inflow
    # velocity is zero, so any positive value serves there.
    ᶠface_below = Operators.BottomBiasedC2F(bottom = Operators.SetValue(one_FT))
    ᶠdz_below = @. lazy(ᶠface_below(ᶜbottom_bias(ᶠdz)))

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
            Y.f.sgsʲs.:($j).u₃, rates, ᶜρ_diffʲs.:($j), ᶠgradᵥ_ᶜΦ, ᶠdz,
            ᶠdz_below, ᶠz, ᶜz, z_sfc,
        ) do ᶠB, ᶠK, ᶜρaʲ, ᶜρʲ, ᶜsgsʲs, ᶜρʲs, ᶠu₃ʲ, rates, ᶜρ_diffʲ,
        ᶠgradᵥ_ᶜΦ, ᶠdz, ᶠdz_below, ᶠz, ᶜz, z_sfc
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
                nh_pressure, α_d, scale_height, a_min, a_max, ᶜz, z_sfc, c_R, R_max,
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
            @. ᶠK = ᶠK_value * (ᶠdz / ᶠdz_below)^2
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
    predict_sgs_ρa!(Y, p, dtγ)

Advance the updraft area-weighted density `ρa` of the Newton initial guess over
the implicit stage, overwriting `Y.c.sgsʲs.:(j).ρa`.

The predictor keeps the terms of the `ρa` residual (`sgs_ρa_implicit_tendency!`)
that carry a signal up the column, which a few Newton iterations with a
lower-bidiagonal coupling transmit only one or two cells per iteration: the
first-order upwind flux divergence with the velocity of the analytic `u₃` solve,
the detrainment of converging mass flux (`ρa_mass_flux_detr_prefactor`), and the
surface mass-flux source of the first cell, together with the net entrainment
minus detrainment `ρa_net_entr_detr_rate` at the predicted velocity and the
cached buoyancy `ᶜρ_diffʲs`. The mass sources of sedimentation and of 0M
precipitation removal and the bounds restoring are left to Newton. With
`u³ = CT3(u₃) + max(0, ᶠuₕ³)` on the faces (the residual upwinds the two parts
separately, `updraft_ρa_flux`, and a downward horizontal part is left to
Newton), `f` the detrainment prefactor, and `r` the net rate, of which a loss
is implicit and a gain explicit so that the update of a non-negative `ρa_old`
stays non-negative, backward Euler reads

```
ρa[i] (1/dtγ + f[i] J⁺u³⁺ / J[i] − min(r[i], 0))
    = ρa_old[i] (1/dtγ + max(r[i], 0)) + δ_{i,1} S_sfc
      + f[i] J⁻u³⁻ / J[i] ρa[i−1],
```

which is swept bottom-to-top with `Operators.column_accumulate!`, with no
inflow through the surface. The prefactor, the net rate, and the area limiters
inside them are evaluated once, at the stage-entering area. Each cell
is clamped to `[0, ρʲ a_max]`. Writes `p.scratch.ᶜtemp_scalar`,
`ᶜtemp_scalar_2`, and `ᶠtemp_CT3`.
"""
NVTX.@annotate function predict_sgs_ρa!(Y, p, dtγ)
    (; turbconv_model) = p.atmos
    (; ᶜρʲs, sfc_mass_flux_sourceʲs, ᶜρ_diffʲs) = p.precomputed
    (; ᶜentr_vel_scaleʲs, ᶜentr_nonvel_rateʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    FT = eltype(p.params)
    (zero_FT, one_FT) = (zero(FT), one(FT))
    turbconv_params = CAP.turbconv_params(p.params)
    a_max = CAP.max_area(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    entr_detr_buoy_inv_tau_max = CAP.entr_detr_buoy_inv_tau_max(turbconv_params)
    ᶜoffset = p.scratch.ᶜtemp_scalar
    ᶜfactor = p.scratch.ᶜtemp_scalar_2
    ᶠu³ʲ = p.scratch.ᶠtemp_CT3
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        # The upward velocity: the analytic `u₃` solve plus the upward part of
        # the horizontal wind's component across sloping levels, which the
        # residual upwinds separately (`updraft_ρa_flux`,
        # `updraft_vertical_advection`); a downward part is left to Newton.
        @. ᶠu³ʲ = CT3(Y.f.sgsʲs.:($$j).u₃) +
                  upward_u³($compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ))
        rates = (ᶜentr_vel_scaleʲs.:($j), ᶜentr_nonvel_rateʲs.:($j))
        DataLayouts.foreach_column(
            ᶜoffset, ᶜfactor, Y.c.sgsʲs.:($j).ρa, ᶠu³ʲ, ᶜρʲs.:($j),
            sfc_mass_flux_sourceʲs.:($j), Y.f.sgsʲs.:($j).u₃, ᶜρ_diffʲs.:($j),
            rates, ᶠgradᵥ_ᶜΦ,
        ) do ᶜoffset, ᶜfactor, ᶜρaʲ, ᶠu³ʲ, ᶜρʲ, mass_flux_source, ᶠu₃ʲ,
        ᶜρ_diffʲ, rates, ᶠgradᵥ_ᶜΦ
            @inline
            (ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ) = rates
            ᶜJ = Fields.local_geometry_field(axes(ᶜρaʲ)).J
            ᶠlg = Fields.local_geometry_field(axes(ᶠu³ʲ))
            ᶠJ = ᶠlg.J
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
            # Net entrainment minus detrainment of the residual
            # (`ρa_net_entr_detr_rate`), at the predicted velocity, the
            # stage-entering area and the cached buoyancy: a net loss is
            # implicit, a net gain explicit, so the update of a non-negative
            # `ρa_old` stays non-negative.
            ᶠw = @. lazy(get_physical_w(ᶠu₃ʲ, ᶠlg))
            ᶜnet_rate = @. lazy(
                ρa_net_entr_detr_rate(
                    ᶜentr_vel_scaleʲ, ᶜinterp(ᶠw), ᶜentr_nonvel_rateʲ,
                    lower_area_limiter_factor(ᶜaʲ, turbconv_params),
                    detr_buoy_coeff,
                    ᶜinterp(
                        detr_buoy_inv_time_scale(
                            ᶠw,
                            vertical_buoyancy_acceleration(
                                ᶠinterp(ᶜρ_diffʲ), ᶠgradᵥ_ᶜΦ, ᶠlg,
                            ),
                            entr_detr_buoy_inv_tau_max,
                        ),
                    ),
                ),
            )
            # `ᶜfactor` holds the denominator until the offset is computed.
            @. ᶜfactor =
                1 / dtγ + ᶜprefactor * ᶜtop_bias(ᶠJu³⁺) / ᶜJ -
                min(ᶜnet_rate, zero_FT)
            @. ᶜoffset =
                (
                    ᶜρaʲ * (1 / dtγ + max(ᶜnet_rate, zero_FT)) +
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
    predict_sgs_tracers!(Y, p, dtγ)

Advance the updraft specific scalars of the Newton initial guess (`mse`,
`q_tot`, and every SGS tracer) over the implicit stage, overwriting them in
`Y.c.sgsʲs.:(j)`; call after `predict_sgs_ρa!`. The transport sweep
(`_sweep_sgs_updraft_tracers!`) applies the cached updraft microphysics sources
`ᶜmp_tendencyʲs`, which the residual applies frozen through the stage
(`microphysics_tendency!`), before the transport. With implicit 1M
microphysics, the processes of `microphysics_1m_implicit_params` are then
evaluated at the draft thermodynamics of the transported state, averaged over
`dtγ` (the CM BMT substep averager), and `dtγ` times them is added to `q_lcl`,
`q_icl`, `q_rai`, and `q_sno`, each floored at zero; the four are then
multiplied by `condensate_rescale_ratio` of their sum and `q_tot` (zero where
either is below `ϵ_numerics`), so that their sum does not exceed `q_tot`. The
implicit evaluation uses `ᶜmp_tendency_implicitʲs` and the rescaling
`p.scratch.ᶜtemp_scalar` as scratch; the first Newton residual refreshes the
former, and `cache_imp!` refreshes the draft thermodynamics at the guess.
"""
function predict_sgs_tracers!(Y, p, dtγ)
    _sweep_sgs_updraft_tracers!(Y, p, dtγ)
    (; turbconv_model, microphysics_model) = p.atmos
    (
        microphysics_model isa NonEquilibriumMicrophysics1M &&
        has_implicit_1m_microphysics(p)
    ) || return nothing
    set_prognostic_edmf_precomputed_quantities_draft_thermo!(Y, p)
    (; ᶜρʲs, ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜmp_tendency_implicitʲs) = p.precomputed
    FT = eltype(p.params)
    thp = CAP.thermodynamics_params(p.params)
    cmp_i = CAP.microphysics_1m_implicit_params(p.params)
    nsubs = microphysics_model.n_substeps
    ϵ = ϵ_numerics(FT)
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶜratio = p.scratch.ᶜtemp_scalar
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        ᶜsgsʲ = Y.c.sgsʲs.:($j)
        ᶜSʲ = ᶜmp_tendency_implicitʲs.:($j)
        ᶜwʲ = @. lazy(ᶜinterp(get_physical_w(Y.f.sgsʲs.:($$j).u₃, ᶠlg)))
        @. ᶜSʲ = microphysics_tendencies_1m(
            ᶜρʲs.:($$j), ᶜq_tot_nonnegʲs.:($$j),
            ᶜsgsʲ.q_lcl, ᶜsgsʲ.q_icl, ᶜsgsʲ.q_rai, ᶜsgsʲ.q_sno,
            ᶜTʲs.:($$j), ᶜwʲ, cmp_i, thp, FT(dtγ), nsubs,
        )
        ᶜq_lcl = @. lazy(max(zero(FT), ᶜsgsʲ.q_lcl + dtγ * ᶜSʲ.dq_lcl_dt))
        ᶜq_icl = @. lazy(max(zero(FT), ᶜsgsʲ.q_icl + dtγ * ᶜSʲ.dq_icl_dt))
        ᶜq_rai = @. lazy(max(zero(FT), ᶜsgsʲ.q_rai + dtγ * ᶜSʲ.dq_rai_dt))
        ᶜq_sno = @. lazy(max(zero(FT), ᶜsgsʲ.q_sno + dtγ * ᶜSʲ.dq_sno_dt))
        @. ᶜratio = condensate_rescale_ratio(
            ᶜq_lcl + ᶜq_icl + ᶜq_rai + ᶜq_sno, ᶜsgsʲ.q_tot, ϵ,
        )
        @. ᶜsgsʲ.q_lcl = ᶜq_lcl * ᶜratio
        @. ᶜsgsʲ.q_icl = ᶜq_icl * ᶜratio
        @. ᶜsgsʲ.q_rai = ᶜq_rai * ᶜratio
        @. ᶜsgsʲ.q_sno = ᶜq_sno * ᶜratio
    end
    return nothing
end

"""
    _sweep_sgs_updraft_tracers!(Y, p, dtγ)

Run the transport sweep of `predict_sgs_tracers!` for every draft and every
draft scalar. The sweep keeps the advective-form first-order upwind transport of
`edmfx_sgs_vertical_advection_tendency!` with the velocity of the analytic `u₃`
solve, the relaxation of `mse` and `q_tot` toward their buoyant surface values
in the first cell (`edmfx_boundary_condition_tendency!`), at the rate `r_sfc` of
the predicted area, the entrainment of `edmfx_entr_detr_tendency!` toward the
stage-entering environment, with the tail increments of `mse` and `q_tot`, and
the cached updraft microphysics `ᶜmp_tendencyʲs`, which the residual applies
frozen through the stage (`microphysics_tendency!`) and which acts on each
scalar before its transport (`cached_mp_rate`). The pressure work,
sedimentation, and the implicit microphysics are left out. With
`c = α dtγ`, `α = J⁻ max(0, u³⁻) / J` from the bottom face (zero in the first
cell, where nothing flows through the surface), `e = dtγ (entr + turb_entr)`,
the local source rate `s`, and, for `mse` and `q_tot` in the first cell, the
surface relaxation `r = dtγ r_sfc` toward `χ_sfc`, backward Euler gives

    χ[i] = (χ_old[i] + dtγ s[i] + e χ⁰[i] + r χ_sfc + c χ[i−1]) / (1 + c + e + r).

For every scalar other than `mse`, `χ_old + dtγ s` and `χ⁰` are floored at
zero, so the predicted values are non-negative. One column loop per draft
writes the offset of the recurrence of every scalar into `Y.c.sgsʲs.:(j)` and
the factor of `χ[i−1]`, shared by all scalars, into `p.scratch.ᶜtemp_scalar`;
each scalar is then swept with `Operators.column_accumulate!`. Writes
`p.scratch.ᶜtemp_scalar` and `ᶠtemp_CT3`.
"""
NVTX.@annotate function _sweep_sgs_updraft_tracers!(Y, p, dtγ)
    (; turbconv_model, microphysics_model) = p.atmos
    (; ᶜK, ᶜh_tot, ᶜρʲs, ᶜentr_vel_scaleʲs, ᶜentr_nonvel_rateʲs, ᶜturb_entrʲs) =
        p.precomputed
    (; sfc_mass_flux_sourceʲs, sfc_mse_buoyantʲs, sfc_q_tot_buoyantʲs) =
        p.precomputed
    ᶜsnapshot = p.precomputed.ᶜsgsʲs_stage_snapshot
    ᶜmp_tendencyʲs =
        hasproperty(p.precomputed, :ᶜmp_tendencyʲs) ?
        p.precomputed.ᶜmp_tendencyʲs : nothing
    # The scheme of `cached_mp_rate`, as a singleton the column kernel can
    # carry.
    mp_scheme = cached_mp_scheme(microphysics_model)
    ᶜadvection_factor = p.scratch.ᶜtemp_scalar
    ᶠu³ʲ = p.scratch.ᶠtemp_CT3
    FT = eltype(p.params)
    (zero_FT, one_FT) = (zero(FT), one(FT))
    a_min = CAP.min_area(CAP.turbconv_params(p.params))
    thermo_params = CAP.thermodynamics_params(p.params)
    f_tail = FT(CAP.edmf_tail_entr_fraction(p.params))
    (; dt) = p
    tail_fields = tail_entrainment_fields(p, f_tail)
    χ_names = (@name(mse), @name(q_tot), sgs_tracer_names(Y)...)
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        # The upward velocity: the analytic `u₃` solve plus the upward part of
        # the horizontal wind's component across sloping levels, which the
        # residual upwinds separately (`updraft_ρa_flux`,
        # `updraft_vertical_advection`); a downward part is left to Newton.
        @. ᶠu³ʲ = CT3(Y.f.sgsʲs.:($$j).u₃) +
                  upward_u³($compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ))
        rates = (
            ᶜentr_vel_scaleʲs.:($j), ᶜentr_nonvel_rateʲs.:($j), ᶜturb_entrʲs.:($j),
        )
        sfc = (
            sfc_mass_flux_sourceʲs.:($j), sfc_mse_buoyantʲs.:($j),
            sfc_q_tot_buoyantʲs.:($j),
        )
        # Without a tendency cache (dry), a placeholder that
        # `cached_mp_rate` ignores.
        ᶜmp_tendencyʲ =
            isnothing(ᶜmp_tendencyʲs) ? zero_FT : ᶜmp_tendencyʲs.:($j)
        DataLayouts.foreach_column(
            Y.c.sgsʲs.:($j), Y.c, ᶜsnapshot, ᶜsnapshot.:($j).ρa,
            Y.f.sgsʲs.:($j).u₃, ᶠu³ʲ, ᶜρʲs.:($j), ᶜadvection_factor, ᶜK, ᶜh_tot,
            rates, sfc, tail_fields, ᶜmp_tendencyʲ,
        ) do ᶜsgsʲ, Yc, ᶜsnapshot, ᶜρaʲ_old, ᶠu₃ʲ, ᶠu³ʲ, ᶜρʲ, ᶜadvection_factor,
        ᶜK, ᶜh_tot, rates, (mass_flux_source, mse_sfc, q_tot_sfc), tail_fields,
        ᶜmp_tendencyʲ
            (ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ, ᶜturb_entrʲ) = rates
            ᶜJ = Fields.local_geometry_field(axes(ᶜρʲ)).J
            ᶠlg = Fields.local_geometry_field(axes(ᶠu³ʲ))
            # Cell 1: no flow through the surface, so α[1] = 0. The surface
            # relaxation rate `r_sfc` acts only in cell 1.
            ᶜzero_first_cell =
                Operators.SetBoundaryOperator(bottom = Operators.SetValue(zero_FT))
            ᶜone_first_cell =
                Operators.SetBoundaryOperator(bottom = Operators.SetValue(one_FT))
            ᶜα = @. ᶜzero_first_cell(
                ᶜbottom_bias(ᶠlg.J * max(zero_FT, ᶠu³ʲ.components.data.:1)) / ᶜJ,
            )
            ᶜr_sfc = @. ᶜone_first_cell(zero_FT * ᶜJ) *
               edmfx_sfc_relaxation_rate(mass_flux_source, ᶜsgsʲ.ρa, ᶜρʲ, a_min)
            # Entrainment of `edmfx_entr_detr_tendency!` at the predicted
            # velocity, toward the stage-entering environment, which the warm
            # start (`warm_start_grid_mean_tracers!`) keeps for the guess.
            ᶜentrʲ = @. compute_entrainment(
                ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ,
                ᶜinterp(get_physical_w(ᶠu₃ʲ, ᶠlg)),
            )
            ᶜentr_tot = @. ᶜentrʲ + ᶜturb_entrʲ
            # Tail-entrainment sources `ε Δχ_tail` of `mse` and `q_tot`, as in
            # `edmfx_entr_detr_tendency!`, at the stage-entering areas; zero
            # when tail entrainment is off.
            (ᶜtail_mse, ᶜtail_q_tot) = if isempty(tail_fields)
                (zero_FT, zero_FT)
            else
                (ᶜT⁰, ᶜq_tot⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜT′T′, ᶜq′q′, ᶜcorr_Tq) = tail_fields
                ᶜρa⁰_old = @. ρa⁰(Yc.ρ, ᶜsnapshot, turbconv_model)
                ᶜtailʲ = @. tail_entrainment_increments(
                    thermo_params, f_tail, ᶜT⁰, ᶜq_tot⁰, ᶜq_liq⁰, ᶜq_ice⁰,
                    ᶜT′T′, ᶜq′q′, ᶜcorr_Tq, ᶜρa⁰_old, ᶜρaʲ_old, ᶜentrʲ, dt,
                )
                (
                    @inline(@. lazy(ᶜentrʲ * ᶜtailʲ.mse)),
                    @inline(@. lazy(ᶜentrʲ * ᶜtailʲ.q_tot)),
                )
            end
            ᶜden = @. 1 / dtγ + ᶜα + ᶜentr_tot
            @. ᶜadvection_factor = ᶜα / ᶜden
            # The grid-mean density of each solved scalar (for mse, ρ (h_tot - K))
            # and its tail-entrainment source.
            ᶜρχs = unrolled_map(χ_names) do χ_name
                χ_name === @name(mse) ? (@inline @. lazy(Yc.ρ * (ᶜh_tot - ᶜK))) :
                MatrixFields.get_field(Yc, get_ρχ_name(χ_name))
            end
            ᶜtails = unrolled_map(χ_names) do χ_name
                χ_name === @name(mse) ? ᶜtail_mse :
                χ_name === @name(q_tot) ? ᶜtail_q_tot : zero_FT
            end
            # The state of draft `j` at each point, with each solved scalar
            # replaced by the offset `b = num / den` of its recurrence, with
            # `num = χ_start / dtγ + ε_tot χ⁰` and `χ_start = χ_old + dtγ s` the
            # old value transformed by the local source `s`, the tail
            # entrainment plus the cached updraft microphysics
            # (`cached_mp_rate`), where the
            # environment value `χ⁰` is diagnosed from the grid-mean densities
            # `ρχs` and the stage-entering drafts (as by `specific_env_value`),
            # and where `χ_start` and `χ⁰` of every scalar other than `mse` are
            # floored at zero. `mse` and `q_tot` also relax toward their
            # buoyant surface values at the rate `r_sfc`, which is zero above
            # cell 1.
            ᶜsgsʲ .=
                (@inline (
                    sgsʲs,
                    env_sgsʲs,
                    ρ,
                    ρχs,
                    tails,
                    mpʲ,
                    entr_tot,
                    den,
                    r_sfc,
                    mse_sfc,
                    q_tot_sfc,
                ) -> begin
                    sgsʲ = drop_auto_broadcasters(sgsʲs)[j]
                    ρa⁰_val = ρa⁰(ρ, env_sgsʲs, turbconv_model)
                    names_ρχs_tails = unrolled_map(
                        tuple, χ_names, drop_auto_broadcasters(ρχs),
                        drop_auto_broadcasters(tails),
                    )
                    unrolled_reduce(
                        names_ρχs_tails,
                        sgsʲ,
                    ) do sgs_offsets, (χ_name, ρχ, tail)
                        χ_old = MatrixFields.get_field(sgsʲ, χ_name)
                        s = tail + cached_mp_rate(mp_scheme, mpʲ, χ_name, χ_old)
                        χ⁰ = specific_env_value(
                            χ_name, ρχ, ρ, env_sgsʲs, ρa⁰_val, turbconv_model,
                        )
                        num =
                            χ_name === @name(mse) ?
                            (χ_old + dtγ * s) / dtγ + entr_tot * χ⁰ :
                            max(zero_FT, χ_old + dtγ * s) / dtγ +
                            entr_tot * max(zero_FT, χ⁰)
                        b =
                            χ_name === @name(mse) ?
                            (num + r_sfc * mse_sfc) / (den + r_sfc) :
                            χ_name === @name(q_tot) ?
                            (num + r_sfc * q_tot_sfc) / (den + r_sfc) :
                            num / den
                        merge(
                            sgs_offsets,
                            NamedTuple{(MatrixFields.extract_first(χ_name),)}((b,)),
                        )
                    end
                end).(
                    Yc.sgsʲs,
                    ᶜsnapshot,
                    Yc.ρ,
                    tuple.(ᶜρχs...),
                    tuple.(ᶜtails...),
                    ᶜmp_tendencyʲ,
                    ᶜentr_tot,
                    ᶜden,
                    ᶜr_sfc,
                    mse_sfc,
                    q_tot_sfc,
                )
        end
        MatrixFields.unrolled_foreach(χ_names) do χ_name
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
            input = @. lazy(tuple(ᶜχʲ, ᶜadvection_factor))
            Operators.column_accumulate!(ᶜχʲ, input; init = zero_FT) do χ_prev, (b, a)
                return b + a * χ_prev
            end
        end
    end
    return nothing
end

"""
    cached_mp_scheme(microphysics_model)
    cached_mp_rate(scheme, mpʲ, χ_name, χ_old)

The rate of change of the draft scalar `χ_name`, at its value `χ_old`, from the
cached updraft microphysics tendency `mpʲ` (a point of `ᶜmp_tendencyʲs`), as
`microphysics_tendency!` applies it to the drafts: for 0M, the precipitation
removal `dq_tot_dt` of `q_tot`, as `dq_tot_dt (1 − q_tot)`, and of `mse`, as
`dq_tot_dt (e_tot_hlpr − mse)`; for 1M and 2M, the specific tendency of each
condensate species and, for 2M, of each number. Zero for every other scalar
and microphysics model.

`cached_mp_scheme` reduces the microphysics model to the singleton `scheme`
that selects the method, which a GPU kernel can carry.
"""
cached_mp_scheme(_) = Val(:none)
cached_mp_scheme(::EquilibriumMicrophysics0M) = Val(:mp0m)
cached_mp_scheme(::NonEquilibriumMicrophysics1M) = Val(:mp1m)
cached_mp_scheme(::NonEquilibriumMicrophysics2M) = Val(:mp2m)
@inline cached_mp_rate(::Val{:none}, mpʲ, χ_name, χ_old) = zero(χ_old)
@inline cached_mp_rate(::Val{:mp0m}, mpʲ, χ_name, χ_old) =
    χ_name === @name(q_tot) ? mpʲ.dq_tot_dt * (1 - χ_old) :
    χ_name === @name(mse) ? mpʲ.dq_tot_dt * (mpʲ.e_tot_hlpr - χ_old) :
    zero(χ_old)
@inline cached_mp_rate(::Val{:mp1m}, mpʲ, χ_name, χ_old) =
    χ_name === @name(q_lcl) ? mpʲ.dq_lcl_dt :
    χ_name === @name(q_icl) ? mpʲ.dq_icl_dt :
    χ_name === @name(q_rai) ? mpʲ.dq_rai_dt :
    χ_name === @name(q_sno) ? mpʲ.dq_sno_dt : zero(χ_old)
@inline cached_mp_rate(::Val{:mp2m}, mpʲ, χ_name, χ_old) =
    χ_name === @name(q_lcl) ? mpʲ.dq_lcl_dt :
    χ_name === @name(n_lcl) ? mpʲ.dn_lcl_dt :
    χ_name === @name(q_rai) ? mpʲ.dq_rai_dt :
    χ_name === @name(n_rai) ? mpʲ.dn_rai_dt :
    χ_name === @name(q_icl) ? mpʲ.dq_ice_dt : zero(χ_old)

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
