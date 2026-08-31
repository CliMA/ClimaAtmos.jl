#####
##### Initialize implicit problem
#####

import ClimaCore
import ClimaCore: Fields, Spaces

"""
    initialize_implicit_stage_problem!(Y, p, dtγ)

Initialize the state `Y` for an IMEX implicit stage with step fraction `dtγ`.

Registered with ClimaTimeSteppers as the `initialize_imp!` hook of
`ClimaODEFunction`, so it runs at the start of every implicit stage, before the
Newton solve. For `PrognosticEDMFX` it overwrites the updraft vertical
velocity `Y.f.sgsʲs.:(j).u₃` and area-weighted density `Y.c.sgsʲs.:(j).ρa` with
the stage-consistent values obtained from the analytic solves
`solve_sgs_u₃_implicit_stage_analytic!` and
`solve_sgs_ρa_implicit_stage_analytic!`. Since these values should not be
changed again by the Newton solve, the implied tendencies

    (u₃_stage - u₃_old) / dtγ    and    (ρa_stage - ρa_old) / dtγ

are cached in `p.precomputed.ᶠu₃_tendencyʲs` and `p.precomputed.ᶜρa_tendencyʲs`,
and are returned as the implicit tendencies of these variables by
`sgs_u₃_implicit_tendency!` and `sgs_ρa_implicit_tendency!`, so that the solver
reproduces the analytic stage state. For all other turbulence-convection models
this is a no-op.

This routine acts as a general hook for implicit-stage initialization.
Returns `nothing`.
"""
function initialize_implicit_stage_problem!(Y, p, dtγ)

    if p.atmos.turbconv_model isa PrognosticEDMFX

        (; ᶠu₃_tendencyʲs, ᶜρa_tendencyʲs) = p.precomputed
        n = n_mass_flux_subdomains(p.atmos.turbconv_model)

        # store -u₃_old / dtγ and -ρa_old / dtγ
        for j in 1:n
            @. ᶠu₃_tendencyʲs.:($$j) = -Y.f.sgsʲs.:($$j).u₃ / dtγ
            @. ᶜρa_tendencyʲs.:($$j) = -Y.c.sgsʲs.:($$j).ρa / dtγ
        end

        # analytic stage solves (overwrite Y)
        solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)
        solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ)

        # add +u₃_stage / dtγ → (u₃_stage - u₃_old) / dtγ
        # add +ρa_stage / dtγ → (ρa_stage - ρa_old) / dtγ
        for j in 1:n
            @. ᶠu₃_tendencyʲs.:($$j) += Y.f.sgsʲs.:($$j).u₃ / dtγ
            @. ᶜρa_tendencyʲs.:($$j) += Y.c.sgsʲs.:($$j).ρa / dtγ
        end
    end
end

"""
    sgs_u₃_implicit_tendency!(Yₜ, Y, p, t, turbconv_model)

Write the cached SGS/updraft `u₃` tendency into `Yₜ.f.sgsʲs.:(j).u₃`.

For `PrognosticEDMFX`, the implicit-stage value of `u₃` is computed
analytically in `initialize_implicit_stage_problem!` and written directly
into `Y`. This routine then supplies the corresponding cached tendency,

    (u₃_stage - u₃_old) / dtγ,

read from `p.precomputed.ᶠu₃_tendencyʲs`, to the implicit ODE solve so that the
analytically computed `u₃` value is preserved. Note that the tendency is
assigned, not accumulated, so any earlier updraft `u₃` tendency is discarded.

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
    (; ᶠu₃_tendencyʲs) = p.precomputed
    n = n_mass_flux_subdomains(turbconv_model)

    # Use the implied tendency cached during implicit-stage initialization.
    # This keeps the analytically solved u₃ value fixed during the ODE solve.
    for j in 1:n
        @. Yₜ.f.sgsʲs.:($$j).u₃ = ᶠu₃_tendencyʲs.:($$j)
    end
end

"""
    solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)

Solve the IMEX/ARK implicit-stage equation for the SGS/updraft vertical
velocity `u₃` analytically, overwriting `Y.f.sgsʲs.:(j).u₃` in each EDMFX
mass-flux subdomain `j`.

The result is intended to remain fixed during the subsequent Newton solve; see
`initialize_implicit_stage_problem!` for how it is held fixed. Returns early
for any `turbconv_model` other than `PrognosticEDMFX`.

The underlying evolution equation for the physical vertical velocity `w` is

    ∂w/∂t + ∂(w²/2)/∂z = b + ε (w₀ − w) − (α_d/H) (w − w₀)²,

which, at an IMEX/ARK stage, becomes an algebraic equation for the new
stage value. The environment velocity is eliminated with

    w₀ − w = (ρ / ρa⁰)(w_env − w) ≈ −(ρ / ρa⁰) w,

so that the turbulent entrainment rate and the entrainment branch of the signed
area-bounding rate contribute a linear sink in `w`, the velocity-proportional
entrainment rate `ε = entr_vel_scale · |w|` contributes a quadratic sink, and
pressure drag is purely quadratic. After rearrangement, the stage equation for
face `i` reduces to

    a u₃² + b u₃ + c − (u₃[i−1] / Δz)² / 2 = 0,

where the prognostic variable is the covariant component `u₃ = w · Δz`, so the
whole equation carries one factor of `Δz` relative to the equation for `w`.
The coefficients `(a, b, c)` collect the implicit stage term `(w − w_old)/dtγ`,
entrainment, nonhydrostatic pressure drag (only when `edmfx_model.nh_pressure`
is `Val(true)`), the Rayleigh sponge damping rate (only when a sponge is
configured), buoyancy reduced by the buoyancy-pressure coefficient
`(1 − α_b)`, and the local part of the vertical advection of kinetic energy.

The equation is swept upward with `Operators.column_accumulate!`, which couples
each face to the one below through `u₃[i−1]`, and each face takes the `+` root
of the quadratic. Clamping the constant term with `min(0, ⋅)` keeps the
discriminant non-negative and the root non-negative, so the solve can never
produce a downward updraft velocity.
"""
function solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)

    p.atmos.turbconv_model isa PrognosticEDMFX || return

    (; params) = p
    (; turbconv_model, rayleigh_sponge) = p.atmos
    (; ᶜρ_diffʲs, ᶜρʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (;
        ᶜturb_entrʲs,
        ᶜentr_vel_scaleʲs,
        ᶜentr_nonvel_rateʲs,
        ᶜarea_bounding_entr_detrʲs,
    ) = p.precomputed
    FT = eltype(p.params)

    turbconv_params = CAP.turbconv_params(params)
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
    a_min = CAP.min_area(turbconv_params)
    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)

    # Approximation factor used in w₀ - w ≈ -(ρ / ρa⁰) w (single-updraft case).
    # For multiple updrafts we approximate ρ / ρa⁰ ≈ 1, which implies w₀ ≈ 0.
    ᶜρ_over_ρa⁰ = p.scratch.ᶜtemp_scalar
    @. ᶜρ_over_ρa⁰ = Y.c.ρ / ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)

    ᶠdz = Fields.Δz_field(axes(Y.f))

    # Face-local coefficients of the rearranged quadratic equation
    #     a w² + b w + c = 0.
    ᶠa = p.scratch.ᶠtemp_scalar
    ᶠb = p.scratch.ᶠtemp_scalar_2
    ᶠc = p.scratch.ᶠtemp_scalar_3

    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n

        # Start from the implicit stage term (w - wᵢ) / (γ Δt).
        @. ᶠa = 0
        @. ᶠb = 1 / dtγ
        @. ᶠc = -1 * (Y.f.sgsʲs.:($$j).u₃.components.data.:1 / dtγ)

        # Implicit entrainment: the velocity-dependent coefficient contributes a
        # quadratic sink (∝ w²) and the background/turbulent entrainment contributes
        # a linear sink (∝ w), after applying the approximation w₀ - w ≈ -(ρ/ρa⁰) w.
        # The positive part of `area_bounding_entr_detr` is the entrainment branch
        # of the signed area-bounding rate (see `area_bounding_entr_detr`).
        # `entr_nonvel_rate` is included in the linear sink, with the assumption
        # that the change in b/w is slow.
        @. ᶠa += ᶠinterp(ᶜentr_vel_scaleʲs.:($$j) * ᶜρ_over_ρa⁰ * ᶜρ_over_ρa⁰) / ᶠdz
        @. ᶠb +=
            ᶠinterp(
                (
                    max(FT(0), ᶜarea_bounding_entr_detrʲs.:($$j)) +
                    ᶜentr_nonvel_rateʲs.:($$j) +
                    ᶜturb_entrʲs.:($$j)
                ) * ᶜρ_over_ρa⁰,
            )

        # Implicit NH pressure drag contributes a quadratic sink in w².
        if p.atmos.edmfx_model.nh_pressure isa Val{true}
            ᶜaʲ = @. lazy(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)))
            ᶜa⁰ = @. lazy(a⁰(Y.c.sgsʲs, ᶜρʲs, turbconv_model))
            # Use a scratch scalar here as @lazy results in a large fused kernel
            # that doesn't work on P100 GPUs.
            ᶜdrag_coeff = p.scratch.ᶜtemp_scalar_2
            @. ᶜdrag_coeff =
                α_d / (2 * scale_height) *
                (1 / sqrt(max(ᶜaʲ, a_min)) + 1 / sqrt(max(ᶜa⁰, a_min)))
            @. ᶠa += ᶠinterp(ᶜdrag_coeff * ᶜa⁰ * ᶜρ_over_ρa⁰ * ᶜρ_over_ρa⁰) / ᶠdz
        end

        # Optional Rayleigh sponge adds extra linear damping near the top.
        if !isnothing(rayleigh_sponge)
            ᶠz = Fields.coordinate_field(Y.f.u₃).z
            zmax = Spaces.z_max(axes(ᶠz))
            @. ᶠb += β_rayleigh_u₃(rayleigh_sponge, ᶠz, zmax)
        end

        # Implicit advection adds a local w² term and couples each face
        # to the previously solved face through w_prev².
        @. ᶠa += (1 / ᶠdz)^2 / 2
        @. ᶠc += (1 - α_b) * ᶠinterp(ᶜρ_diffʲs.:($$j)) * ᶠgradᵥ_ᶜΦ.components.data.:1

        input = @. lazy(tuple(ᶠa, ᶠb, ᶠc, ᶠdz))
        Operators.column_accumulate!(
            Y.f.sgsʲs.:($j).u₃,
            input;
            init = C3(FT(0)),
        ) do u₃_prev_face, (a_face, b_face, c_face, dz_face)

            return C3(
                (
                    -b_face + sqrt(
                        b_face * b_face -
                        4 * a_face *
                        min(0, c_face - (u₃_prev_face[1] / dz_face)^2 / 2),
                    )
                ) / (2 * a_face),
            )

        end
    end

end

"""
    sgs_ρa_implicit_tendency!(Yₜ, Y, p, t, turbconv_model)

Write the cached updraft area-weighted density tendency into
`Yₜ.c.sgsʲs.:(j).ρa`.

For `PrognosticEDMFX`, the implicit-stage value of `ρa` is computed
analytically in `initialize_implicit_stage_problem!` and written directly
into `Y`. This routine supplies the corresponding cached tendency,

    (ρa_stage - ρa_old) / dtγ,

read from `p.precomputed.ᶜρa_tendencyʲs`, to the implicit ODE solve so that the
analytically computed `ρa` value is preserved. The tendency is assigned, not
accumulated, so any earlier updraft `ρa` tendency is discarded.

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
    (; ᶜρa_tendencyʲs) = p.precomputed
    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        @. Yₜ.c.sgsʲs.:($$j).ρa = ᶜρa_tendencyʲs.:($$j)
    end
end

"""
    solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ)

Solve the IMEX/ARK implicit-stage equation for the updraft area-weighted
density `ρa` analytically, overwriting `Y.c.sgsʲs.:(j).ρa` in each EDMFX
mass-flux subdomain `j`.

Returns early for any `turbconv_model` other than `PrognosticEDMFX`; see
`initialize_implicit_stage_problem!` for how the result is held fixed during
the Newton solve.

The flux-form stage equation `∂ρa/∂t + ∂(ρa·w)/∂z = (ε − δ)·ρa` reduces
under first-order upwinding (for upward `ᶠu₃ʲ`) to the forward recurrence

    ρa_new[i] = (numerator[i] + α_bot[i] · ρa_new[i−1]) / denominator[i],
    numerator[i] = ρa_old[i]/dtγ,
    denominator[i] = max(0.1/dtγ, 1/dtγ + α_top[i] − (ε − δ)[i]),
    α_face = (ᶠinterp(ρʲ·J)/ᶠJ · ᶠu₃ʲ/Δz_face) / (ρʲ_upwind · Δz[i]),

evaluated bottom-to-top with `Operators.column_accumulate!`. The floor on the
denominator caps the per-step growth at roughly a factor of ten.

`(ε − δ)` is assembled inline from area-bounding, velocity-scale, and
buoyancy-driven pieces (see `detr_buoy_inv_time_scale`). The
mass-flux-divergence component of detrainment is not part of `(ε − δ)`: it
becomes a multiplicative prefactor on both `α_top` and `α_bot`, so that it is
treated implicitly together with the flux divergence. In the first cell,
`α_bot` is zeroed (the surface flux vanishes because `u₃ = 0` there, and the
upwind density would otherwise read an undefined ghost cell), and the capped
surface mass-flux source `F_sfc/Δz` from `p.precomputed.sfc_mass_flux_sourceʲs`
is added to the numerator as an area-independent constant.

The area limiters in `(ε − δ)` are evaluated at the previous-iterate area
(explicit treatment), so they cannot guarantee `a ∈ [0, a_max]` at the
implicit stage value. The sweep therefore clamps `ρa ∈ [0, ρʲ·a_max]`
per-cell, using the updraft density `ρʲ` so that the cap acts on the draft
area `a = ρa/ρʲ`. Clipping from above is mass-conserving — the excess `ρa` is
absorbed by the environment automatically, acting like instantaneous
detrainment. Clipping from below at `ρʲ·a_min` would *not* be conservative
because it would create updraft mass out of nothing, so the lower bound is set
to zero instead.
"""
function solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ)

    p.atmos.turbconv_model isa PrognosticEDMFX || return

    (; turbconv_model) = p.atmos
    (; ᶜρ_diffʲs, ᶜρʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜentr_vel_scaleʲs, ᶜentr_nonvel_rateʲs, ᶜarea_bounding_entr_detrʲs) =
        p.precomputed
    FT = eltype(p.params)

    turbconv_params = CAP.turbconv_params(p.params)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶠdz = Fields.Δz_field(axes(Y.f))
    ᶠlg = Fields.local_geometry_field(Y.f)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    entr_detr_buoy_inv_tau_max = CAP.entr_detr_buoy_inv_tau_max(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)
    a_max = CAP.max_area(turbconv_params)

    # Cell-centred coefficients of the recurrence:
    #   ᶜnumerator            = ρa_old[i] / dtγ
    #   ᶜdenominator          = 1/dtγ + α_top · (1 − implicit_detr_prefactor) − (ε − δ)
    #   ᶜmass_flux_factor_bot = α_bot · (1 − implicit_detr_prefactor)
    # where α_face = (ᶠinterp(ρʲ·J)/ᶠJ · ᶠu₃ʲ/Δz_face) / (ρʲ_upwind · Δz).
    # For upward flow the upwind density at the bottom face is ρʲ[i−1], which we
    # extract via `ᶠleft_bias(ᶜρʲs)`. The mass-flux-divergence component of the
    # detrainment is folded into `ᶜone_minus_implicit_detr_prefactor` (a
    # multiplicative correction on the implicit advection term), leaving the
    # `(ε − δ)` term to carry only the area-bounding, velocity-scale
    # entrainment, and buoyancy-based detrainment pieces.
    ᶜnumerator = p.scratch.ᶜtemp_scalar
    ᶜdenominator = p.scratch.ᶜtemp_scalar_2
    ᶜmass_flux_factor_bot = p.scratch.ᶜtemp_scalar_3
    ᶜexplicit_entr_minus_detr = p.scratch.ᶜtemp_scalar_4
    ᶠw = p.scratch.ᶠtemp_scalar

    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        @. ᶠw = Y.f.sgsʲs.:($$j).u₃.components.data.:1 / ᶠdz

        # entr and detr
        ᶜlower_limiter_factor = @. lazy(
            lower_area_limiter_factor(
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
                turbconv_params,
            ),
        )
        ᶜupper_limiter_factor = @. lazy(
            upper_area_limiter_factor(
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
                turbconv_params,
            ),
        )
        # Mass-flux-divergence detrainment fraction:
        #   detr_coeff = 1 − U · (1 − L · C)
        # with U = upper limiter (→ 0 as a → a_max), L = lower limiter
        # (→ 0 as a → a_min), C = detr_massflux_vertdiv_coeff.
        # At U = 0 (a → a_max): detr_coeff = 1 — all converging mass is
        # detrained, capping area at a_max regardless of C.
        # The recurrence uses `one_minus_prefactor = U · (1 − L · C)`.
        ᶜone_minus_implicit_detr_prefactor = @. lazy(
            ifelse(
                ᶜdivᵥ(ᶠleft_bias(Y.c.sgsʲs.:($$j).ρa) * Y.f.sgsʲs.:($$j).u₃) < 0,
                ᶜupper_limiter_factor *
                (FT(1) - ᶜlower_limiter_factor * detr_massflux_vertdiv_coeff),
                FT(1),
            ),
        )
        # Inverse buoyancy time-scale on faces (where w lives), interpolated
        # to centers for a smoother (ε − δ).
        ᶜbuoy_inv_time_scale = @. lazy(
            ᶜinterp(
                detr_buoy_inv_time_scale(
                    ᶠw,
                    vertical_buoyancy_acceleration(
                        ᶠinterp(ᶜρ_diffʲs.:($$j)),
                        ᶠgradᵥ_ᶜΦ,
                        ᶠlg,
                    ),
                    entr_detr_buoy_inv_tau_max,
                ),
            ),
        )
        @. ᶜexplicit_entr_minus_detr =
            ᶜarea_bounding_entr_detrʲs.:($$j) +
            ᶜentr_vel_scaleʲs.:($$j) * ᶜinterp(ᶠw) +
            ᶜentr_nonvel_rateʲs.:($$j) -
            ᶜlower_limiter_factor * detr_buoy_coeff * ᶜbuoy_inv_time_scale

        @. ᶜnumerator = Y.c.sgsʲs.:($$j).ρa / dtγ
        # Floor at 0.1/dtγ ⇒ (ε − δ) ≤ 0.9/dtγ (≈10× per-step growth cap).
        @. ᶜdenominator =
            max(
                FT(0.1) / dtγ,
                1 / dtγ - ᶜexplicit_entr_minus_detr +
                ᶜone_minus_implicit_detr_prefactor * ᶜright_bias(
                    ᶠinterp(ᶜρʲs.:($$j) * ᶜJ) / ᶠJ * ᶠw,
                ) / ᶜρʲs.:($$j) / ᶜdz,
            )
        @. ᶜmass_flux_factor_bot =
            ᶜone_minus_implicit_detr_prefactor * ᶜleft_bias(
                ᶠinterp(ᶜρʲs.:($$j) * ᶜJ) / ᶠJ * ᶠw / ᶠleft_bias(ᶜρʲs.:($$j)),
            ) / ᶜdz
        # Cell 1: overwrite α_bot[1] with 0 to bypass the NaN from
        # `ᶠleft_bias(ᶜρʲ)` reading the undefined ghost cell below
        # (physical flux is zero there: u₃ = 0 at the surface).
        ᶜmass_flux_factor_bot_first =
            Fields.field_values(Fields.level(ᶜmass_flux_factor_bot, 1))
        @. ᶜmass_flux_factor_bot_first = FT(0)

        # Surface mass-flux BC: add the capped volumetric source `F_sfc/dz`
        # (an a-independent constant, precomputed) to the cell-1 numerator.
        mass_flux_source_val = Fields.field_values(
            Fields.level(p.precomputed.sfc_mass_flux_sourceʲs.:($j), 1),
        )
        ᶜnumerator_first = Fields.field_values(Fields.level(ᶜnumerator, 1))
        @. ᶜnumerator_first += mass_flux_source_val

        # ρʲ is in the input tuple so the per-cell cap `ρʲ·a_max` is available
        # inside the closure.
        input = @. lazy(tuple(
            ᶜnumerator,
            ᶜdenominator,
            ᶜmass_flux_factor_bot,
            ᶜρʲs.:($$j),
        ))

        # Bottom-to-top sweep. `clamp` to `ρa ∈ [0, ρʲ·a_max]` per-cell.
        Operators.column_accumulate!(
            Y.c.sgsʲs.:($j).ρa,
            input;
            init = FT(0),
        ) do ρa_prev, (num, den, mf_bot, ρ_cell)
            return clamp(
                (num + mf_bot * ρa_prev) / den,
                zero(ρa_prev),
                a_max * ρ_cell,
            )
        end
    end
end
