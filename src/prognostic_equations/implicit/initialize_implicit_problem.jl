#####
##### Initialize implicit problem
#####

import ClimaCore
import ClimaCore: Fields, Spaces

"""
    initialize_implicit_stage_problem!(Y, p, dtγ)

Initialize the state `Y` for the IMEX implicit stage.

Currently, this analytically solves SGS/updraft `u₃` and overwrites `Y`
with the stage-consistent value. Since this value should not be modified
by the subsequent ODE/Newton solve, we store the implied tendency

    (u₃_stage - u₃_old) / dtγ

in `p.precomputed.ᶠu₃_tendencyʲs`, so that the solver preserves the
updated state.

This routine acts as a general hook for implicit-stage initialization.
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

Set the cached SGS/updraft `u₃` tendency for the implicit stage.

For `PrognosticEDMFX`, the implicit-stage value of `u₃` is computed
analytically in `initialize_implicit_stage_problem!` and written directly
into `Y`. This routine then supplies the corresponding cached tendency,

    (u₃_stage - u₃_old) / dtγ,

to the implicit ODE solve so that the analytically computed `u₃` value is
preserved.

For other turbulence-convection models, this tendency is not applied.
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

Compute and set the IMEX/ARK implicit-stage solution for the SGS/updraft
vertical velocity `u₃` in each EDMFX mass-flux subdomain.

This routine **analytically solves** the implicit stage equation for `u₃`
and writes the stage-consistent value directly into `Y`. The result is
intended to remain fixed during the subsequent ODE solve.

The underlying evolution equation for the vertical velocity `w` is

    ∂w/∂t + ∂k/∂z = b + ε (w₀ − w) − (α/H) (w − w₀)²,

which, at an IMEX/ARK stage, becomes an algebraic equation for the new
stage value. For a single updraft we use the approximation

    w₀ − w = (ρ / ρa⁰)(w_env − w) ≈ −(ρ / ρa⁰) w,

so that entrainment acts as a linear sink in `w` and pressure drag becomes
a quadratic sink in `w²`. After rearrangement, the stage equation reduces to

    a w² + b w + c = 0,

which is solved analytically using a numerically stable quadratic formula.

The coefficients `(a, b, c)` include contributions from:

  - the implicit stage term (`dtγ` scaling),
  - entrainment/detrainment,
  - nonhydrostatic pressure drag,
  - optional Rayleigh damping,
  - and (optionally) implicit vertical advection.

In this discretization, the prognostic variable is the covariant vertical
velocity component `u₃`. The physical vertical velocity `w` is obtained via
a metric scaling (division by Δz), which is consistently applied when
forming the quadratic coefficients.
"""
function solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)

    p.atmos.turbconv_model isa PrognosticEDMFX || return

    (; params) = p
    (; turbconv_model, rayleigh_sponge) = p.atmos
    (; ᶜρ_diffʲs, ᶜρʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜturb_entrʲs, ᶜentr_vel_scaleʲs, ᶜarea_bounding_entr_detrʲs) = p.precomputed
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
        @. ᶠa += ᶠinterp(ᶜentr_vel_scaleʲs.:($$j) * ᶜρ_over_ρa⁰ * ᶜρ_over_ρa⁰) / ᶠdz
        @. ᶠb +=
            ᶠinterp(
                (max(FT(0), ᶜarea_bounding_entr_detrʲs.:($$j)) + ᶜturb_entrʲs.:($$j)) *
                ᶜρ_over_ρa⁰,
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
            @. ᶠa += ᶠinterp(ᶜdrag_coeff * ᶜρ_over_ρa⁰ * ᶜρ_over_ρa⁰) / ᶠdz
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

Set the cached updraft area tendency for the implicit stage.

For `PrognosticEDMFX`, the implicit-stage value of `ρa` is computed
analytically in `initialize_implicit_stage_problem!` and written directly
into `Y`. This routine supplies the corresponding cached tendency,

    (ρa_stage - ρa_old) / dtγ,

to the implicit ODE solve so that the analytically computed `ρa` value is
preserved.
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

Compute and set the IMEX/ARK implicit-stage solution for the updraft
area-weighted density `ρa` in each EDMFX mass-flux subdomain.

The implicit stage equation for ρaʲ in flux form is

    ∂ρa/∂t + ∂(ρa · w) / ∂z = (ε − δ) · ρa,

With first-order upwinding for upward ᶠu₃ʲ, the stage equation reduces to
the forward recurrence

    ρa_new[i] = (ρa_old[i]/dtγ + α_bot · ρa_new[i−1]) / denominator[i],

with

    denominator[i] = 1/dtγ + α_top − (ε − δ)[i],
    α_face         = (ρʲ_face · w_face) / (ρʲ_upwind · Δz[i])
                   = (ᶠinterp(ρʲ·J)/ᶠJ · ᶠu₃ʲ/Δz_face) / (ρʲ_upwind · Δz[i]),

evaluated at the top (i + ½) and bottom (i − ½) faces. The upwind density
`ρʲ_upwind` is taken from the cell below each face for upward flow: ρʲ[i]
at the top face and ρʲ[i−1] at the bottom face.

The `(ε − δ)` piece of the recurrence is assembled inline from
`ᶜarea_bounding_entr_detrʲs`, `ᶜentr_vel_scaleʲs`, and the buoyancy
detrainment expression (see [`detr_buoy_inv_time_scale`](@ref)) using the
freshly solved `u₃` from the preceding analytic solve; the
mass-flux-divergence component of detrainment is folded into a
multiplicative prefactor on the implicit advection term (so it is treated
implicitly together with the flux divergence) rather than added to
`(ε − δ)`.

Surface boundary: at face ½, `u₃ = 0` (no-penetration), so the physical
mass flux into cell 1 vanishes. To avoid NaN from `ᶠleft_bias(ᶜρʲ)`
hitting the undefined ghost cell below the domain, `α_bot[1]` is explicitly
overwritten to zero. The recurrence then reduces at cell 1 to
`ρa_new[1] = ρa_old[1]/dtγ / denominator[1]`, i.e. the first cell evolves
purely from its own forcing and from local entrainment/detrainment, with
zero inflow from below.

`denominator` is floored at `0.1/dtγ`, which is equivalent to bounding
`(ε − δ) ≤ 0.9/dtγ` and has a similar effect as the protection previously
provided by `limit_detrainment`. This caps the per-step growth at ≈ 10×.

The sweep uses `Operators.column_accumulate!` for performance.
"""
function solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ)

    p.atmos.turbconv_model isa PrognosticEDMFX || return

    (; turbconv_model) = p.atmos
    (; ᶜρ_diffʲs, ᶜρʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜentr_vel_scaleʲs, ᶜarea_bounding_entr_detrʲs) = p.precomputed
    FT = eltype(p.params)

    turbconv_params = CAP.turbconv_params(p.params)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶠdz = Fields.Δz_field(axes(Y.f))
    ᶠlg = Fields.local_geometry_field(Y.f)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    detr_buoy_inv_tau_max = CAP.detr_buoy_inv_tau_max(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)

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
    # `ᶠleft_bias`/`ᶠright_bias` (C2F) and `ᶜleft_bias`/`ᶜright_bias` (F2C) come
    # from `abbreviations.jl`.
    ᶜnumerator = p.scratch.ᶜtemp_scalar
    ᶜdenominator = p.scratch.ᶜtemp_scalar_2
    ᶜmass_flux_factor_bot = p.scratch.ᶜtemp_scalar_3
    ᶜexplicit_entr_minus_detr = p.scratch.ᶜtemp_scalar_4
    ᶠw = p.scratch.ᶠtemp_scalar

    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        @. ᶠw = Y.f.sgsʲs.:($$j).u₃.components.data.:1 / ᶠdz

        # entr and detr
        ᶜarea_limiter_factor = @. lazy(
            detr_lower_area_limiter_factor(
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
                turbconv_params,
            ),
        )
        ᶜone_minus_implicit_detr_prefactor = @. lazy(
            ifelse(
                ᶜdivᵥ(ᶠleft_bias(Y.c.sgsʲs.:($$j).ρa) * Y.f.sgsʲs.:($$j).u₃) < 0,
                FT(1) - ᶜarea_limiter_factor * detr_massflux_vertdiv_coeff,
                FT(1),
            ),
        )
        # Evaluate the inverse buoyancy time scale at faces (where w
        # lives naturally) and interpolate to centers for smoother
        # behaviour in the (ε − δ) term.
        ᶜbuoy_inv_time_scale = @. lazy(
            ᶜinterp(
                detr_buoy_inv_time_scale(
                    ᶠw,
                    vertical_buoyancy_acceleration(
                        ᶠinterp(ᶜρ_diffʲs.:($$j)),
                        ᶠgradᵥ_ᶜΦ,
                        ᶠlg,
                    ),
                    detr_buoy_inv_tau_max,
                ),
            ),
        )
        @. ᶜexplicit_entr_minus_detr =
            ᶜarea_bounding_entr_detrʲs.:($$j) +
            ᶜentr_vel_scaleʲs.:($$j) * ᶜinterp(ᶠw) -
            ᶜarea_limiter_factor * detr_buoy_coeff * ᶜbuoy_inv_time_scale

        @. ᶜnumerator = Y.c.sgsʲs.:($$j).ρa / dtγ
        # Floor at 0.1 / dtγ so (ε − δ) is effectively bounded by 0.9/dtγ,
        # mirroring the protection that the previous `limit_detrainment` call
        # provided.
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
        # At cell 1, `ᶠleft_bias(ᶜρʲ)` evaluates at the surface face where
        # the cell below the domain is undefined → NaN. Physically the mass
        # flux is zero there (u₃ = 0 at the surface), so we overwrite
        # α_bot[1] with FT(0) to keep the recurrence well-defined.
        ᶜmass_flux_factor_bot_first =
            Fields.field_values(Fields.level(ᶜmass_flux_factor_bot, 1))
        @. ᶜmass_flux_factor_bot_first = FT(0)

        # Surface mass-flux boundary condition. The capped volumetric
        # mass source rate (`F_sfc / dz`, equivalent to `div(F·ẑ)` at
        # level 1) is precomputed and consumed here as a constant
        # a-independent source in the first-cell numerator.
        mass_flux_source_val = Fields.field_values(
            Fields.level(p.precomputed.sfc_mass_flux_sourceʲs.:($j), 1),
        )
        ᶜnumerator_first = Fields.field_values(Fields.level(ᶜnumerator, 1))
        @. ᶜnumerator_first += mass_flux_source_val

        input = @. lazy(tuple(
            ᶜnumerator,
            ᶜdenominator,
            ᶜmass_flux_factor_bot,
        ))

        # Bottom-to-top sweep. The zero-flux boundary at the surface is
        # imposed by `init = FT(0)` (zero ρa below the column), so the first
        # cell sees no inflow from below and is free to evolve.
        Operators.column_accumulate!(
            Y.c.sgsʲs.:($j).ρa,
            input;
            init = FT(0),
        ) do ρa_prev, (num, den, mf_bot)
            return (num + mf_bot * ρa_prev) / den
        end
    end
end
