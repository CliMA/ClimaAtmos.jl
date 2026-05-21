#####
##### Initialize implicit problem
#####

import ClimaCore
import ClimaCore: Fields

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

        (; ᶠu₃_tendencyʲs) = p.precomputed
        n = n_mass_flux_subdomains(p.atmos.turbconv_model)

        # store -u₃_old / dtγ
        for j in 1:n
            @. ᶠu₃_tendencyʲs.:($$j) = -Y.f.sgsʲs.:($$j).u₃ / dtγ
        end

        # analytic stage solve (overwrites Y)
        solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)

        # add +u₃_stage / dtγ → (u₃_stage - u₃_old) / dtγ
        for j in 1:n
            @. ᶠu₃_tendencyʲs.:($$j) += Y.f.sgsʲs.:($$j).u₃ / dtγ
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
    (; ᶠρ_diffʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜturb_entrʲs, ᶜentrʲs) = p.precomputed
    FT = eltype(p.params)

    turbconv_params = CAP.turbconv_params(params)
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
    H_up_min = CAP.min_updraft_top(turbconv_params)
    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
    drag_coeff = α_d / max(H_up_min, scale_height)

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

        # Implicit entrainment/detrainment contributes a linear sink in w.
        @. ᶠb += ᶠinterp((ᶜentrʲs.:($$j) + ᶜturb_entrʲs.:($$j)) * ᶜρ_over_ρa⁰)

        # Implicit NH pressure drag contributes a quadratic sink in w².
        if p.atmos.edmfx_model.nh_pressure isa Val{true}
            @. ᶠa += ᶠinterp(drag_coeff * ᶜρ_over_ρa⁰ * ᶜρ_over_ρa⁰) / ᶠdz
        end

        # Optional Rayleigh sponge adds extra linear damping near the top.
        if !isnothing(rayleigh_sponge)
            ᶠz = Fields.coordinate_field(Y.f.u₃).z
            zmax = z_max(axes(ᶠz))
            @. ᶠb += β_rayleigh_u₃(rayleigh_sponge, ᶠz, zmax)
        end

        # Implicit advection adds a local w² term and couples each face
        # to the previously solved face through w_prev².
        @. ᶠa += (1 / ᶠdz)^2 / 2
        @. ᶠc += (1 - α_b) * ᶠρ_diffʲs.:($$j) * ᶠgradᵥ_ᶜΦ.components.data.:1

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
