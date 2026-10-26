#####
##### Initialize implicit problem
#####

import ClimaCore
import ClimaCore: Fields, Spaces

"""
    initialize_implicit_stage_problem!(Y, p, dtγ)

Prepare the state `Y` for an IMEX implicit stage with step fraction `dtγ`,
registered as the `initialize_imp!` hook of `ClimaODEFunction`. A no-op
unless `p.atmos.turbconv_model isa PrognosticEDMFX`.

For `PrognosticEDMFX`:

 1. Run `edmfx_model.updraft_picard_rounds` Picard rounds that alternate
    the analytic `u₃` solve with the semi-implicit tracer sweep, each
    round starting from the stage-initial `Y.(f,c).sgsʲs` (snapshotted
    into pre-allocated scratch). The sweep transports every solved tracer
    and, for the non-equilibrium microphysics, evaluates the draft
    thermodynamics and `ᶜmp_tendencyʲs` at the transported state and adds
    `dtγ` times the source to the condensate and precipitation tracers;
    the draft thermodynamics and buoyancy are refreshed again at the full
    iterate, and the precipitation terminal velocities are refreshed after
    the last round.

 2. Solve `u₃` and `ρa` analytically with the stage-initial values as
    `*_old` and the Picard-converged buoyancy, microphysics, and
    sedimentation inputs, then cache the
    implied tendencies `(u₃_stage - u₃_old) / dtγ` and
    `(ρa_stage - ρa_old) / dtγ` in `p.precomputed.ᶠu₃_tendencyʲs` and
    `p.precomputed.ᶜρa_tendencyʲs`. The matching `sgs_*_implicit_tendency!`
    routines feed these to the Newton solve so it reproduces the analytic
    values exactly. `ᶜmp_tendencyʲs` is held fixed through Newton so the
    Jacobian — which does not include its derivatives — stays consistent.
"""
function initialize_implicit_stage_problem!(Y, p, dtγ)

    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing

    (; microphysics_model, turbconv_model, edmfx_model) = p.atmos
    (; ᶠu₃_tendencyʲs, ᶜρa_tendencyʲs) = p.precomputed
    (; ᶜρ_diffʲs, ᶜρʲs) = p.precomputed
    n = n_mass_flux_subdomains(turbconv_model)
    n_picard = edmfx_model.updraft_picard_rounds
    nonequilibrium_mp =
        microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
    # Cast `dtγ` to the parameter float type so broadcasts stay type-stable
    # (ClimaTimeSteppers passes `dtγ` as `Float64` even when the state is
    # `Float32`).
    dtγ = eltype(p.params)(dtγ)

    for j in 1:n
        @. ᶠu₃_tendencyʲs.:($$j) = -Y.f.sgsʲs.:($$j).u₃ / dtγ
        @. ᶜρa_tendencyʲs.:($$j) = -Y.c.sgsʲs.:($$j).ρa / dtγ
    end

    if n_picard > 0
        ᶠsnapshot = p.precomputed.ᶠsgsʲs_picard_snapshot
        ᶜsnapshot = p.precomputed.ᶜsgsʲs_picard_snapshot
        for j in 1:n
            @. ᶠsnapshot.:($$j) = Y.f.sgsʲs.:($$j)
            @. ᶜsnapshot.:($$j) = Y.c.sgsʲs.:($$j)
            @. ᶜρ_diffʲs.:($$j) = (ᶜρʲs.:($$j) - Y.c.ρ) / ᶜρʲs.:($$j)
        end
        for round in 1:n_picard
            solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)
            solve_sgs_updraft_tracers_implicit_stage_analytic!(Y, p, dtγ)
            # Refresh the buoyancy inputs for the next u₃ solve. For the
            # non-equilibrium schemes, the tracer sweep evaluates
            # ᶜmp_tendencyʲs at the transported state and applies it, and
            # that value is kept for the Newton solve; the 0-moment scheme
            # has no prognostic condensate tracers and evaluates it here, at
            # the full iterate. ᶜρ_diffʲs is only read by the next u₃ solve,
            # so it is safe to fuse into the restore loop.
            set_prognostic_edmf_precomputed_quantities_draft_thermo!(Y, p)
            nonequilibrium_mp || update_updraft_microphysics_tendency_cache!(
                Y, p, microphysics_model, turbconv_model, dtγ,
            )
            # Restore for the next round (or for the final u₃ solve).
            # Faces: always, so u₃_old = stage u₃. Centers: only between
            # rounds — the last round leaves the Picard-converged draft
            # scalars as a warm start for the Newton solve.
            for j in 1:n
                @. ᶜρ_diffʲs.:($$j) = (ᶜρʲs.:($$j) - Y.c.ρ) / ᶜρʲs.:($$j)
                @. Y.f.sgsʲs.:($$j) = ᶠsnapshot.:($$j)
                round == n_picard || @. Y.c.sgsʲs.:($$j) = ᶜsnapshot.:($$j)
            end
        end
        nonequilibrium_mp && set_precipitation_velocities!(
            Y, p, microphysics_model, turbconv_model,
        )
    end

    solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)
    solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ)

    for j in 1:n
        @. ᶠu₃_tendencyʲs.:($$j) += Y.f.sgsʲs.:($$j).u₃ / dtγ
        @. ᶜρa_tendencyʲs.:($$j) += Y.c.sgsʲs.:($$j).ρa / dtγ
    end
    return nothing
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

    ∂w/∂t + ∂(w²/2)/∂z = (1 − α_b) b + E (w₀ − w) − a⁰ d_c (w − w₀) |w − w₀|,

with `d_c` the coefficient of `pressure_drag_coefficient` and `a⁰ = ρ̂⁰/ρ` the
environment weight of the pairwise form drag, which, at an IMEX/ARK stage,
becomes an algebraic equation for the new stage value. The environment
velocity is eliminated with the mass-flux constraint,

    w₀ − w ≈ −w / a⁰,

so that each velocity difference brings one factor of `1/a⁰`: the
velocity-independent entrainment rates (background, the entrainment branch of
the signed area-bounding rate, and turbulent) contribute the linear sink
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
    a_max = CAP.max_area(turbconv_params)
    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)

    # Approximation factor used in w₀ - w ≈ -(1 / a⁰) w (single-updraft case).
    # Guard against unphysical extrapolated areas (ρa is solved analytically
    # AFTER u₃, so the u₃ solve sees a possibly out-of-range ρa). Clamp
    # a⁰ ∈ [1−a_max, 1].
    ᶜa⁰_inv = p.scratch.ᶜtemp_scalar
    @. ᶜa⁰_inv = FT(1) / clamp(a⁰(Y.c.sgsʲs, ᶜρʲs, turbconv_model), 1 - a_max, FT(1))

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

        # Implicit entrainment E (w₀ - w) ≈ -E w / a⁰. The velocity-dependent
        # rate scales with the relative velocity, `entr_vel_scale · |w - w₀|`
        # ≈ `entr_vel_scale · w / a⁰`, so it contributes a quadratic sink (∝ w²)
        # with two factors of 1/a⁰; the velocity-independent rates contribute a
        # linear sink (∝ w) with one. The ρa solve evaluates the same rate with
        # w₀ ≈ 0 (`compute_entrainment`). The positive part of
        # `area_bounding_entr_detr` is the entrainment branch of the signed
        # area-bounding rate (see `area_bounding_entr_detr`). `entr_nonvel_rate`
        # is included in the linear sink, with the assumption that the change
        # in b/w is slow.
        @. ᶠa += ᶠinterp(ᶜentr_vel_scaleʲs.:($$j) * ᶜa⁰_inv * ᶜa⁰_inv) / ᶠdz
        @. ᶠb +=
            ᶠinterp(
                (
                    max(FT(0), ᶜarea_bounding_entr_detrʲs.:($$j)) +
                    ᶜentr_nonvel_rateʲs.:($$j) +
                    ᶜturb_entrʲs.:($$j)
                ) * ᶜa⁰_inv,
            )

        # Implicit NH pressure drag a⁰ d_c (w - w₀)|w - w₀| ≈ d_c w² / a⁰
        # contributes a quadratic sink: two factors of 1/a⁰ from the velocity
        # differences, one cancelled by the environment weight a⁰ of the
        # pairwise form.
        if p.atmos.edmfx_model.nh_pressure
            ᶜaʲ = @. lazy(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)))
            # Use a scratch scalar here as @lazy results in a large fused kernel
            # that doesn't work on P100 GPUs. `1 / ᶜa⁰_inv` is the clamped
            # environment area, which keeps the subdomain sum out of this
            # kernel; the clamp inside the coefficient is then idempotent.
            ᶜdrag_coeff = p.scratch.ᶜtemp_scalar_2
            @. ᶜdrag_coeff = pressure_drag_coefficient(
                α_d, scale_height, ᶜaʲ, 1 / ᶜa⁰_inv, a_min, a_max,
            )
            @. ᶠa += ᶠinterp(ᶜdrag_coeff * ᶜa⁰_inv) / ᶠdz
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
    denominator[i] = max(0.1/dtγ, 1/dtγ + α_top[i] − (ε − δ)[i] − r_sed[i]),
    α_face = (ᶠinterp(ρʲ·J)/ᶠJ · ᶠu₃ʲ/Δz_face) / (ρʲ_upwind · Δz[i]),

evaluated bottom-to-top with `Operators.column_accumulate!`. The floor on the
denominator caps the per-step growth at roughly a factor of ten.

`r_sed = S_sed · ρ̂⁻¹` is the sedimentation mass source of the condensate
species from `updraft_sedimentation_mass_source!` (absent under the
equilibrium 0-moment microphysics) as a rate, with `ρ̂⁻¹` the regularized
inverse `specific(1, ρa, 0, ρʲ, turbconv_model)` that the tracer tendencies
use. It is evaluated at the stage-start area and grid-mean state and the
Picard-converged updraft tracers and terminal velocities, so the stage source
is `r_sed · ρa_new`. Where the draft narrows downward, `S_sed` is proportional
to the local area and `r_sed` does not depend on it, so this matches the
dilution `χ S_sed ρ̂⁻¹` that the Newton solve applies at the stage area; `ρaʲ`
then receives the sedimented mass that `q_totʲ` receives, as the grid-mean `ρ`
does.

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
    #                           − r_sed
    #   ᶜmass_flux_factor_bot = α_bot · (1 − implicit_detr_prefactor)
    # where α_face = (ᶠinterp(ρʲ·J)/ᶠJ · ᶠu₃ʲ/Δz_face) / (ρʲ_upwind · Δz).
    # For upward flow the upwind density at the bottom face is ρʲ[i−1], which we
    # extract via `ᶠbottom_bias(ᶜρʲs)`. The mass-flux-divergence component of the
    # detrainment is folded into `ᶜone_minus_implicit_detr_prefactor` (a
    # multiplicative correction on the implicit advection term), leaving the
    # `(ε − δ)` term to carry only the area-bounding, velocity-scale
    # entrainment, and buoyancy-based detrainment pieces.
    ᶜnumerator = p.scratch.ᶜtemp_scalar
    ᶜdenominator = p.scratch.ᶜtemp_scalar_2
    ᶜmass_flux_factor_bot = p.scratch.ᶜtemp_scalar_3
    ᶜexplicit_entr_minus_detr = p.scratch.ᶜtemp_scalar_4
    ᶠw = p.scratch.ᶠtemp_scalar
    ᶜsed_rate = p.scratch.ᶜtemp_scalar_5
    nonequilibrium_mp =
        p.atmos.microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}

    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        # Computed first because `updraft_sedimentation!` writes the face
        # scratch that holds ᶠw.
        if nonequilibrium_mp
            updraft_sedimentation_mass_source!(ᶜsed_rate, Y, p, j)
            @. ᶜsed_rate *= specific(
                FT(1),
                Y.c.sgsʲs.:($$j).ρa,
                FT(0),
                ᶜρʲs.:($$j),
                turbconv_model,
            )
        else
            @. ᶜsed_rate = 0
        end
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
                ᶜdivᵥ(ᶠbottom_bias(Y.c.sgsʲs.:($$j).ρa) * Y.f.sgsʲs.:($$j).u₃) < 0,
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
        # Floor at 0.1/dtγ ⇒ (ε − δ) + r_sed ≤ 0.9/dtγ
        # (≈10× per-step growth cap).
        @. ᶜdenominator =
            max(
                FT(0.1) / dtγ,
                1 / dtγ - ᶜexplicit_entr_minus_detr - ᶜsed_rate +
                ᶜone_minus_implicit_detr_prefactor * ᶜtop_bias(
                    ᶠinterp(ᶜρʲs.:($$j) * ᶜJ) / ᶠJ * ᶠw,
                ) / ᶜρʲs.:($$j) / ᶜdz,
            )
        @. ᶜmass_flux_factor_bot =
            ᶜone_minus_implicit_detr_prefactor * ᶜbottom_bias(
                ᶠinterp(ᶜρʲs.:($$j) * ᶜJ) / ᶠJ * ᶠw / ᶠbottom_bias(ᶜρʲs.:($$j)),
            ) / ᶜdz
        # Cell 1: overwrite α_bot[1] with 0 to bypass the NaN from
        # `ᶠbottom_bias(ᶜρʲ)` reading the undefined ghost cell below
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

"""
    solve_sgs_updraft_tracers_implicit_stage_analytic!(Y, p, dtγ)

Advance the buoyancy-affecting SGS/updraft specific tracers over the implicit
stage by a semi-implicit first-order upwind sweep, overwriting
`Y.c.sgsʲs.:(j).χ` in each EDMFX mass-flux subdomain `j`. For the
non-equilibrium microphysics, also refresh the draft thermodynamics
(`ᶜTʲs`, `ᶜρʲs`, and the draft condensate fields) and `ᶜmp_tendencyʲs` in
`p.precomputed`. Uses `p.scratch`. A no-op for any `turbconv_model` other
than `PrognosticEDMFX`; returns `nothing`.

Solved: `mse`, `q_tot`, and, for the non-equilibrium microphysics, `q_lcl`,
`q_icl`, `q_rai`, `q_sno`. The advective form with lateral and turbulent
entrainment `ε_tot = ε + ε_turb` toward the environment value `χ⁰` and, in
cell 1 for `mse` and `q_tot`, the surface mass-flux relaxation at the rate
`r_sfc` (`edmfx_sfc_relaxation_rate`) toward the buoyant surface value
`χ_sfc`,

    (χ_new - χ_old) / dtγ + α · (χ_new - χ_new[i-1])
        = ε_tot · (χ⁰ - χ_new) + δ_{i,1} r_sfc · (χ_sfc - χ_new),
    α = ᶜbottom_bias(ᶠw) / ᶜdz,

is swept bottom-to-top with `Operators.column_accumulate!` as

    χ_new[i] =
        (χ_old[i]/dtγ + ε_tot[i] · χ⁰[i]
            + δ_{i,1} r_sfc · χ_sfc + α[i] · χ_new[i-1]) /
        (1/dtγ + α[i] + ε_tot[i] + δ_{i,1} r_sfc).

`α[1]` is zero: the bottom face of cell 1 is the surface, through which there
is no flow, while the covariant `u₃` there is nonzero over sloped terrain
(`set_velocity_at_surface!`). The entrainment rate uses the `u₃` iterate of
the current Picard round, which is the velocity the Newton solve's entrainment
tendency sees once `ᶜuʲs` is refreshed from the stage `u₃`. `χ⁰` is diagnosed
from `Y` before the sweep overwrites the draft scalar.

For the non-equilibrium microphysics, the source follows the transport:
`set_prognostic_edmf_precomputed_quantities_draft_thermo!` and
`update_updraft_microphysics_tendency_cache!` evaluate the draft
thermodynamics and `ᶜmp_tendencyʲs` at the transported state, and each
condensate and precipitation tracer then receives `dtγ · S_mp`. Evaluating
the source after the transport keeps the vapor of each cell consistent with
the condensate that arrives with it, so water that arrives as condensate is
not condensed a second time, and the cached source is the rate that takes
the transported condensate to the stage condensate, as the Newton solve
with frozen `ᶜmp_tendencyʲs` requires.

Called from the Picard loop in `initialize_implicit_stage_problem!` as a
predictor between successive `u₃` solves. The final analytic `u₃` and `ρa`
solves at the end of `initialize_implicit_stage_problem!` evaluate buoyancy and
`ᶜmp_tendencyʲs` at the final predictor iterate, which is also the initial
guess for the Newton solve. Detrainment does not enter the intensive tracer
equation because it removes updraft mass at the updraft's own scalar value.
"""
function solve_sgs_updraft_tracers_implicit_stage_analytic!(Y, p, dtγ)
    p.atmos.turbconv_model isa PrognosticEDMFX || return
    (; microphysics_model, turbconv_model) = p.atmos

    # Names missing from the SGS schema (the condensates under 0M) are
    # skipped by the sweep.
    sgs_tracers = (
        @name(mse),
        @name(q_tot),
        @name(q_lcl),
        @name(q_icl),
        @name(q_rai),
        @name(q_sno),
    )
    _sweep_sgs_updraft_tracers!(Y, p, dtγ, sgs_tracers)

    microphysics_model isa
    Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ||
        return nothing
    set_prognostic_edmf_precomputed_quantities_draft_thermo!(Y, p)
    update_updraft_microphysics_tendency_cache!(
        Y, p, microphysics_model, turbconv_model, dtγ,
    )
    # (χ_name, S_name): a tracer and its entry in `ᶜmp_tendencyʲs`; a pair
    # with either name missing is skipped.
    condensate_sources = (
        (@name(q_lcl), @name(dq_lcl_dt)),
        (@name(q_icl), @name(dq_icl_dt)),
        (@name(q_rai), @name(dq_rai_dt)),
        (@name(q_sno), @name(dq_sno_dt)),
    )
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        sgsʲ = Y.c.sgsʲs.:($j)
        ᶜmp_tendencyʲ = p.precomputed.ᶜmp_tendencyʲs.:($j)
        MatrixFields.unrolled_foreach(condensate_sources) do (χ_name, S_name)
            MatrixFields.has_field(sgsʲ, χ_name) &&
            MatrixFields.has_field(ᶜmp_tendencyʲ, S_name) || return
            χ = MatrixFields.get_field(sgsʲ, χ_name)
            S = MatrixFields.get_field(ᶜmp_tendencyʲ, S_name)
            @. χ += dtγ * S
        end
    end
    return nothing
end

"""
    _sweep_sgs_updraft_tracers!(Y, p, dtγ, sgs_tracers)

Run the transport sweep of `solve_sgs_updraft_tracers_implicit_stage_analytic!`
for the tracer names in `sgs_tracers`, skipping any name missing from the SGS
schema.
"""
function _sweep_sgs_updraft_tracers!(Y, p, dtγ, sgs_tracers)
    FT = eltype(p.params)
    a_min = CAP.min_area(CAP.turbconv_params(p.params))
    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶠdz = Fields.Δz_field(axes(Y.f))
    ᶜnumerator = p.scratch.ᶜtemp_scalar
    ᶜdenominator = p.scratch.ᶜtemp_scalar_2
    ᶜadvection_factor = p.scratch.ᶜtemp_scalar_3
    ᶜentr_tot = p.scratch.ᶜtemp_scalar_4
    ᶠw = p.scratch.ᶠtemp_scalar
    # Cell-1 views: the surface relaxation rate `r_sfc` and the entries it
    # modifies.
    sfc_rate_val = p.scratch.temp_data_level
    num_first = Fields.field_values(Fields.level(ᶜnumerator, 1))
    den_first = Fields.field_values(Fields.level(ᶜdenominator, 1))
    advection_factor_first =
        Fields.field_values(Fields.level(ᶜadvection_factor, 1))

    for j in 1:n_mass_flux_subdomains(p.atmos.turbconv_model)
        sgsʲ = Y.c.sgsʲs.:($j)
        @. ᶠw = Y.f.sgsʲs.:($$j).u₃.components.data.:1 / ᶠdz
        @. ᶜadvection_factor = ᶜbottom_bias(ᶠw) / ᶜdz
        # Cell 1: no flow through the surface, whatever the covariant `u₃`
        # there; with α[1] = 0 the sweep's `init` value drops out.
        @. advection_factor_first = FT(0)
        @. ᶜentr_tot =
            compute_entrainment(
                p.precomputed.ᶜentr_vel_scaleʲs.:($$j),
                p.precomputed.ᶜentr_nonvel_rateʲs.:($$j),
                p.precomputed.ᶜarea_bounding_entr_detrʲs.:($$j),
                ᶜinterp(ᶠw),
            ) + p.precomputed.ᶜturb_entrʲs.:($$j)
        @. ᶜdenominator = 1 / dtγ + ᶜadvection_factor + ᶜentr_tot

        ρʲ_int_val =
            Fields.field_values(Fields.level(p.precomputed.ᶜρʲs.:($j), 1))
        ρaʲ_int_val = Fields.field_values(Fields.level(sgsʲ.ρa, 1))
        mass_flux_source_val = Fields.field_values(
            Fields.level(p.precomputed.sfc_mass_flux_sourceʲs.:($j), 1),
        )
        @. sfc_rate_val = edmfx_sfc_relaxation_rate(
            mass_flux_source_val, ρaʲ_int_val, ρʲ_int_val, a_min,
        )

        MatrixFields.unrolled_foreach(sgs_tracers) do χ_name
            MatrixFields.has_field(sgsʲ, χ_name) || return
            χ = MatrixFields.get_field(sgsʲ, χ_name)
            ᶜχ⁰ =
                χ_name === @name(mse) ? ᶜspecific_env_mse(Y, p) :
                ᶜspecific_env_value(χ_name, Y, p)
            @. ᶜnumerator = χ / dtγ + ᶜentr_tot * ᶜχ⁰
            # Cell 1 of `mse` and `q_tot` relaxes toward the buoyant surface
            # value; the denominator includes `r_sfc` only while one of them
            # is swept.
            sfc_forced = χ_name === @name(mse) || χ_name === @name(q_tot)
            if sfc_forced
                sfc_buoyant_field =
                    χ_name === @name(mse) ?
                    p.precomputed.sfc_mse_buoyantʲs.:($j) :
                    p.precomputed.sfc_q_tot_buoyantʲs.:($j)
                buoyant_val =
                    Fields.field_values(Fields.level(sfc_buoyant_field, 1))
                @. num_first += sfc_rate_val * buoyant_val
                @. den_first += sfc_rate_val
            end
            # Sweep bottom-to-top:
            #   χ_new[i] = (num[i] + α[i] · χ_new[i-1]) / den[i]
            input = @. lazy(tuple(ᶜnumerator, ᶜdenominator, ᶜadvection_factor))
            Operators.column_accumulate!(
                χ, input; init = zero(eltype(χ)),
            ) do χ_prev, (num, den, α)
                return (num + α * χ_prev) / den
            end
            sfc_forced && (@. den_first -= sfc_rate_val)
        end
    end
    return nothing
end
