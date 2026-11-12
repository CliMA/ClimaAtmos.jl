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
    hasproperty(p.scratch, :sfc_precip_lin_state) &&
    hasproperty(p.scratch, :precip_stage_idx) &&
    hasproperty(p.scratch, :n_imp_stages) &&
    p.scratch.n_imp_stages > 0

"""
    reset_precip_stage_accumulator!(p)

Reset `p.scratch.precip_stage_idx` to `0` when it has reached `n_imp_stages`,
i.e. at the first implicit stage of a new timestep. A no-op when the
stage-weighted precipitation accumulator is not active in `p.scratch`.
"""
@inline function reset_precip_stage_accumulator!(p)
    if has_precip_stage_accumulator(p)
        if p.scratch.precip_stage_idx[] >= p.scratch.n_imp_stages
            p.scratch.precip_stage_idx[] = 0
        end
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

 1. Record `-u₃_old / dtγ` and `-ρa_old / dtγ` from the stage-entering state
    (so the Newton residual reproduces `u₃_stage` and `ρa_stage`), refresh the
    implicit precomputed cache (`set_implicit_precomputed_quantities_core!`)
    when at least one Picard round runs, and clamp `Y.c.sgsʲs.:(j).ρa` to
    `[0, a_max ρʲ]` so the Picard and Newton predictors start from a feasible
    draft area. The grid-mean condensates are left at their stage-entering
    values: the grid-mean, draft, and environment thermodynamics and the 1M
    microphysics tendencies floor each condensate species at zero where they
    read it, whereas in a single-iteration Newton solve with an inexact
    Jacobian, raising the initial guess above `temp` enters `R(U^g)` as a source
    that the sedimentation Jacobian drains through the surface.

 2. Run `edmfx_model.updraft_picard_rounds` Picard rounds that alternate
    the analytic `u₃` solve, the two-pass area solve (`solve_sgs_ρa_two_pass!`),
    and the semi-implicit tracer sweep, each round starting from the clamped
    stage-initial `Y.(f,c).sgsʲs` (snapshotted into pre-allocated scratch).
    The sweep diagnoses the entrained environment from `ᶜsgsʲs_picard_snapshot`,
    transports every solved tracer, and, for the non-equilibrium microphysics,
    evaluates the draft thermodynamics and `ᶜmp_tendencyʲs` at the transported
    state, adds `dtγ` times the source to the condensate and precipitation
    tracers, and floors each draft condensate at zero before proportionally
    rescaling them so their sum does not exceed `q_totʲ`; the draft
    thermodynamics and buoyancy are refreshed again
    at the full iterate, and the precipitation terminal velocities are refreshed
    after the last round.

 3. Solve `u₃` and `ρa` (in two passes via `solve_sgs_ρa_two_pass!`)
    analytically with the stage-initial values as `*_old` and the
    Picard-converged buoyancy, microphysics, and sedimentation inputs, then
    complete the cached tendencies `(u₃_stage - u₃_old) / dtγ` and
    `(ρa_stage - ρa_old) / dtγ` in `p.precomputed.ᶠu₃_tendencyʲs` and
    `p.precomputed.ᶜρa_tendencyʲs`. The matching `sgs_*_implicit_tendency!`
    routines feed these to the Newton solve so it reproduces the analytic
    values. `ᶜmp_tendencyʲs` is held fixed through Newton so the Jacobian —
    which does not include its derivatives — stays consistent.

 4. With 1M microphysics and at least one Picard round, set the grid-mean
    `ρq_lcl`, `ρq_icl`, `ρq_rai`, `ρq_sno`, `ρq_tot`, and `ρe_tot` of the
    guess so that each environment specific content / mse keeps its
    stage-initial value (`warm_start_grid_mean_tracers!`).

Each Picard round ends by restoring the snapshot of the draft velocities and
with one column loop for every draft that applies the microphysics sources,
refreshes the draft thermodynamics and buoyancy, and restores the snapshot of
the draft scalars.
"""
function initialize_implicit_stage_problem!(Y, p, dtγ)
    reset_precip_stage_accumulator!(p)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing

    (; microphysics_model, turbconv_model, edmfx_model) = p.atmos
    (; ᶠu₃_tendencyʲs, ᶜρa_tendencyʲs) = p.precomputed
    (; ᶜρ_diffʲs, ᶜρʲs) = p.precomputed
    n = n_mass_flux_subdomains(turbconv_model)
    n_picard = edmfx_model.updraft_picard_rounds
    # Cast `dtγ` to the parameter float type so broadcasts stay type-stable
    # (ClimaTimeSteppers passes `dtγ` as `Float64` even when the state is
    # `Float32`).
    FT = eltype(p.params)
    dtγ = FT(dtγ)
    zero_FT = zero(FT)

    # `-u₃_old / dtγ` and `-ρa_old / dtγ` are recorded before the area clamp so
    # the Newton residual reproduces `u₃_stage` and `ρa_stage`. The clamp bounds
    # the predictor that `ᶜsnapshot` and the Picard solves start from, `ρa` to
    # `[0, a_max ρʲ]`. The implicit cache, including the draft densities
    # `ᶜρʲs`, is refreshed at the stage-entering state only for the Picard
    # rounds, which recompute the buoyancy from it. Without Picard rounds, the
    # `u₃` and `ρa` solves use the buoyancy `ᶜρ_diffʲs` and densities `ᶜρʲs` of
    # the last cache update, which the Newton iterations do not revise, so an
    # extrapolation error in a refreshed buoyancy would persist through the
    # stage; `ClimaTimeSteppers` refreshes `cache_imp!` before the Newton solve
    # (combining `updraft_picard_rounds == 0` with `update_cache_every = "step"`
    # holds `ᶜρ_diffʲs` at the step-start state while `cache_imp!` updates
    # `ᶜρʲs` after each stage; restarts rebuild the full cache).
    # `initialize_imp!` does not receive `t`, and `t` is unused in the call
    # below, so `zero(FT)` is passed.
    DataLayouts.foreach_point(ᶠu₃_tendencyʲs, Y.f) do ᶠu₃_tendencyʲs, Yf
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶠu₃_tendencyʲs.:($$j) = -Yf.sgsʲs.:($$j).u₃ / dtγ
        end
    end
    DataLayouts.foreach_point(ᶜρa_tendencyʲs, Y.c) do ᶜρa_tendencyʲs, Yc
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶜρa_tendencyʲs.:($$j) = -Yc.sgsʲs.:($$j).ρa / dtγ
        end
    end
    n_picard > 0 && set_implicit_precomputed_quantities_core!(Y, p, zero(FT))
    a_max = CAP.max_area(CAP.turbconv_params(p.params))
    for j in 1:n
        @. Y.c.sgsʲs.:($$j).ρa =
            clamp(Y.c.sgsʲs.:($$j).ρa, zero_FT, a_max * ᶜρʲs.:($$j))
    end

    if n_picard > 0
        ᶠsnapshot = p.precomputed.ᶠsgsʲs_picard_snapshot
        ᶜsnapshot = p.precomputed.ᶜsgsʲs_picard_snapshot
        DataLayouts.foreach_point(ᶠsnapshot, Y.f) do ᶠsnapshot, Yf
            for j in 1:n_mass_flux_subdomains(turbconv_model)
                @. ᶠsnapshot.:($$j) = Yf.sgsʲs.:($$j)
            end
        end
        snapshot_drafts = unrolled_map(StaticOneTo(n)) do j
            @inline
            (ᶜsnapshot.:($j), ᶜρ_diffʲs.:($j), Y.c.sgsʲs.:($j), ᶜρʲs.:($j))
        end
        DataLayouts.foreach_point(snapshot_drafts, Y.c.ρ) do snapshot_drafts, ᶜρ
            for (ᶜsnapshotʲ, ᶜρ_diffʲ, ᶜsgsʲ, ᶜρʲ) in snapshot_drafts
                @. ᶜsnapshotʲ = ᶜsgsʲ
                @. ᶜρ_diffʲ = (ᶜρʲ - ᶜρ) / ᶜρʲ
            end
        end
        (; ᶜmp_tendencyʲs, ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
        thermo_params = CAP.thermodynamics_params(p.params)
        T_min_sgs = CAP.T_min_sgs(p.params)
        ϵ = ϵ_numerics(FT)
        implicit_mp = microphysics_model isa NonEquilibriumMicrophysics1M
        nonequilibrium_mp =
            microphysics_model isa
            Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
        for round in 1:n_picard
            solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)
            solve_sgs_ρa_two_pass!(Y, p, dtγ, ᶜsnapshot)
            _sweep_sgs_updraft_tracers!(
                Y,
                p,
                dtγ,
                solved_sgs_tracer_candidates,
                ᶜsnapshot,
            )
            # The draft thermodynamics and `ᶜmp_tendencyʲs` at the transported
            # state. The 0-moment scheme has no prognostic condensate tracers,
            # so this is its full iterate, at which it evaluates the source.
            set_prognostic_edmf_precomputed_quantities_draft_thermo!(Y, p)
            update_updraft_microphysics_tendency_cache!(
                Y, p, microphysics_model, turbconv_model, dtγ,
            )
            # Restore the snapshot of the draft velocities, so that `u₃_old` of the
            # next `u₃` solve is the stage `u₃` (with the surface and top values
            # set by `set_implicit_precomputed_quantities_core!`).
            DataLayouts.foreach_point(ᶠsnapshot, Y.f) do ᶠsnapshot, Yf
                for j in 1:n_mass_flux_subdomains(turbconv_model)
                    @. Yf.sgsʲs.:($$j) = ᶠsnapshot.:($$j)
                end
            end
            # End the round in one column loop for every draft. For the non-equilibrium
            # microphysics, add the explicit and, for the 1M scheme, the implicit
            # sources to the condensate and precipitation tracers, floor each of them
            # at zero, and rescale them so that their sum does not exceed `q_totʲ`
            # (`add_updraft_microphysics_sources!`), and refresh the draft
            # thermodynamics at the full iterate. Then refresh `ᶜρ_diffʲs`, the
            # buoyancy input of the next `u₃` solve, and, between rounds, restore
            # the snapshot of the stage-initial draft scalars; the last round
            # leaves the Picard-converged draft scalars as a warm start for the
            # Newton solve.
            last_round = round == n_picard
            let dtγ = dtγ
                round_drafts = unrolled_map(StaticOneTo(n)) do j
                    @inline
                    (
                        Y.c.sgsʲs.:($j), ᶜρʲs.:($j), ᶜρ_diffʲs.:($j), ᶜsnapshot.:($j),
                        nonequilibrium_mp ?
                        (
                            ᶜmp_tendencyʲs.:($j),
                            implicit_mp ? p.precomputed.ᶜmp_tendency_implicitʲs.:($j) : (),
                            ᶜTʲs.:($j), ᶜq_tot_nonnegʲs.:($j), ᶜq_liqʲs.:($j),
                            ᶜq_iceʲs.:($j),
                        ) : (),
                    )
                end
                DataLayouts.foreach_column(
                    round_drafts, Y.c.ρ, p.core.ᶜΦ, p.precomputed.ᶜp,
                ) do round_drafts, ᶜρ, ᶜΦ, ᶜp
                    for (ᶜsgsʲ, ᶜρʲ, ᶜρ_diffʲ, ᶜsnapshotʲ, sources_args) in round_drafts
                        if !isempty(sources_args)
                            (
                                ᶜmp_tendencyʲ, ᶜmp_tendency_implicitʲ, ᶜTʲ, ᶜq_tot_nonnegʲ,
                                ᶜq_liqʲ, ᶜq_iceʲ,
                            ) = sources_args
                            add_updraft_microphysics_sources!(
                                ᶜsgsʲ, ᶜmp_tendencyʲ, ᶜmp_tendency_implicitʲ, dtγ, ϵ,
                            )
                            set_draft_thermo_column!(
                                ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ, ᶜρʲ, ᶜsgsʲ, ᶜΦ,
                                ᶜp, microphysics_model, thermo_params, T_min_sgs,
                            )
                        end
                        @. ᶜρ_diffʲ = (ᶜρʲ - ᶜρ) / ᶜρʲ
                        last_round || @. ᶜsgsʲ = ᶜsnapshotʲ
                    end
                end
            end
        end
        microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} &&
            set_precipitation_velocities!(Y, p, microphysics_model, turbconv_model)
    end

    solve_sgs_u₃_implicit_stage_analytic!(Y, p, dtγ)
    solve_sgs_ρa_two_pass!(
        Y,
        p,
        dtγ,
        n_picard > 0 ? p.precomputed.ᶜsgsʲs_picard_snapshot : nothing,
    )
    if n_picard > 0
        warm_start_grid_mean_tracers!(Y, p)
    end

    DataLayouts.foreach_point(ᶠu₃_tendencyʲs, Y.f) do ᶠu₃_tendencyʲs, Yf
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶠu₃_tendencyʲs.:($$j) += Yf.sgsʲs.:($$j).u₃ / dtγ
        end
    end
    DataLayouts.foreach_point(ᶜρa_tendencyʲs, Y.c) do ᶜρa_tendencyʲs, Yc
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            @. ᶜρa_tendencyʲs.:($$j) += Yc.sgsʲs.:($$j).ρa / dtγ
        end
    end
    return nothing
end

"""
    warm_start_grid_mean_tracers!(Y, p)

Move the grid-mean `ρq_χ` of the Newton initial guess, for each 1M condensate
tracer `χ ∈ {q_lcl, q_icl, q_rai, q_sno}` and for `ρq_tot` and `ρe_tot`, by the
Picard predictor's updraft increment, so that each environment specific
content (and `mse⁰`) keeps its stage-initial value:

    Δρq_χ   = (ρa⁰_new − ρa⁰_old) max(0, q_χ⁰_old)
            + Σⱼ (ρaʲ_new q_χʲ_new − ρaʲ_old q_χʲ_old),
    Δρq_tot = (ρa⁰_new − ρa⁰_old) max(0, q_tot⁰_old)
            + Σⱼ (ρaʲ_new q_totʲ_new − ρaʲ_old q_totʲ_old),
    Δρe_tot = (ρa⁰_new − ρa⁰_old) mse⁰_old
            + Σⱼ (ρaʲ_new mseʲ_new − ρaʲ_old mseʲ_old),

with `ρa⁰_new`, `ρaʲ_new`, and `χʲ_new` the Picard-converged draft state and the
`*_old` values diagnosed from `Y.c` and the stage-initial draft scalars in
`p.precomputed.ᶜsgsʲs_picard_snapshot`. The Picard predictor moves `χʲ` (by
transport, entrainment, and the explicit / implicit microphysics sources);
without this update the environment of the guess loses `Σⱼ ρaʲ Δχʲ / ρa⁰` and
goes negative where the environment is clear. For `q_lcl` / `q_icl`, the
environment source has zero derivative on that clipped branch; for `q_rai` /
`q_sno`, the same holds for the precipitation evaporation / sublimation branches.

The increment is the only difference between the guess and the stage-entering
state `temp`: where the environment value is negative, `max(0, ·)` scales the
area change by zero and the negative value persists, scaled by
`ρa⁰_old / ρa⁰_new`. In a single-iteration (truncated) Newton solve with an
inexact Jacobian, raising the guess above `temp` enters `R(U^g)` as a source
that the sedimentation Jacobian drains through the surface as an upward flux.
When the updraft increment vanishes, the guess equals `temp`, so `T_imp`
evaluates the vertical advection on the stage-entering profile (when the
updraft increment is nonzero, its central grid-mean vertical advection is not
compensated by the `1 + 2 C_γ` post-Newton dissipation weight).

Applies with `PrognosticEDMFX` 1M microphysics, regardless of which
implicit microphysics sources are on. Writes `p.scratch.ᶜtemp_scalar`.
"""
function warm_start_grid_mean_tracers!(Y, p)
    (; microphysics_model, turbconv_model) = p.atmos
    microphysics_model isa NonEquilibriumMicrophysics1M || return nothing
    ᶜsnapshot = p.precomputed.ᶜsgsʲs_picard_snapshot
    ᶜq⁰_old = p.scratch.ᶜtemp_scalar
    FT = eltype(p.params)
    zero_FT = zero(FT)

    condensate_names = (
        (@name(c.ρq_lcl), @name(q_lcl)),
        (@name(c.ρq_icl), @name(q_icl)),
        (@name(c.ρq_rai), @name(q_rai)),
        (@name(c.ρq_sno), @name(q_sno)),
    )
    MatrixFields.unrolled_foreach(condensate_names) do (gm_name, sgs_name)
        MatrixFields.has_field(Y, gm_name) || return nothing
        ρq_χ = MatrixFields.get_field(Y, gm_name)
        contrib(sgsʲ) = sgsʲ.ρa * MatrixFields.get_field(sgsʲ, sgs_name)
        @. ᶜq⁰_old = max(
            zero_FT,
            specific(
                env_value(ρq_χ, contrib, ᶜsnapshot),
                ρa⁰(Y.c.ρ, ᶜsnapshot, turbconv_model),
                ρq_χ,
                Y.c.ρ,
                turbconv_model,
            ),
        )
        @. ρq_χ +=
            (
                ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model) -
                ρa⁰(Y.c.ρ, ᶜsnapshot, turbconv_model)
            ) * ᶜq⁰_old +
            draft_sum(contrib, Y.c.sgsʲs) - draft_sum(contrib, ᶜsnapshot)
        return nothing
    end

    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        ρq_tot = Y.c.ρq_tot
        contrib_qt(sgsʲ) = sgsʲ.ρa * sgsʲ.q_tot
        ᶜρa⁰_old = @. lazy(ρa⁰(Y.c.ρ, ᶜsnapshot, turbconv_model))
        @. ᶜq⁰_old = max(
            zero_FT,
            specific(
                env_value(ρq_tot, contrib_qt, ᶜsnapshot),
                ᶜρa⁰_old,
                ρq_tot,
                Y.c.ρ,
                turbconv_model,
            ),
        )
        @. ρq_tot +=
            (ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model) - ᶜρa⁰_old) * ᶜq⁰_old +
            draft_sum(contrib_qt, Y.c.sgsʲs) -
            draft_sum(contrib_qt, ᶜsnapshot)
    end

    # Energy warm-start. The EDMFX decomposition
    #   ρe_tot = ρa⁰·mse⁰ + Σⱼ ρaʲ·mseʲ + (ρK − p)
    # follows from `ρ·mse_bar = ρ·(h_tot − K)` and `ρe_tot = ρ·h_tot − p`.
    # Holding `mse⁰` and `(ρK − p)` at their stage-initial values while
    # swapping in the predictor's updated `ρaʲ`, `mseʲ`, `ρa⁰` gives
    #   Δρe_tot = (ρa⁰_new − ρa⁰_old)·mse⁰_old
    #           + Σⱼ (ρaʲ_new·mseʲ_new − ρaʲ_old·mseʲ_old).
    if MatrixFields.has_field(Y, @name(c.ρe_tot))
        (; ᶜh_tot, ᶜK) = p.precomputed
        ᶜmse⁰_old = p.scratch.ᶜtemp_scalar
        ᶜρa⁰_old = @. lazy(ρa⁰(Y.c.ρ, ᶜsnapshot, turbconv_model))
        @. ᶜmse⁰_old =
            specific_env_mse(Y.c.ρ, ᶜsnapshot, ᶜh_tot, ᶜK, ᶜρa⁰_old, turbconv_model)
        @. Y.c.ρe_tot +=
            (ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model) - ᶜρa⁰_old) * ᶜmse⁰_old +
            draft_sum(sgsʲ -> sgsʲ.ρa * sgsʲ.mse, Y.c.sgsʲs) -
            draft_sum(sgsʲ -> sgsʲ.ρa * sgsʲ.mse, ᶜsnapshot)
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
    drafts = unrolled_map(StaticOneTo(n)) do j
        @inline
        (Yₜ.f.sgsʲs.:($j).u₃, ᶠu₃_tendencyʲs.:($j))
    end
    DataLayouts.foreach_point(drafts) do drafts
        for (ᶠu₃ₜʲ, ᶠu₃_tendencyʲ) in drafts
            @. ᶠu₃ₜʲ = ᶠu₃_tendencyʲ
        end
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
            ᶜentr_vel_scaleʲs.:($j), ᶜarea_bounding_entr_detrʲs.:($j),
            ᶜentr_nonvel_rateʲs.:($j), ᶜturb_entrʲs.:($j),
        )
        DataLayouts.foreach_column(
            ᶠB, ᶠK, Y.c.sgsʲs.:($j).ρa, ᶜρʲs.:($j), Y.c.sgsʲs, ᶜρʲs,
            Y.f.sgsʲs.:($j).u₃, rates, ᶜρ_diffʲs.:($j), ᶠgradᵥ_ᶜΦ, ᶠdz, ᶠz,
        ) do ᶠB, ᶠK, ᶜρaʲ, ᶜρʲ, ᶜsgsʲs, ᶜρʲs, ᶠu₃ʲ, rates, ᶜρ_diffʲ, ᶠgradᵥ_ᶜΦ,
        ᶠdz, ᶠz
            @inline
            (ᶜentr_vel_scaleʲ, ᶜarea_bounding_entr_detrʲ, ᶜentr_nonvel_rateʲ, ᶜturb_entrʲ) =
                rates
            # Approximation factor used in w₀ - w ≈ -(1 / a⁰) w (single-updraft
            # case). Guard against unphysical extrapolated areas (ρa is solved
            # analytically AFTER u₃, so the u₃ solve sees a possibly
            # out-of-range ρa). Clamp a⁰ ∈ [1−a_max, 1].
            ᶜa⁰ = @. clamp(a⁰(ᶜsgsʲs, ᶜρʲs, turbconv_model), 1 - a_max, one_FT)
            ᶜa⁰_inv = @. one_FT / ᶜa⁰

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
            # Implicit NH pressure drag a⁰ d_c (w - w₀)|w - w₀| ≈ d_c w² / a⁰
            # contributes a quadratic sink: two factors of 1/a⁰ from the velocity
            # differences, one cancelled by the environment weight a⁰ of the
            # pairwise form. The clamp inside the coefficient is idempotent. (The
            # flag is tested at each point, so that the loop body has one type.)
            ᶜquadratic_rate = @. (
                ᶜentr_vel_scaleʲ * ᶜa⁰_inv + ifelse(
                    nh_pressure,
                    pressure_drag_coefficient(
                        α_d, scale_height, draft_area(ᶜρaʲ, ᶜρʲ), ᶜa⁰, a_min, a_max,
                    ),
                    zero_FT,
                )
            ) * ᶜa⁰_inv
            ᶜlinear_rate = @. (
                max(zero_FT, ᶜarea_bounding_entr_detrʲ) + ᶜentr_nonvel_rateʲ +
                ᶜturb_entrʲ
            ) * ᶜa⁰_inv

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
                    (1 - α_b) * ᶠinterp(ᶜρ_diffʲ) * ᶠgradᵥ_ᶜΦ.components.data.:1
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
    drafts = unrolled_map(StaticOneTo(n)) do j
        @inline
        (Yₜ.c.sgsʲs.:($j).ρa, ᶜρa_tendencyʲs.:($j))
    end
    DataLayouts.foreach_point(drafts) do drafts
        for (ᶜρaₜʲ, ᶜρa_tendencyʲ) in drafts
            @. ᶜρaₜʲ = ᶜρa_tendencyʲ
        end
    end
end

"""
    solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ, ᶜsnapshot = Y.c.sgsʲs)

Solve the IMEX/ARK implicit-stage equation for the updraft area-weighted
density `ρa` analytically, overwriting `Y.c.sgsʲs.:(j).ρa` in each EDMFX
mass-flux subdomain `j`.

Returns early for any `turbconv_model` other than `PrognosticEDMFX`; see
`initialize_implicit_stage_problem!` for how the result is held fixed during
the Newton solve.

The stage-start area `ᶜρa_old` is read from `ᶜsnapshot`: when `ᶜsnapshot` is
`nothing`, `-dtγ * p.precomputed.ᶜρa_tendencyʲs.:(j)` clamped to
`[0, a_max ρʲ]`; otherwise `ᶜsnapshot.:(j).ρa` (defaulting to the entering
`Y.c.sgsʲs`). The flux-form stage equation `∂ρa/∂t + ∂(ρa·w)/∂z = (ε − δ)·ρa`
reduces under first-order upwinding (for upward `ᶠu₃ʲ`) to the forward
recurrence

    ρa_new[i] = (numerator[i] + α_bot[i] · ρa_new[i−1]) / denominator[i],
    numerator[i] = max(0, ρa_old[i]) / dtγ,
    denominator[i] = max(0.1/dtγ, 1/dtγ + α_top[i] − (ε − δ)[i] − r_mass[i]),
    α_face = (ᶠinterp(ρʲ·J)/ᶠJ · ᶠu₃ʲ/Δz_face) / (ρʲ_upwind · Δz[i]),

evaluated bottom-to-top with `Operators.column_accumulate!`. The floor on the
denominator caps the per-step growth at roughly a factor of ten.

`r_mass` is the mass source rate of the updraft condensate. Under the
non-equilibrium microphysics, it is the sedimentation rate from
`updraft_sedimentation_rate!`, `Σ_μ Δq_μ / dtγ` with `Δq_μ` the change of the
updraft condensate by sedimentation in one linearized backward-Euler step from
the current updraft tracers and terminal velocities (stage-entering values
during the Picard rounds, and Picard-converged tracers and refreshed terminal
velocities in the final post-Picard solve), evaluated at the current
`Y.c.sgsʲs.:(j).ρa` (`ρa_old` in pass 1, `ρa_pass1` in pass 2) and grid-mean
state. The Newton solve takes the same step for the condensate, so
`r_mass ρ̂` matches the sum `S_sed` of the condensate sedimentation tendencies
in the dilution `χ S_sed ρ̂⁻¹` that it applies, also at Courant numbers above
one, where the condensate arriving over the stage differs from the stage-start
flux convergence. Where the draft narrows downward, `S_sed` and its derivative
are proportional to the local area and `Δq_μ` does not depend on it, so the
rate also holds at the stage area. Under the 0-moment microphysics with
implicit microphysics tendencies, `r_mass` is the updraft precipitation removal
rate `ᶜmp_tendencyʲs.dq_tot_dt`, the rate at which the Newton solve removes
`q_totʲ` (with explicit tendencies, the removal enters `ρaʲ` through the
explicit tendency, and `r_mass` is zero). The stage source is
`r_mass · ρa_new`, so `ρaʲ` receives the mass that `q_totʲ` receives, as the
grid-mean `ρ` does.

`(ε − δ)` is assembled inline from area-bounding, velocity-scale, and
buoyancy-driven pieces (see `detr_buoy_inv_time_scale`). The
mass-flux-divergence component of detrainment is not part of `(ε − δ)`: it
becomes a multiplicative prefactor on both `α_top` and `α_bot`, so that it is
treated implicitly together with the flux divergence. In the first cell,
`α_bot` is zeroed (the surface flux vanishes because `u₃ = 0` there, and the
upwind density would otherwise read an undefined ghost cell), and the capped
surface mass-flux source `F_sfc/Δz` from `p.precomputed.sfc_mass_flux_sourceʲs`
is added to the numerator as an area-independent constant.

The lower and upper area-limiter factors (which gate buoyancy detrainment and
the mass-flux-divergence detrainment prefactor) and the mass-flux convergence
sign test are evaluated at `ρa_eff = max(ρa_old, ρa)` (so pass 2 of
`solve_sgs_ρa_two_pass!` sees `max(ρa_old, ρa_pass1)` where an updraft advances
into cells with `ρa_old < ρʲ a_min`), which cannot guarantee `a ∈ [0, a_max]` at
the implicit stage value. The sweep therefore clamps `ρa ∈ [0, ρʲ·a_max]`
per-cell, using the updraft density `ρʲ` so that the cap acts on the draft area
`a = ρa/ρʲ`. Clipping from above returns the excess `ρa` to the environment at
fixed grid-mean `ρ`, acting like instantaneous detrainment; clipping from below
at `ρʲ·a_min` would draw mass from the environment into an empty draft without
an entrainment source, so the lower bound is set to zero instead.
"""
function solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ, ᶜsnapshot = Y.c.sgsʲs)

    p.atmos.turbconv_model isa PrognosticEDMFX || return

    (; turbconv_model) = p.atmos
    (; ᶜρ_diffʲs, ᶜρʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜentr_vel_scaleʲs, ᶜentr_nonvel_rateʲs, ᶜarea_bounding_entr_detrʲs) =
        p.precomputed
    FT = eltype(p.params)
    (zero_FT, one_FT, point_one) = (zero(FT), one(FT), FT(0.1))

    turbconv_params = CAP.turbconv_params(p.params)
    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶠdz = Fields.Δz_field(axes(Y.f))
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    entr_detr_buoy_inv_tau_max = CAP.entr_detr_buoy_inv_tau_max(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)
    a_max = CAP.max_area(turbconv_params)

    # Cell-centred coefficients of the recurrence, computed in one column loop
    # per draft:
    #   numerator            = max(0, ρa_old[i]) / dtγ
    #   denominator          = 1/dtγ + α_top · (1 − implicit_detr_prefactor) − (ε − δ)
    #                          − r_mass
    #   mass_flux_factor_bot = α_bot · (1 − implicit_detr_prefactor)
    # where α_face = (ᶠinterp(ρʲ·J)/ᶠJ · ᶠu₃ʲ/Δz_face) / (ρʲ_upwind · Δz).
    # For upward flow the upwind density at the bottom face is ρʲ[i−1], which we
    # extract via `ᶠbottom_bias(ᶜρʲs)`. The mass-flux-divergence component of the
    # detrainment is folded into `ᶜone_minus_implicit_detr_prefactor` (a
    # multiplicative correction on the implicit advection term), leaving the
    # `(ε − δ)` term to carry only the area-bounding, velocity-scale
    # entrainment, and buoyancy-based detrainment pieces. The sweep reads the
    # offset `numerator / denominator` and the factor
    # `mass_flux_factor_bot / denominator` of the recurrence.
    ᶜoffset = p.scratch.ᶜtemp_scalar
    ᶜfactor = p.scratch.ᶜtemp_scalar_2
    ᶜmass_source_rate = p.scratch.ᶜtemp_scalar_5
    nonequilibrium_mp =
        p.atmos.microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
    implicit_0m_mp =
        p.atmos.microphysics_model isa EquilibriumMicrophysics0M

    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        # The mass source rate r_mass (`()` for none, which the loop reads as
        # zero).
        nonequilibrium_mp &&
            updraft_sedimentation_rate!(ᶜmass_source_rate, Y, p, j, dtγ)
        ᶜr_mass =
            nonequilibrium_mp ? ᶜmass_source_rate :
            implicit_0m_mp ? p.precomputed.ᶜmp_tendencyʲs.:($j).dq_tot_dt : ()
        rates = (
            ᶜarea_bounding_entr_detrʲs.:($j), ᶜentr_vel_scaleʲs.:($j),
            ᶜentr_nonvel_rateʲs.:($j),
        )
        # The source of the stage-start area `ρa_old`: the area tendency, which
        # the loop multiplies by `-dtγ` and clamps to `[0, a_max ρʲ]`, when
        # `ᶜsnapshot` is `nothing`, and otherwise the snapshot area (`()` for the
        # one that is not read).
        ρa_old_sources =
            isnothing(ᶜsnapshot) ? (p.precomputed.ᶜρa_tendencyʲs.:($j), ()) :
            ((), ᶜsnapshot.:($j).ρa)
        DataLayouts.foreach_column(
            ᶜoffset, ᶜfactor, Y.c.sgsʲs.:($j).ρa, ρa_old_sources,
            Y.f.sgsʲs.:($j).u₃, ᶜρʲs.:($j), ᶜρ_diffʲs.:($j), rates, ᶜr_mass,
            p.precomputed.sfc_mass_flux_sourceʲs.:($j), ᶠgradᵥ_ᶜΦ, ᶜdz, ᶠdz,
        ) do ᶜoffset, ᶜfactor, ᶜρaʲ, (ᶜρa_tendencyʲ, ᶜρa_snapshotʲ), ᶠu₃ʲ, ᶜρʲ,
        ᶜρ_diffʲ, rates, ᶜr_mass_column, mass_flux_source, ᶠgradᵥ_ᶜΦ, ᶜdz, ᶠdz
            (ᶜarea_bounding_entr_detrʲ, ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ) = rates
            ᶜr_massʲ = ᶜr_mass_column === () ? zero_FT : ᶜr_mass_column
            ᶜJ = Fields.local_geometry_field(axes(ᶜρaʲ)).J
            ᶠlg = Fields.local_geometry_field(axes(ᶠu₃ʲ))
            ᶠJ = ᶠlg.J
            ᶠw = @. ᶠu₃ʲ.components.data.:1 / ᶠdz
            ᶜρa_old =
                ᶜρa_snapshotʲ === () ?
                (@. clamp(-dtγ * ᶜρa_tendencyʲ, zero_FT, a_max * ᶜρʲ)) :
                ᶜρa_snapshotʲ

            # entr and detr, with the area limiters and the convergence test
            # evaluated at `ρa_eff = max(ρa_old, ρa)`
            ᶜρa_eff = @. max(ᶜρa_old, ᶜρaʲ)
            ᶜaʲ = @. draft_area(ᶜρa_eff, ᶜρʲ)
            ᶜlower_limiter_factor = @. lower_area_limiter_factor(ᶜaʲ, turbconv_params)
            ᶜupper_limiter_factor = @. upper_area_limiter_factor(ᶜaʲ, turbconv_params)
            # Mass-flux-divergence detrainment fraction:
            #   detr_coeff = 1 − U · (1 − L · C)
            # with U = upper limiter (→ 0 as a → a_max), L = lower limiter
            # (→ 0 as a → a_min), C = detr_massflux_vertdiv_coeff.
            # At U = 0 (a → a_max): detr_coeff = 1 — all converging mass is
            # detrained, capping area at a_max regardless of C.
            # The recurrence uses `one_minus_prefactor = U · (1 − L · C)`.
            ᶜone_minus_implicit_detr_prefactor = @. ifelse(
                ᶜdivᵥ(ᶠbottom_bias(ᶜρa_eff) * ᶠu₃ʲ) < 0,
                ᶜupper_limiter_factor *
                (one_FT - ᶜlower_limiter_factor * detr_massflux_vertdiv_coeff),
                one_FT,
            )
            # Inverse buoyancy time-scale on faces (where w lives), interpolated
            # to centers for a smoother (ε − δ).
            ᶜbuoy_inv_time_scale = @. ᶜinterp(
                detr_buoy_inv_time_scale(
                    ᶠw,
                    vertical_buoyancy_acceleration(ᶠinterp(ᶜρ_diffʲ), ᶠgradᵥ_ᶜΦ, ᶠlg),
                    entr_detr_buoy_inv_tau_max,
                ),
            )
            ᶜexplicit_entr_minus_detr = @. ᶜarea_bounding_entr_detrʲ +
               ᶜentr_vel_scaleʲ * ᶜinterp(ᶠw) +
               ᶜentr_nonvel_rateʲ -
               ᶜlower_limiter_factor * detr_buoy_coeff * ᶜbuoy_inv_time_scale

            # Draft mass flux through each face, ᶠinterp(ρʲ·J)/ᶠJ · ᶠu₃ʲ/Δz_face
            ᶠmass_flux = @. ᶠinterp(ᶜρʲ * ᶜJ) / ᶠJ * ᶠw
            # Floor at 0.1/dtγ ⇒ (ε − δ) + r_mass ≤ 0.9/dtγ
            # (≈10× per-step growth cap).
            ᶜdenominator = @inline @. lazy(
                max(
                    point_one / dtγ,
                    1 / dtγ - ᶜexplicit_entr_minus_detr - ᶜr_massʲ +
                    ᶜone_minus_implicit_detr_prefactor * ᶜtop_bias(ᶠmass_flux) / ᶜρʲ /
                    ᶜdz,
                ),
            )
            # Cell 1: α_bot[1] is 0, which bypasses the NaN from
            # `ᶠbottom_bias(ᶜρʲ)` reading the undefined ghost cell below
            # (physical flux is zero there: u₃ = 0 at the surface), and the
            # capped volumetric surface mass-flux source `F_sfc/dz` (an
            # a-independent constant, precomputed) is added to the numerator.
            ᶜzero_first_cell =
                Operators.SetBoundaryOperator(bottom = Operators.SetValue(zero_FT))
            ᶜone_first_cell =
                Operators.SetBoundaryOperator(bottom = Operators.SetValue(one_FT))
            @. ᶜoffset =
                (
                    max(zero_FT, ᶜρa_old) / dtγ +
                    ᶜone_first_cell(zero_FT * ᶜdz) * mass_flux_source
                ) / ᶜdenominator
            @. ᶜfactor =
                ᶜzero_first_cell(
                    ᶜone_minus_implicit_detr_prefactor *
                    ᶜbottom_bias(ᶠmass_flux / ᶠbottom_bias(ᶜρʲ)) / ᶜdz,
                ) / ᶜdenominator
        end

        # ρʲ is in the input tuple so the per-cell cap `ρʲ·a_max` is available
        # inside the closure.
        input = @. lazy(tuple(ᶜoffset, ᶜfactor, ᶜρʲs.:($$j)))

        # Bottom-to-top sweep. `clamp` to `ρa ∈ [0, ρʲ·a_max]` per-cell.
        Operators.column_accumulate!(
            Y.c.sgsʲs.:($j).ρa,
            input;
            init = FT(0),
        ) do ρa_prev, (offset, factor, ρ_cell)
            return clamp(offset + factor * ρa_prev, zero(ρa_prev), a_max * ρ_cell)
        end
    end
end

"""
    solve_sgs_ρa_two_pass!(Y, p, dtγ, ᶜsnapshot = nothing)

Run two passes of `solve_sgs_ρa_implicit_stage_analytic!` starting from the
clamped stage-initial area (`ᶜsnapshot.:(j).ρa`, or `-dtγ * ᶜρa_tendencyʲs`
clamped to `[0, a_max ρʲ]` when `ᶜsnapshot` is `nothing`), so that the area
limiters and buoyancy detrainment gate in the second pass see
`max(ρa_old, ρa_pass1)` where an updraft advances into cells with
`ρa_old < ρʲ a_min` over the implicit stage.
"""
function solve_sgs_ρa_two_pass!(
    Y,
    p,
    dtγ,
    ᶜsnapshot = nothing,
)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing
    FT = eltype(p.params)
    a_max = CAP.max_area(CAP.turbconv_params(p.params))
    (; ᶜρʲs) = p.precomputed
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    if isnothing(ᶜsnapshot)
        (; ᶜρa_tendencyʲs) = p.precomputed
        for j in 1:n
            @. Y.c.sgsʲs.:($$j).ρa = clamp(
                -dtγ * ᶜρa_tendencyʲs.:($$j),
                zero(FT),
                a_max * ᶜρʲs.:($$j),
            )
        end
    else
        for j in 1:n
            @. Y.c.sgsʲs.:($$j).ρa = ᶜsnapshot.:($$j).ρa
        end
    end
    solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ, ᶜsnapshot)
    solve_sgs_ρa_implicit_stage_analytic!(Y, p, dtγ, ᶜsnapshot)
    return nothing
end

"""
    solve_sgs_updraft_tracers_implicit_stage_analytic!(
        Y, p, dtγ, ᶜsnapshot = nothing,
    )

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
tendency sees once `ᶜuʲs` is refreshed from the stage `u₃`. The entrained
value `χ⁰` and the stage-initial `χ` of every tracer other than `mse` enter
the recurrence clamped at zero, so a negative diagnosed environment residual
cannot pull the draft tracer below zero. When `ᶜsnapshot` is passed, `χ⁰` is
diagnosed from `Y.c` and the stage-entering draft state in `ᶜsnapshot` so
that updating `Y.c.sgsʲs.:(j).ρa` in `solve_sgs_ρa_two_pass!` stays
consistent with the stage-entering grid-mean `Y.c.ρχ` and `Y.c.ρe_tot`; the
`ᶜsnapshot === nothing` fallback diagnoses `χ⁰` from the current `Y.c.sgsʲs`
for 3-argument test calls.

For the non-equilibrium microphysics, the source follows the transport:
`set_prognostic_edmf_precomputed_quantities_draft_thermo!` and
`update_updraft_microphysics_tendency_cache!` evaluate the draft
thermodynamics and `ᶜmp_tendencyʲs` at the transported state, and each
condensate and precipitation tracer then receives `dtγ · S_mp`, is floored at
zero, and is proportionally rescaled (`add_updraft_microphysics_sources!`) so
their sum does not exceed `q_totʲ`. Evaluating the source after the transport
keeps the vapor of each cell consistent with the condensate that arrives with
it, so water that arrives as condensate is not condensed a second time. The
post-source floor and proportional rescale bound the predictor and Newton
initial guess (and the draft thermodynamics evaluated on it), while
`ᶜmp_tendencyʲs` retains the un-clamped physical rate so the Newton solve
applies `transported + dtγ · S_mp` consistently across both the draft and the
area-weighted grid mean.

The Picard loop of `initialize_implicit_stage_problem!` takes the same steps as
a predictor between successive `u₃` solves, with the sources applied at the
end of each Picard round. The final analytic `u₃` and `ρa`
solves at the end of `initialize_implicit_stage_problem!` evaluate buoyancy and
`ᶜmp_tendencyʲs` at the final predictor iterate, which is also the initial
guess for the Newton solve. Detrainment does not enter the intensive tracer
equation because it removes updraft mass at the updraft's own scalar value.
"""
function solve_sgs_updraft_tracers_implicit_stage_analytic!(
    Y,
    p,
    dtγ,
    ᶜsnapshot = nothing,
)
    p.atmos.turbconv_model isa PrognosticEDMFX || return
    (; microphysics_model, turbconv_model) = p.atmos
    _sweep_sgs_updraft_tracers!(
        Y,
        p,
        dtγ,
        solved_sgs_tracer_candidates,
        ᶜsnapshot,
    )

    microphysics_model isa
    Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ||
        return nothing
    set_prognostic_edmf_precomputed_quantities_draft_thermo!(Y, p)
    update_updraft_microphysics_tendency_cache!(
        Y, p, microphysics_model, turbconv_model, dtγ,
    )
    (; ᶜmp_tendencyʲs) = p.precomputed
    implicit_mp = microphysics_model isa NonEquilibriumMicrophysics1M
    ϵ = ϵ_numerics(eltype(p.params))
    drafts = unrolled_map(StaticOneTo(n_mass_flux_subdomains(turbconv_model))) do j
        @inline
        (
            Y.c.sgsʲs.:($j), ᶜmp_tendencyʲs.:($j),
            implicit_mp ? p.precomputed.ᶜmp_tendency_implicitʲs.:($j) : (),
        )
    end
    DataLayouts.foreach_column(drafts) do drafts
        for (ᶜsgsʲ, ᶜmp_tendencyʲ, ᶜmp_tendency_implicitʲ) in drafts
            add_updraft_microphysics_sources!(
                ᶜsgsʲ, ᶜmp_tendencyʲ, ᶜmp_tendency_implicitʲ, dtγ, ϵ,
            )
        end
    end
    return nothing
end

"""
    add_updraft_microphysics_sources!(
        ᶜsgsʲ, ᶜmp_tendencyʲ, ᶜmp_tendency_implicitʲ, dtγ, ϵ,
    )

Column-loop body: add `dtγ` times the microphysics sources of the draft `ᶜsgsʲ`,
evaluated at the draft thermodynamics of the transported state, to each
condensate and precipitation tracer: the explicit source `ᶜmp_tendencyʲ` and,
for the 1M scheme, the implicit source `ᶜmp_tendency_implicitʲ` (`()`
otherwise). Each of these tracers is then floored at zero, and the four
condensates are multiplied by
`min(1, q_totʲ / (q_lclʲ + q_iclʲ + q_raiʲ + q_snoʲ))`, or by zero where the
condensate sum or `q_totʲ` is below `ϵ` (`ϵ_numerics`), so that their sum does
not exceed `q_totʲ`.
"""
@inline function add_updraft_microphysics_sources!(
    ᶜsgsʲ, ᶜmp_tendencyʲ, ᶜmp_tendency_implicitʲ, dtγ, ϵ,
)
    zero_FT = zero(ϵ)
    # (χ_name, S_name): a tracer and its entry in `ᶜmp_tendencyʲs` /
    # `ᶜmp_tendency_implicitʲs`; a pair with either name missing is skipped.
    # Reading the implicit tendency from CM BMT (substep-averaged) directly
    # subsumes the earlier manual τ·(1 − exp(−dtγ/τ))·S damping and the
    # single-shot backward-Euler draft increment `dtγ·S / (1 + dtγ/τ)`.
    condensate_sources = (
        (@name(q_lcl), @name(dq_lcl_dt)),
        (@name(q_icl), @name(dq_icl_dt)),
        (@name(q_rai), @name(dq_rai_dt)),
        (@name(q_sno), @name(dq_sno_dt)),
    )
    MatrixFields.unrolled_foreach(condensate_sources) do (χ_name, S_name)
        @inline
        MatrixFields.has_field(ᶜsgsʲ, χ_name) &&
        MatrixFields.has_field(ᶜmp_tendencyʲ, S_name) || return nothing
        χ = MatrixFields.get_field(ᶜsgsʲ, χ_name)
        S = MatrixFields.get_field(ᶜmp_tendencyʲ, S_name)
        if ᶜmp_tendency_implicitʲ === ()
            @. χ = max(zero_FT, χ + dtγ * S)
        else
            S_i = MatrixFields.get_field(ᶜmp_tendency_implicitʲ, S_name)
            @. χ = max(zero_FT, χ + dtγ * S + dtγ * S_i)
        end
    end
    MatrixFields.has_field(ᶜsgsʲ, @name(q_lcl)) &&
    MatrixFields.has_field(ᶜsgsʲ, @name(q_icl)) &&
    MatrixFields.has_field(ᶜsgsʲ, @name(q_rai)) &&
    MatrixFields.has_field(ᶜsgsʲ, @name(q_sno)) || return nothing
    ᶜq_cond = @. ᶜsgsʲ.q_lcl + ᶜsgsʲ.q_icl + ᶜsgsʲ.q_rai + ᶜsgsʲ.q_sno
    ᶜratio = @. condensate_rescale_ratio(ᶜq_cond, ᶜsgsʲ.q_tot, ϵ)
    @. ᶜsgsʲ.q_lcl *= ᶜratio
    @. ᶜsgsʲ.q_icl *= ᶜratio
    @. ᶜsgsʲ.q_rai *= ᶜratio
    @. ᶜsgsʲ.q_sno *= ᶜratio
    return nothing
end

# The draft scalars that the tracer predictor solves; `_sweep_sgs_updraft_tracers!`
# skips the names missing from the SGS schema (the condensates under 0M).
const solved_sgs_tracer_candidates =
    (@name(mse), @name(q_tot), @name(q_lcl), @name(q_icl), @name(q_rai), @name(q_sno))

"""
    _sweep_sgs_updraft_tracers!(Y, p, dtγ, sgs_tracers, ᶜsnapshot = nothing)

Run the transport sweep of `solve_sgs_updraft_tracers_implicit_stage_analytic!`
for every draft and every tracer name in `sgs_tracers`, skipping any name
missing from the SGS schema, with the recurrence of each solved scalar written
as

    χ_new[i] = b[i] + a[i] · χ_new[i-1],    a = α / den,    b = num / den,

where `num` and `den` are the numerator and denominator of the recurrence
(including the surface relaxation in cell 1, and `a[1] = 0`). One column loop
computes the entrainment rate, `α`, and `den`, writes `a`
to `p.scratch.ᶜtemp_scalar`, and overwrites each solved scalar with its `b`,
and each scalar is then swept on its own, which reads two values per level.
The environment values `χ⁰` are diagnosed from the draft states in `ᶜsnapshot`,
or from `Y.c.sgsʲs` when it is `nothing`.
"""
function _sweep_sgs_updraft_tracers!(
    Y,
    p,
    dtγ,
    sgs_tracers,
    ᶜsnapshot = nothing,
)
    (; turbconv_model) = p.atmos
    (; ᶜK, ᶜh_tot, ᶜρʲs, ᶜentr_vel_scaleʲs, ᶜentr_nonvel_rateʲs) = p.precomputed
    (; ᶜarea_bounding_entr_detrʲs, ᶜturb_entrʲs) = p.precomputed
    (; sfc_mass_flux_sourceʲs, sfc_mse_buoyantʲs, sfc_q_tot_buoyantʲs) = p.precomputed
    ᶜadvection_factor = p.scratch.ᶜtemp_scalar
    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶠdz = Fields.Δz_field(axes(Y.f))
    FT = eltype(p.params)
    (zero_FT, one_FT) = (zero(FT), one(FT))
    a_min = CAP.min_area(CAP.turbconv_params(p.params))
    χ_names = unrolled_filter(
        χ_name -> MatrixFields.has_field(Y.c.sgsʲs.:(1), χ_name),
        sgs_tracers,
    )
    ᶜenv_sgsʲs = isnothing(ᶜsnapshot) ? Y.c.sgsʲs : ᶜsnapshot
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        rates = (
            ᶜentr_vel_scaleʲs.:($j), ᶜentr_nonvel_rateʲs.:($j),
            ᶜarea_bounding_entr_detrʲs.:($j), ᶜturb_entrʲs.:($j),
        )
        sfc = (
            sfc_mass_flux_sourceʲs.:($j), sfc_mse_buoyantʲs.:($j),
            sfc_q_tot_buoyantʲs.:($j),
        )
        DataLayouts.foreach_column(
            Y.c.sgsʲs.:($j), Y.c, ᶜenv_sgsʲs, Y.f.sgsʲs.:($j).u₃, ᶜρʲs.:($j),
            ᶜadvection_factor, ᶜdz, ᶠdz, ᶜK, ᶜh_tot, rates, sfc,
        ) do ᶜsgsʲ, Yc, ᶜenv_sgsʲs, ᶠu₃ʲ, ᶜρʲ, ᶜadvection_factor, ᶜdz, ᶠdz, ᶜK,
        ᶜh_tot, rates, (mass_flux_source, mse_sfc, q_tot_sfc)
            (ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ, ᶜarea_bounding_entr_detrʲ, ᶜturb_entrʲ) =
                rates
            ᶠw = @. ᶠu₃ʲ.components.data.:1 / ᶠdz
            # Cell 1: no flow through the surface, whatever the covariant `u₃`
            # there, so α[1] = 0. The surface relaxation rate `r_sfc` acts only
            # in cell 1.
            ᶜzero_first_cell =
                Operators.SetBoundaryOperator(bottom = Operators.SetValue(zero_FT))
            ᶜone_first_cell =
                Operators.SetBoundaryOperator(bottom = Operators.SetValue(one_FT))
            ᶜα = @. ᶜzero_first_cell(ᶜbottom_bias(ᶠw) / ᶜdz)
            ᶜr_sfc = @. ᶜone_first_cell(zero_FT * ᶜdz) *
               edmfx_sfc_relaxation_rate(mass_flux_source, ᶜsgsʲ.ρa, ᶜρʲ, a_min)
            ᶜentr_tot = @. compute_entrainment(
                ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ, ᶜarea_bounding_entr_detrʲ,
                ᶜinterp(ᶠw),
            ) + ᶜturb_entrʲ
            ᶜden = @. 1 / dtγ + ᶜα + ᶜentr_tot
            @. ᶜadvection_factor = ᶜα / ᶜden
            # The grid-mean density of each solved scalar (for mse, ρ (h_tot - K))
            ᶜρχs = unrolled_map(χ_names) do χ_name
                χ_name === @name(mse) ? (@inline @. lazy(Yc.ρ * (ᶜh_tot - ᶜK))) :
                MatrixFields.get_field(Yc, get_ρχ_name(χ_name))
            end
            # The state of draft `j` at each point, with each solved scalar replaced by the
            # offset `b = num / den` of its recurrence, with `num = χ_old / dtγ + ε_tot χ⁰`,
            # where the environment value `χ⁰` is diagnosed from the grid-mean densities
            # `ρχs` of the solved scalars and the draft states `env_sgsʲs` (as by
            # `specific_env_value` and `specific_env_mse`), and where `χ_old` and `χ⁰` of
            # every scalar other than `mse` are floored at zero. `mse` and `q_tot` also
            # relax toward their buoyant surface values at the rate `r_sfc`, which is zero
            # above cell 1.
            ᶜsgsʲ .=
                (@inline (
                    sgsʲs,
                    env_sgsʲs,
                    ρ,
                    ρχs,
                    entr_tot,
                    den,
                    r_sfc,
                    mse_sfc,
                    q_tot_sfc,
                ) -> begin
                    sgsʲ = drop_auto_broadcasters(sgsʲs)[j]
                    ρa⁰_val = ρa⁰(ρ, env_sgsʲs, turbconv_model)
                    names_and_ρχs =
                        unrolled_map(tuple, χ_names, drop_auto_broadcasters(ρχs))
                    unrolled_reduce(names_and_ρχs, sgsʲ) do sgs_offsets, (χ_name, ρχ)
                        χ_old = MatrixFields.get_field(sgsʲ, χ_name)
                        χ⁰ = specific_env_value(
                            χ_name, ρχ, ρ, env_sgsʲs, ρa⁰_val, turbconv_model,
                        )
                        num =
                            χ_name === @name(mse) ? χ_old / dtγ + entr_tot * χ⁰ :
                            max(zero_FT, χ_old) / dtγ + entr_tot * max(zero_FT, χ⁰)
                        b =
                            χ_name === @name(mse) ?
                            (num + r_sfc * mse_sfc) / (den + r_sfc) :
                            χ_name === @name(q_tot) ?
                            (num + r_sfc * q_tot_sfc) / (den + r_sfc) : num / den
                        merge(
                            sgs_offsets,
                            NamedTuple{(MatrixFields.extract_first(χ_name),)}((b,)),
                        )
                    end
                end).(
                    Yc.sgsʲs,
                    ᶜenv_sgsʲs,
                    Yc.ρ,
                    tuple.(ᶜρχs...),
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
