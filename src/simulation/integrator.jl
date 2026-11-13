import ClimaTimeSteppers as CTS
import ClimaUtilities.TimeManager: ITime
import LinearAlgebra
import StaticArrays: SVector

#####
##### ODE integrator construction. Takes typed inputs only, the YAML-shaped
##### entry points live in `config/type_getters.jl`.
#####

"""
    MicrophysicsAveragingWindow

Tag type that selects the time window `set_microphysics_tendency_cache!` for 1M
hands to CM's `LinearizedAverage` substep solver. Two concrete subtypes:

  - `StepAveraging` averages over the full dynamics step `p.dt`.
  - `StageAveraging` averages over the implicit-stage window `p.dtγ` (so
    stage-fired `cache!` updates see a window matched to the stage).

The choice is baked into `AtmosCache` at construction (see `build_cache`) from
`update_cache_every` and the ODE scheme's `γ`; `mp_averaging_dt` reads it at
cache-update time.
"""
abstract type MicrophysicsAveragingWindow end

"""
    StepAveraging()

Full dynamics-step averaging; the default.
"""
struct StepAveraging <: MicrophysicsAveragingWindow end

"""
    StageAveraging()

Implicit-stage averaging; constructed when `update_cache_every == "stage"` and
the ODE scheme has a well-defined single-γ DIRK diagonal.
"""
struct StageAveraging <: MicrophysicsAveragingWindow end

"""
    mp_averaging_dt(p, window)

Return the dt the 1M bulk-tendency averager should use at `cache!` time. The
`window` tag selects `p.dt` (step) or `p.dtγ` (stage).
"""
@inline mp_averaging_dt(p, ::StepAveraging) = p.dt
@inline mp_averaging_dt(p, ::StageAveraging) = p.dtγ

"""
    imex_gamma(ode_algo)

Return the single-`γ` DIRK diagonal of `ode_algo`'s implicit Butcher tableau, or
`nothing` when `ode_algo` is not a single-`γ` `IMEXAlgorithm` with at least one
implicit stage (`build_cache` rejects an `IMEXAlgorithm` that has a non-zero
implicit diagonal with non-uniform `γ`). Explicit zero-diagonal stages are
skipped. Used by `build_cache` to resolve `dtγ` and the microphysics averaging
window.
"""
imex_gamma(_) = nothing
function imex_gamma(alg::CTS.IMEXAlgorithm)
    a = alg.tableau.a_imp
    nonzero_diags = filter(!iszero, LinearAlgebra.diag(a))
    isempty(nonzero_diags) && return nothing
    γ = first(nonzero_diags)
    all(x -> x ≈ γ, nonzero_diags) ? γ : nothing
end

"""
    imex_b_imp_stages(ode_algo, ::Type{FT}) where {FT}

Return `(b_imp_stages, n_imp_stages)`: a fixed-size `SVector{8, FT}` containing
the implicit Butcher tableau stage weights `FT(b_imp[i])` for every implicit stage
`i` (`a_imp[i, i] ≠ 0`) of `ode_algo` padded with zeros, and the count `n_imp_stages::Int`
of implicit stages. Returns `(zero(SVector{8, FT}), 0)` when `ode_algo` is not
an `IMEXAlgorithm`. Used by `build_cache` to accumulate stage-weighted surface
precipitation fluxes across implicit stages with a solver-invariant scratch layout.

A stage with `a_imp[i, i] == 0` and `b_imp[i] ≠ 0` (the first stage of the
`SSP` tableaus) evaluates its implicit tendency explicitly, outside the
`T_post_imp!` hook that samples the sedimentation flux. Its weight is
redistributed over the sampled stages in proportion to their own weights, so the
diagnosed flux keeps the step's total weight, and a warning names the
approximation.
"""
imex_b_imp_stages(_, ::Type{FT}) where {FT} = (zero(SVector{8, FT}), 0)
function imex_b_imp_stages(alg::CTS.IMEXAlgorithm, ::Type{FT}) where {FT}
    nm = alg.newtons_method
    if isnothing(nm) ||
       !isnothing(nm.krylov_method) ||
       !isnothing(nm.line_search) ||
       isnothing(imex_gamma(alg)) ||
       !(
           nm.update_j == CTS.UpdateEvery(CTS.NewNewtonIteration) ||
           (
               nm.update_j == CTS.UpdateEvery(CTS.NewNewtonSolve) &&
               nm.max_iters == 1
           )
       )
        return (zero(SVector{8, FT}), 0)
    end
    a = alg.tableau.a_imp.coeffs
    b = alg.tableau.b_imp.coeffs
    s = size(a, 1)
    sampled = [i for i in 1:s if !iszero(a[i, i])]
    unsampled_weight = sum(FT(b[i]) for i in 1:s if iszero(a[i, i]); init = zero(FT))
    sampled_weight = sum(FT(b[i]) for i in sampled; init = zero(FT))
    if !iszero(unsampled_weight) && !iszero(sampled_weight)
        @warn "$(nameof(typeof(alg.name))) weights an explicit-diagonal stage \
               with b_imp = $unsampled_weight; the surface precipitation \
               diagnostic redistributes this weight over the implicit stages."
    end
    scale =
        iszero(sampled_weight) ? one(FT) :
        (sampled_weight + unsampled_weight) / sampled_weight
    b_imp = [scale * FT(b[i]) for i in sampled]
    return (pad_b_imp_stages(b_imp, FT), length(b_imp))
end

"""
    get_jacobian(ode_algo, Y, atmos, jacobian::JacobianAlgorithm, debug_jacobian; verbose = false)

Build the `Jacobian` object for an implicit or Rosenbrock algorithm, and return `nothing`
for explicit algorithms, which never need one.

`jacobian` selects the algorithm (`ManualSparseJacobian`, `AutoSparseJacobian`, or
`AutoDenseJacobian`), `debug_jacobian` turns on the Jacobian's own verbose checks, and
`verbose` logs the chosen algorithm and its derivative flags. Called from
`args_integrator`.
"""
function get_jacobian(
    ode_algo, Y, atmos, jacobian::JacobianAlgorithm, debug_jacobian;
    verbose = false,
)
    ode_algo isa Union{CTS.IMEXAlgorithm, CTS.RosenbrockAlgorithm} ||
        return nothing
    verbose && @info "Jacobian algorithm: $(summary_string(jacobian))"
    jac = Jacobian(jacobian, Y, atmos; verbose = debug_jacobian)
    if verbose && hasproperty(jac.cache, :derivative_flags)
        flags_str = join(
            ("$k = $(typeof(v).name.name)" for (k, v) in pairs(jac.cache.derivative_flags)),
            ", ",
        )
        @info "Jacobian derivative flags: $flags_str"
    end
    return jac
end

"""
    ode_configuration(::Type{FT}, ode_name, update_jacobian_every, max_newton_iters_ode,
                      use_krylov_method, use_dynamic_krylov_rtol,
                      eisenstat_walker_forcing_alpha, krylov_rtol, use_newton_rtol,
                      newton_rtol, jvp_step_adjustment) where {FT}

Build the ClimaTimeSteppers algorithm named by `ode_name`.

Rosenbrock names give a `RosenbrockAlgorithm` (which only supports
`update_jacobian_every = "solve"`, warning otherwise), explicit Runge-Kutta names give an
`ExplicitAlgorithm`, and IMEX-ARK names give an `IMEXAlgorithm` whose Newton solver is
configured here:

  - `update_jacobian_every`: `"dt"`, `"stage"`, or `"solve"`, mapped to updating the
    Jacobian at each new timestep, Newton solve, or Newton iteration respectively; any
    other value raises an error.
  - `max_newton_iters_ode`: Maximum number of Newton iterations per solve.
  - `use_krylov_method`: Solve the linear system with a Jacobian-free Krylov method whose
    finite-difference step is scaled by `jvp_step_adjustment`. The relative tolerance is
    either the Eisenstat-Walker forcing term with exponent
    `eisenstat_walker_forcing_alpha` (when `use_dynamic_krylov_rtol`) or the constant
    `krylov_rtol`.
  - `use_newton_rtol`: Stop Newton iterations once the maximum relative error falls below
    `newton_rtol`, instead of always taking `max_newton_iters_ode` iterations.
"""
function ode_configuration(::Type{FT}, ode_name, update_jacobian_every,
    max_newton_iters_ode, use_krylov_method, use_dynamic_krylov_rtol,
    eisenstat_walker_forcing_alpha, krylov_rtol, use_newton_rtol, newton_rtol,
    jvp_step_adjustment,
) where {FT}
    ode_algo_name = getproperty(CTS, Symbol(ode_name))
    @info "Using ODE config: `$ode_algo_name`"
    return if ode_algo_name <: CTS.RosenbrockAlgorithmName
        if update_jacobian_every != "solve"
            @warn "Rosenbrock algorithms in ClimaTimeSteppers currently only \
                   support `update_jacobian_every` = \"solve\""
        end
        CTS.RosenbrockAlgorithm(CTS.tableau(ode_algo_name()))
    elseif ode_algo_name <: CTS.ERKAlgorithmName
        CTS.ExplicitAlgorithm(ode_algo_name())
    else
        @assert ode_algo_name <: CTS.IMEXARKAlgorithmName
        newtons_method = CTS.NewtonsMethod(;
            max_iters = max_newton_iters_ode,
            update_j = if update_jacobian_every == "dt"
                CTS.UpdateEvery(CTS.NewTimeStep)
            elseif update_jacobian_every == "stage"
                CTS.UpdateEvery(CTS.NewNewtonSolve)
            elseif update_jacobian_every == "solve"
                CTS.UpdateEvery(CTS.NewNewtonIteration)
            else
                error("Unknown value of `update_jacobian_every`: \
                        $(update_jacobian_every)")
            end,
            krylov_method = if use_krylov_method
                CTS.KrylovMethod(;
                    jacobian_free_jvp = CTS.ForwardDiffJVP(;
                        step_adjustment = FT(
                            jvp_step_adjustment,
                        ),
                    ),
                    forcing_term = if use_dynamic_krylov_rtol
                        α = FT(eisenstat_walker_forcing_alpha)
                        CTS.EisenstatWalkerForcing(; α)
                    else
                        CTS.ConstantForcing(FT(krylov_rtol))
                    end,
                )
            else
                nothing
            end,
            convergence_checker = if use_newton_rtol
                norm_condition = CTS.MaximumRelativeError(
                    FT(newton_rtol),
                )
                CTS.ConvergenceChecker(; norm_condition)
            else
                nothing
            end,
        )
        CTS.IMEXAlgorithm(ode_algo_name(), newtons_method)
    end
end

"""
    update_cache_signal_handler(freq_str)

Map a YAML frequency string (`"stage"`, `"step"`) to the matching
ClimaTimeSteppers `UpdateSignalHandler` for `ClimaODEFunction`'s
`update_cache` field. `"dss"` is not a valid choice here because CTS never
fires `cache!` with `WithDSSSignal` — only at state-ready
(`EndOfStage`/`EndOfStep`) sites.
"""
function update_cache_signal_handler(freq_str)
    if freq_str == "stage"
        return CTS.UpdateEvery(CTS.EndOfStage)
    elseif freq_str == "step"
        return CTS.UpdateEvery(CTS.EndOfStep)
    else
        error("Unknown `update_cache_every = $(freq_str)`; expected `stage` or `step`")
    end
end

"""
    update_constrain_state_signal_handler(freq_str)

Map a YAML frequency string (`"stage"`, `"step"`, `"dss"`) to the matching
ClimaTimeSteppers `UpdateSignalHandler` for `ClimaODEFunction`'s
`update_constrain_state` field. `"dss"` fires `constrain_state!` at every
`dss!` site (including pre-implicit and post-`initialize_imp!` DSSes).
"""
function update_constrain_state_signal_handler(freq_str)
    if freq_str == "stage"
        return CTS.UpdateEvery(CTS.EndOfStage)
    elseif freq_str == "step"
        return CTS.UpdateEvery(CTS.EndOfStep)
    elseif freq_str == "dss"
        return CTS.UpdateEvery(CTS.WithDSS)
    else
        error(
            "Unknown `update_constrain_state_every = $(freq_str)`; expected `stage`, `step`, or `dss`",
        )
    end
end

"""
    args_integrator(Y, p, tspan, ode_algo, callback, jacobian, debug_jacobian,
                    prescribed_flow, dt_integrator, update_cache_every,
                    update_constrain_state_every; verbose = false)

Assemble the positional arguments and keyword arguments for `ClimaTimeSteppers.init`.

Passes the tendency and stage hooks to a `ClimaODEFunction`: the explicit and
limited tendencies, the implicit tendency with its Jacobian (both replaced by
`fully_explicit_tendency!` when `prescribed_flow` is set, so sound waves are not
treated implicitly), the implicit-stage initializer
(`initialize_implicit_stage_problem!`), the post-Newton stage hook
(`post_newton_stage_update!`, skipped when `T_imp!` is absent or
`p.dtγ == 0`), the cache and
implicit-cache updates, the limiter, DSS, and the state constraints. The cache
and constraint update frequencies come from `update_cache_signal_handler` and
`update_constrain_state_signal_handler`.

# Returns

`(args, kwargs)`: `args` is `(problem, ode_algo)`, and `kwargs` holds `saveat` (the two
endpoints of `tspan`), the `callback` set, and the promoted timestep `dt`.
"""
function args_integrator(Y, p, tspan, ode_algo, callback,
    jacobian, debug_jacobian, prescribed_flow, dt_integrator,
    update_cache_every, update_constrain_state_every;
    verbose = false,
)
    (; atmos) = p
    if !(atmos.microphysics_model isa DryModel)
        (; energy_humidity_upwinding, tracer_upwinding) = atmos.numerics
        if !isempty(sedimenting_tracer_names(Y)) &&
           energy_humidity_upwinding != tracer_upwinding
            @info(
                "Sedimenting microphysics species follow \
                 `energy_humidity_upwinding` ($(energy_humidity_upwinding)); \
                 `tracer_upwinding` ($(tracer_upwinding)) applies to the \
                 passive tracers."
            )
        end
        (; edmfx_mse_humidity_upwinding, edmfx_tracer_upwinding) = atmos.numerics
        if atmos.turbconv_model isa PrognosticEDMFX &&
           !isempty(sedimenting_sgs_tracer_names(Y)) &&
           edmfx_mse_humidity_upwinding != edmfx_tracer_upwinding
            @info(
                "Updraft microphysics species follow \
                 `edmfx_mse_humidity_upwinding` ($(edmfx_mse_humidity_upwinding)); \
                 `edmfx_tracer_upwinding` ($(edmfx_tracer_upwinding)) applies \
                 to the passive updraft tracers."
            )
        end
    end
    # With "dss", `constrain_state!` runs both before `temp .= U` and after
    # `initialize_imp!`; when `warm_start_grid_mean_tracers!` moves the
    # grid-mean condensates of the guess (`PrognosticEDMFX`, 1M, Picard rounds
    # > 0), the second pass moves `temp + Δ` when negative, altering `R(U^g)`
    # relative to `temp` so a single-iteration Newton solve drains the excess.
    if update_constrain_state_every == "dss" &&
       isnothing(prescribed_flow) &&
       !iszero(p.dtγ) &&
       atmos.turbconv_model isa PrognosticEDMFX &&
       atmos.microphysics_model isa NonEquilibriumMicrophysics1M &&
       atmos.edmfx_model.updraft_picard_rounds > 0
        @warn(
            "`update_constrain_state_every = dss` re-clamps the warm-started \
             grid-mean condensates of the Newton guess after the \
             stage-entering state is copied, which can produce an upward \
             surface precipitation flux; use `stage` or `step`."
        )
    end
    if isnothing(prescribed_flow) && !iszero(p.dtγ)
        if atmos.numerics.energy_humidity_upwinding == Val(:none) &&
           atmos.diff_mode != Implicit()
            @warn(
                "`energy_humidity_upwinding = none` skips the post-Newton \
                 dissipation, leaving vertical advection of `e_tot - Φ`, \
                 `q_tot`, and any condensates forward-Euler central."
            )
        end
        if atmos.numerics.energy_humidity_upwinding != Val(:none) &&
           ode_algo isa CTS.IMEXAlgorithm &&
           !isnothing(ode_algo.newtons_method)
            nm = ode_algo.newtons_method
            if (
                jacobian isa ManualSparseJacobian &&
                isnothing(nm.krylov_method) &&
                nm.max_iters > 1
            )
                @warn(
                    "With `ManualSparseJacobian` and `max_newton_iters_ode > 1`, \
                     additional Newton iterations iterate the explicit central \
                     scalar step while the `+2 C_γ` dissipation weight is \
                     disabled (`compensate_explicit_scalar_advection = false`)."
                )
            end
        end
    end
    @timed_log verbose "Built tendency function" begin
        if isnothing(prescribed_flow)

            # This is the default case
            T_exp_T_lim! = remaining_tendency!
            T_imp! = CTS.ODEFunction(
                implicit_tendency!;
                jac_prototype = get_jacobian(
                    ode_algo, Y, atmos, jacobian, debug_jacobian; verbose,
                ),
                Wfact = update_jacobian!,
            )
            cache_imp! = set_implicit_precomputed_quantities!
        else
            # `prescribed_flow` is an experimental case where the flow is prescribed,
            # so implicit tendencies are treated explicitly to avoid treatment of sound waves
            T_exp_T_lim! = fully_explicit_tendency!
            T_imp! = nothing
            cache_imp! = nothing
        end
        # `T_post_imp!` applies the implicit upwind dissipation and the
        # high-order correction of the vertical transport at the solved
        # velocity, provided `T_imp!` is active and the tableau has a single-γ
        # implicit stage (`p.dtγ > 0`); otherwise `nothing`, so CTS skips the
        # post-Newton path (and its `cache_imp!` refresh) at every stage. Fully
        # explicit tableaus such as `SSP33ShuOsher` have no implicit stage to
        # correct.
        T_post_imp! =
            (isnothing(T_imp!) || iszero(p.dtγ)) ? nothing :
            post_newton_stage_update!
        tendency_function = CTS.ClimaODEFunction(;
            T_exp_T_lim!, T_imp!, T_post_imp!,
            cache! = set_precomputed_quantities!, cache_imp!,
            lim! = limiters_func!,
            dss!, constrain_state!,
            update_cache = update_cache_signal_handler(update_cache_every),
            update_constrain_state = update_constrain_state_signal_handler(
                update_constrain_state_every,
            ),
            initialize_imp! = initialize_implicit_stage_problem!,
        )
    end
    problem = CTS.ODEProblem(tendency_function, Y, tspan, p)
    # Promote to ensure t_begin, t_end, and dt_integrator all have the same type
    # (dt_integrator is ITime, p.dt is FT)
    t_begin, t_end, dt = promote(tspan[1], tspan[2], dt_integrator)
    # Save solution to integrator.sol at the beginning and end
    saveat = [t_begin, t_end]
    args = (problem, ode_algo)
    kwargs = (; saveat, callback, dt)
    return (args, kwargs)
end
