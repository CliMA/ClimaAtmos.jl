import ClimaTimeSteppers as CTS
import ClimaUtilities.TimeManager: ITime
import LinearAlgebra
import StaticArrays: SVector

#####
##### ODE integrator construction. Takes typed inputs only, the YAML-shaped
##### entry points live in `config/type_getters.jl`.
#####

"""
    imex_gamma(ode_algo)

Return the single-`γ` DIRK diagonal of `ode_algo`'s implicit Butcher tableau, or
`nothing` when `ode_algo` is not a single-`γ` `IMEXAlgorithm` with at least one
implicit stage (`build_cache` rejects an `IMEXAlgorithm` that has a non-zero
implicit diagonal with non-uniform `γ`). Explicit zero-diagonal stages are
skipped. Used by `build_cache` to resolve `dtγ`.
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
    get_jacobian(ode_algo, Y, atmos, jacobian::JacobianAlgorithm, debug_jacobian; verbose = false)

Build the `Jacobian` object for an implicit algorithm, and return `nothing` for
explicit algorithms.

`jacobian` selects the algorithm (`ManualSparseJacobian`, `AutoSparseJacobian`, or
`AutoDenseJacobian`), `debug_jacobian` turns on the Jacobian's own verbose checks, and
`verbose` logs the chosen algorithm and its derivative flags. Called from
`args_integrator`.
"""
function get_jacobian(
    ode_algo, Y, atmos, jacobian::JacobianAlgorithm, debug_jacobian;
    verbose = false,
)
    ode_algo isa CTS.IMEXAlgorithm || return nothing
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

Explicit Runge-Kutta names give an `ExplicitAlgorithm`, and IMEX-ARK names give
an `IMEXAlgorithm` configured with `NewtonsMethod`. Rosenbrock algorithms (such
as `SSPKnoth`) are rejected with an `ArgumentError` because post-ARK vertical
scalar transport requires an IMEX/ARK tableau with `T_imp` and `b_imp`.

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
        throw(
            ArgumentError(
                "Rosenbrock algorithm `$ode_algo_name` is not supported: post-ARK vertical scalar transport requires an IMEX/ARK tableau with `T_imp` and `b_imp` (e.g. `ARS343` or `SSP333`)",
            ),
        )
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
`update_cache` field. `"dss"` is not valid here because CTS only fires `cache!`
at state-ready (`EndOfStage`/`EndOfStep`) sites. End-of-step updates inside
`step_u!` are suppressed (`StageOnlyUpdateHandler` for `"stage"`,
`NeverUpdateHandler` for `"step"`) because `post_step_callback!` refreshes the
cache after post-ARK physics.
"""
function update_cache_signal_handler(freq_str)
    if freq_str == "stage"
        return StageOnlyUpdateHandler()
    elseif freq_str == "step"
        return NeverUpdateHandler()
    else
        error("Unknown `update_cache_every = $(freq_str)`; expected `stage` or `step`")
    end
end

"""
    update_constrain_state_signal_handler(freq_str)

Map a YAML frequency string (`"stage"`, `"step"`, `"dss"`) to the matching
ClimaTimeSteppers `UpdateSignalHandler` for `ClimaODEFunction`'s
`update_constrain_state` field. `"dss"` fires `constrain_state!` at every
`dss!` site. End-of-step firing inside `step_u!` is suppressed
(`StageOnlyUpdateHandler` for `"stage"`, `NeverUpdateHandler` for `"step"`,
`WithDSSOnlyUpdateHandler` for `"dss"`) because `post_step_callback!` applies
`constrain_state!` after post-ARK physics.
"""
function update_constrain_state_signal_handler(freq_str)
    if freq_str == "stage"
        return StageOnlyUpdateHandler()
    elseif freq_str == "step"
        return NeverUpdateHandler()
    elseif freq_str == "dss"
        return WithDSSOnlyUpdateHandler()
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
(`initialize_implicit_stage_problem!`), `T_post_imp! = nothing`, the cache and
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
    ode_algo isa CTS.RosenbrockAlgorithm && throw(
        ArgumentError(
            "Rosenbrock algorithms are not supported: post-ARK vertical scalar transport requires an IMEX/ARK tableau with `T_imp` and `b_imp` (e.g. `ARS343` or `SSP333`)",
        ),
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
        T_post_imp! = nothing
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
