import ClimaTimeSteppers as CTS
import ClimaUtilities.TimeManager: ITime

#####
##### ODE integrator construction. Takes typed inputs only, the YAML-shaped
##### entry points live in `config/type_getters.jl`.
#####

function get_jacobian(
    ode_algo, Y, atmos, jacobian::JacobianAlgorithm, debug_jacobian;
    verbose = false,
)
    ode_algo isa AcousticMultirate && (ode_algo = ode_algo.inner_alg)
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

function args_integrator(Y, p, tspan, ode_algo, callback,
    jacobian, debug_jacobian, prescribed_flow, dt_integrator,
    update_cache_every, update_constrain_state_every;
    verbose = false,
)
    (; atmos) = p
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
        tendency_function = CTS.ClimaODEFunction(;
            T_exp_T_lim!, T_imp!,
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
