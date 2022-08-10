using UnPack

using OrdinaryDiffEq:
    AbstractNLSolver,
    isnewton,
    DIRK,
    update_W!,
    initialize!,
    Divergence,
    get_new_W!,
    initial_η,
    compute_step!,
    apply_step!,
    Convergence,
    isJcurrent,
    postamble!,
    TryAgain
using OrdinaryDiffEq.SciMLBase: RECOMPILE_BY_DEFAULT

import OrdinaryDiffEq: nlsolve!
import OrdinaryDiffEq.SciMLBase: SplitFunction
import OrdinaryDiffEq as ODE
using DiffEqBase: @..

# TODO: remove this hack after updating to ClimaTimeSteppers.jl
function ODE.calc_tderivative!(
    integrator::ODE.ODEIntegrator,
    cache,
    dtd1,
    repeat_step,
)
    @inbounds begin
        @unpack fsalfirst, linsolve_tmp = cache
        @.. linsolve_tmp = fsalfirst
    end
    return nothing
end

#=
Issue: calc_W! attempts to access f.Wfact when has_Wfact(f) and f.Wfact_t when
       has_Wfact_t(f). However, when f is a SplitFunction, Wfact and Wfact_t are
       stored in f.f1, and the corresponding fields in f are set to nothing.
       In addition, build_J_W uses f.jac_prototype to construct J and W when
       f.jac_prototype !== nothing. However, when f is a SplitFunction,
       jac_prototype is stored in f.f1, and f.jac_prototype is set to nothing.
       Similarly, calc_tderivative! uses f.tgrad when has_tgrad(f).
=#
function SplitFunction{iip}(f1, f2; kwargs...) where {iip}
    f1 = ODEFunction(f1)
    f2 = ODEFunction{iip}(f2)
    SplitFunction{iip, RECOMPILE_BY_DEFAULT}(
        f1,
        f2;
        kwargs...,
        Wfact = f1.Wfact,
        Wfact_t = f1.Wfact_t,
        jac_prototype = f1.jac_prototype,
        tgrad = f1.tgrad,
    )
end

#=
Issue: nlsolve! has a goto statement that restarts the solver when the
       conditions for DiffEqBase.SlowConvergence (denoted by TryAgain) are
       satisfied. These conditions include !isJcurrent, which is always true
       when a custom Wfact or Wfact_t is used because of how calc_W! is written.
       So, when a custom Wfact or Wfact_t is used, nlsolve! sometimes enters an
       infinite loop (though there are also situations where it converges after
       several iterations of the loop). This is probably not the intended
       behavior, so this patch limits the number of loop iterations to 100,
       after which divergence is declared.
=#
function nlsolve!(
    nlsolver::AbstractNLSolver,
    integrator,
    cache = nothing,
    repeat_step = false,
)
    num_redos = 0
    @label REDO
    if isnewton(nlsolver)
        cache === nothing && throw(
            ArgumentError(
                "cache is not passed to `nlsolve!` when using NLNewton",
            ),
        )
        if nlsolver.method === DIRK
            γW = nlsolver.γ * integrator.dt
        else
            γW = nlsolver.γ * integrator.dt / nlsolver.α
        end
        update_W!(nlsolver, integrator, cache, γW, repeat_step)
    end

    @unpack maxiters, κ, fast_convergence_cutoff = nlsolver

    initialize!(nlsolver, integrator)
    nlsolver.status = Divergence
    η = get_new_W!(nlsolver) ? initial_η(nlsolver, integrator) : nlsolver.ηold

    local ndz
    for iter in 1:maxiters
        nlsolver.iter = iter

        # compute next step and calculate norm of residuals
        iter > 1 && (ndzprev = ndz)
        ndz = compute_step!(nlsolver, integrator)
        if !isfinite(ndz)
            nlsolver.status = Divergence
            break
        end

        # check divergence (not in initial step)
        if iter > 1
            θ = ndz / ndzprev

            # divergence
            if θ > 2
                nlsolver.status = Divergence
                break
            end
        end

        apply_step!(nlsolver, integrator)

        # check for convergence
        iter > 1 && (η = θ / (1 - θ))
        if (iter == 1 && ndz < 1e-5) ||
           (iter > 1 && (η >= zero(η) && η * ndz < κ))
            nlsolver.status = Convergence
            break
        end
    end

    if isnewton(nlsolver) &&
       nlsolver.status == Divergence &&
       !isJcurrent(nlsolver, integrator)
        num_redos += 1
        if num_redos < 100
            nlsolver.status = TryAgain
            @goto REDO
        end
    end

    nlsolver.ηold = η
    postamble!(nlsolver, integrator)
end
