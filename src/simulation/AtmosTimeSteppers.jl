import ClimaTimeSteppers as CTS

# The main reason why having a AtmosTimeStepper wrapper is convenient is that we work with a
# potentially large collection of algorithms from different packages (ODE and CTS) with
# differently properties an slightly different interfaces. Having a single object allows us
# to provide constructors that massage the algorithms in what we need but also handle
# details of the algorithm more easily (e.g., providing functions to identify whether an
# algorithm comes from ClimaTimeStepper).
#
# The only requirement for algorithm is that it has to be compatible with the OrdinaryDiffEq
# interface.
#

struct AtmosTimeStepper
    algorithm::Any
end

function Base.getproperty(timestepper::AtmosTimeStepper, v::Symbol)
    if v == :is_cts
        return typeof(timestepper.algorithm) <: CTS.DistributedODEAlgorithm
    else
        return getfield(timestepper, v)
    end
end

"""
   AtmosTimeStepper(algorithm::T;
                    max_newton_iters_ode = 1) where {T <: CTS.IMEXARKAlgorithmName}

Keyword arguments
=================

- `max_newton_iters_ode`: Maximum number of iterations allowed in the Newton solver.


"""
function AtmosTimeStepper(
    algorithm::T;
    max_newton_iters_ode = 1,
) where {T <: CTS.IMEXARKAlgorithmName}
    # TODO: Support krylov_method and convergence_checker
    newtons_method = CTS.NewtonsMethod(;
        max_iters = max_newton_iters_ode,
        krylov_method = nothing,
        convergence_checker = nothing,
    )
    return AtmosTimeStepper(CTS.IMEXAlgorithm(algorithm, newtons_method))
end
