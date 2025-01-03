import SciMLBase

#####
##### Callback helpers
#####

callback_from_affect(x::AtmosCallback) = x
function callback_from_affect(affect!)
    for p in propertynames(affect!)
        x = getproperty(affect!, p)
        if x isa AtmosCallback
            return x
        end
    end
    return nothing
end
function atmos_callbacks(cbs::SciMLBase.CallbackSet)
    all_cbs = [cbs.continuous_callbacks..., cbs.discrete_callbacks...]
    callback_objs = map(cb -> callback_from_affect(cb.affect!), all_cbs)
    filter!(x -> (x isa AtmosCallback), callback_objs)
    return callback_objs
end

n_measured_calls(integrator) = n_measured_calls(integrator.callback)
n_measured_calls(cbs::SciMLBase.CallbackSet) =
    map(x -> x.n_measured_calls, atmos_callbacks(cbs))

n_expected_calls(integrator) = n_expected_calls(
    integrator.callback,
    integrator.dt,
    integrator.sol.prob.tspan,
)
n_expected_calls(cbs::SciMLBase.CallbackSet, dt, tspan) =
    map(x -> n_expected_calls(x, dt, tspan), atmos_callbacks(cbs))

n_steps_per_cycle(integrator) =
    n_steps_per_cycle(integrator.callback, integrator.dt)
function n_steps_per_cycle(cbs::SciMLBase.CallbackSet, dt)
    nspc = n_steps_per_cycle_per_cb(cbs, dt)
    return isempty(nspc) ? 1 : lcm(nspc)
end

n_steps_per_cycle_per_cb(integrator) =
    n_steps_per_cycle_per_cb(integrator.callback, integrator.dt)

function n_steps_per_cycle_per_cb(cbs::SciMLBase.CallbackSet, dt)
    return map(atmos_callbacks(cbs)) do cb
        cbf = callback_frequency(cb)
        if cbf isa EveryΔt
            Int(ceil(cbf.Δt / dt))
        elseif cbf isa EveryNSteps
            cbf.n
        else
            error("Uncaught case")
        end
    end
end

n_steps_per_cycle_per_cb_diagnostic(cbs) =
    [callback_frequency(cb).n for cb in cbs if callback_frequency(cb).n > 0]


# TODO: Move to ClimaUtilities once we move the schedules there
import ClimaDiagnostics.Schedules: AbstractSchedule

"""
    CappedGeometricSeriesSchedule(max_steps)

True every 2^N iterations or every `max_steps`.

This is useful to have an exponential ramp up of something that saturates to a constant
frequency. (For instance, reporting something more frequently at the beginning of the
simulation, and less frequency later)
"""
struct CappedGeometricSeriesSchedule <: AbstractSchedule
    """GeometricSeriesSchedule(integrator) is true every 2^N iterations or every max_steps"""
    max_steps::Int
    """Last step that this returned true"""
    step_last::Base.RefValue{Int}

    function CappedGeometricSeriesSchedule(max_steps; step_last = Ref(0))
        return new(max_steps, step_last)
    end
end

"""
    CappedGeometricSeriesSchedule(integrator)

Returns true if `integrator.step >= last_step + max_steps`, or when `integrator.step` is a
power of 2. `last_step` is the last step this function was true and `max_step` is maximum
allowed interval as defined in the schedule.
"""
function (schedule::CappedGeometricSeriesSchedule)(integrator)::Bool
    if isinteger(log2(integrator.step)) ||
       integrator.step > schedule.step_last[] + schedule.max_steps
        schedule.step_last[] = integrator.step
        return true
    else
        return false
    end
end
