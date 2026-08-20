import ClimaTimeSteppers as CTS

#####
##### AtmosCallback — wraps a step function with a frequency and a call counter.
#####

"""
    AbstractCallbackFrequency

How often an `AtmosCallback` is meant to fire.

Subtypes:

  - `EveryNSteps`: every `n` timesteps.
  - `EveryΔt`: every `Δt` of simulation time.

The frequency is carried alongside the callback purely for bookkeeping: it is what
`n_expected_calls` and `n_steps_per_cycle` reason about. The actual firing is decided by
the `DiscreteCallback` condition built in `call_every_n_steps` or `call_every_dt`.
"""
abstract type AbstractCallbackFrequency end

"""
    EveryNSteps(n)

Callback frequency of once every `n` timesteps.

# Fields

  - `n`: Number of timesteps between calls [-].
"""
struct EveryNSteps <: AbstractCallbackFrequency
    n::Int
end

"""
    EveryΔt(Δt)

Callback frequency of once every `Δt` of simulation time.

# Fields

  - `Δt`: Interval between calls [s]. May be an `ITime`.
"""
struct EveryΔt{FT} <: AbstractCallbackFrequency
    Δt::FT
end

"""
    AtmosCallback(f!, cbf)
    AtmosCallback(f!, cbf, measured_calls)

Wrap a callback function with its intended frequency and a call counter.

Calling an `AtmosCallback` on an integrator invokes `f!` and increments the counter, so
that `n_measured_calls` can be compared against `n_expected_calls` at the end of a run.
The wrapper is what makes a `DiscreteCallback` recognizable to `atmos_callbacks`; plain
functions passed to the timestepper are invisible to that accounting.

# Fields

  - `f!`: Step function called as `f!(integrator)`; expected to return `nothing`.
  - `cbf`: An `AbstractCallbackFrequency` describing the intended cadence.
  - `measured_calls`: `Ref` counting how many times the callback has actually fired [-].
    Initialized to zero by the two-argument constructor.
"""
struct AtmosCallback{F, CBF <: AbstractCallbackFrequency}
    f!::F
    cbf::CBF
    measured_calls::Base.RefValue{Int}
end

AtmosCallback(f!, cbf) = AtmosCallback(f!, cbf, Ref(0))

"""
    (cb::AtmosCallback)(integrator)

Invoke the wrapped step function and count the call.

Mutates `integrator` through `cb.f!` and increments `cb.measured_calls`. Returns `nothing`.
"""
function (cb::AtmosCallback)(integrator)
    cb.f!(integrator)
    cb.measured_calls[] += 1
    return nothing
end

"""
    callback_frequency(cb::AtmosCallback)

Return the `AbstractCallbackFrequency` a callback was constructed with.
"""
callback_frequency(cb::AtmosCallback) = cb.cbf

"""
    prescribed_every_n_steps(x::EveryNSteps)
    prescribed_every_n_steps(cb::AtmosCallback)

Return the prescribed step interval `n` of a step-based callback frequency.
"""
prescribed_every_n_steps(x::EveryNSteps) = x.n
prescribed_every_n_steps(cb::AtmosCallback) = prescribed_every_n_steps(cb.cbf)

"""
    prescribed_every_Δt_steps(x::EveryΔt)
    prescribed_every_Δt_steps(cb::AtmosCallback)

Return the prescribed time interval `Δt` of a time-based callback frequency [s].
"""
prescribed_every_Δt_steps(x::EveryΔt) = x.Δt
prescribed_every_Δt_steps(cb::AtmosCallback) = prescribed_every_Δt_steps(cb.cbf)

"""
    n_measured_calls(cb::AtmosCallback)
    n_measured_calls(integrator)

Return how many times a callback has actually fired.

The integrator form returns one count per `AtmosCallback` attached to it, in the order
given by `atmos_callbacks`. Compare against `n_expected_calls` to detect callbacks that
are firing more or less often than intended.
"""
n_measured_calls(cb::AtmosCallback) = cb.measured_calls[]

# TODO: improve accuracy
"""
    n_expected_calls(cbf::AbstractCallbackFrequency, dt, tspan)
    n_expected_calls(cb::AtmosCallback, dt, tspan)
    n_expected_calls(cbs, dt, tspan)
    n_expected_calls(integrator)

Return how many times a callback should fire over `tspan`, from its prescribed frequency.

This is the ideal count implied by the frequency, not a prediction of the actual one: it
ignores the `skip_first` and `call_at_end` adjustments and does not round to whole steps,
so small discrepancies against `n_measured_calls` are expected. The collection and
integrator forms return one count per attached `AtmosCallback`.
"""
n_expected_calls(cbf::EveryΔt, _, tspan) = (tspan[2] - tspan[1]) / cbf.Δt
n_expected_calls(cbf::EveryNSteps, dt, tspan) =
    ((tspan[2] - tspan[1]) / dt) / cbf.n
n_expected_calls(cb::AtmosCallback, dt, tspan) =
    n_expected_calls(cb.cbf, dt, tspan)

#####
##### Callback helpers
#####

"""
    call_every_n_steps(f!, n = 1; skip_first = false, call_at_end = false,
                       condition = nothing)

Build a `DiscreteCallback` that calls `f!(integrator)` every `n` timesteps.

The step function is wrapped in an `AtmosCallback` so that its calls are counted.
Unless `skip_first` is set, `f!` also runs once at initialization, before the first step;
that initial call is not governed by `n`.

# Arguments

  - `f!`: Step function called as `f!(integrator)`.
  - `n = 1`: Number of timesteps between calls [-]. Must be finite.

# Keyword Arguments

  - `skip_first = false`: Whether to suppress the call at initialization.
  - `call_at_end = false`: Whether to also fire on the final step of the simulation, even if
    it is not a multiple of `n`. Ignored when `condition` is given.
  - `condition = nothing`: Replacement for the built-in step-counting condition, called as
    `condition(u, t, integrator)`. Supplying one overrides `n` and `call_at_end` entirely;
    the frequency recorded on the `AtmosCallback` still reports `n`.

# Returns

A `ClimaTimeSteppers.DiscreteCallback`.
"""
function call_every_n_steps(
    f!,
    n = 1;
    skip_first = false,
    call_at_end = false,
    condition = nothing,
)
    @assert n ≠ Inf "Adding callback that never gets called!"
    cond = if isnothing(condition)
        previous_step = Ref(0)
        (u, t, integrator) ->
            (previous_step[] += 1) % n == 0 ||
            (call_at_end && t == integrator.sol.prob.tspan[2])
    else
        condition
    end
    cb! = AtmosCallback(f!, EveryNSteps(n))
    return CTS.DiscreteCallback(
        cond,
        cb!;
        initialize = (cb, u, t, integrator) -> skip_first || cb!(integrator),
    )
end

"""
    call_every_dt(f!, dt; skip_first = false, call_at_end = false)

Build a `DiscreteCallback` that calls `f!(integrator)` every `dt` of simulation time.

Because the callback can only fire on a step boundary, it fires on the first step at or
after each target time, and the next target is advanced from that target rather than from
the actual firing time, so the cadence does not drift. Unless `skip_first` is set, `f!`
also runs once at initialization.

# Arguments

  - `f!`: Step function called as `f!(integrator)`.
  - `dt`: Interval between calls [s]. Must be finite.

# Keyword Arguments

  - `skip_first = false`: Whether to suppress the call at initialization.
  - `call_at_end = false`: Whether to clamp the targets to the end of the simulation, so
    that the callback also fires on the final step.

# Returns

A `ClimaTimeSteppers.DiscreteCallback`.
"""
function call_every_dt(f!, dt; skip_first = false, call_at_end = false)
    cb! = AtmosCallback(f!, EveryΔt(dt))
    @assert Float64(dt) ≠ Inf "Adding callback that never gets called!"
    next_t = Ref{typeof(dt)}()
    affect! = function (integrator)
        cb!(integrator)

        t = integrator.t
        t_end = integrator.sol.prob.tspan[2]
        next_t[] = max(t, next_t[] + dt)
        if call_at_end
            next_t[] = min(next_t[], t_end)
        end
    end
    return CTS.DiscreteCallback(
        (u, t, integrator) -> t >= next_t[],
        affect!;
        initialize = (cb, u, t, integrator) -> begin
            skip_first || cb!(integrator)
            t_end = integrator.sol.prob.tspan[2]
            next_t[] =
                (call_at_end && t < t_end) ? min(t_end, t + dt) : t + dt
        end,
    )
end

"""
    callback_from_affect(affect!)

Recover the `AtmosCallback` behind a `DiscreteCallback`'s affect function.

The affect is sometimes the `AtmosCallback` itself and sometimes a closure over it, as in
`call_every_dt`, so this searches one level of fields. Returns `nothing` if the affect is
not backed by an `AtmosCallback`, which is how `atmos_callbacks` filters out callbacks
that opt out of the call accounting.
"""
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
"""
    atmos_callbacks(cbs)

Extract the `AtmosCallback`s from a `CallbackSet`.

Only discrete callbacks are considered, and only those whose affect is backed by an
`AtmosCallback`; everything else is dropped. This is the basis of the call accounting in
`n_measured_calls`, `n_expected_calls`, and `n_steps_per_cycle`.
"""
function atmos_callbacks(cbs)
    all_cbs = collect(cbs.discrete_callbacks)
    callback_objs = map(cb -> callback_from_affect(cb.affect!), all_cbs)
    filter!(x -> (x isa AtmosCallback), callback_objs)
    return callback_objs
end

n_measured_calls(integrator) =
    map(n_measured_calls, atmos_callbacks(integrator.callback))

n_expected_calls(integrator) = n_expected_calls(
    integrator.callback,
    integrator.dt,
    integrator.sol.prob.tspan,
)
n_expected_calls(cbs, dt, tspan) =
    map(x -> n_expected_calls(x, dt, tspan), atmos_callbacks(cbs))

"""
    n_steps_per_cycle(integrator)
    n_steps_per_cycle(cbs, dt)

Return the number of timesteps after which the whole callback pattern repeats.

This is the least common multiple of the individual periods from
`n_steps_per_cycle_per_cb`, or `1` when no `AtmosCallback` is attached. Benchmarks use it
to time a representative stretch of the simulation rather than a stretch that happens to
skip the expensive callbacks.
"""
n_steps_per_cycle(integrator) =
    n_steps_per_cycle(integrator.callback, integrator.dt)
function n_steps_per_cycle(cbs, dt)
    nspc = n_steps_per_cycle_per_cb(cbs, dt)
    return isempty(nspc) ? 1 : lcm(nspc)
end

"""
    n_steps_per_cycle_per_cb(integrator)
    n_steps_per_cycle_per_cb(cbs, dt)

Return each attached callback's period, expressed in timesteps.

Time-based frequencies are converted with `Δt / dt`, step-based ones are returned as-is.
Errors on a frequency type it does not recognize.
"""
n_steps_per_cycle_per_cb(integrator) =
    n_steps_per_cycle_per_cb(integrator.callback, integrator.dt)

function n_steps_per_cycle_per_cb(cbs, dt)
    return map(atmos_callbacks(cbs)) do cb
        cbf = callback_frequency(cb)
        if cbf isa EveryΔt
            cbf.Δt / dt
        elseif cbf isa EveryNSteps
            cbf.n
        else
            error("Uncaught case")
        end
    end
end

"""
    n_steps_per_cycle_per_cb_diagnostic(cbs)

Return the step periods of a collection of diagnostic callbacks.

Unlike `n_steps_per_cycle_per_cb`, this takes the callbacks directly rather than a
`CallbackSet`, and assumes every frequency is step-based. Periods of zero or fewer steps
are dropped.
"""
n_steps_per_cycle_per_cb_diagnostic(cbs) =
    [callback_frequency(cb).n for cb in cbs if callback_frequency(cb).n > 0]


# TODO: Move to ClimaUtilities once we move the schedules there
import ClimaDiagnostics.Schedules: AbstractSchedule

"""
    CappedGeometricSeriesSchedule(max_steps; step_last = Ref(0))

Schedule that fires on powers of two, and at least once every `max_steps` steps.

The result is an exponential ramp-down in frequency that saturates to a constant one:
useful for reporting that should be dense at the start of a simulation, when a user is
still watching for trouble, and sparse later.

# Fields

  - `max_steps`: Longest allowed gap between firings [-].
  - `step_last`: `Ref` holding the step at which the schedule last fired [-].
"""
struct CappedGeometricSeriesSchedule <: AbstractSchedule
    # GeometricSeriesSchedule(integrator) is true every 2^N iterations or every max_steps
    max_steps::Int
    # Last step that this returned true
    step_last::Base.RefValue{Int}

    function CappedGeometricSeriesSchedule(max_steps; step_last = Ref(0))
        return new(max_steps, step_last)
    end
end

"""
    (schedule::CappedGeometricSeriesSchedule)(integrator)

Return whether the schedule fires on this step.

True when `integrator.step` is a power of two, or when more than `max_steps` steps have
passed since the last firing. Mutates `schedule.step_last` whenever it returns `true`.
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
