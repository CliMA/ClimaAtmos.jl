import ClimaUtilities.OnlineLogging: WallTimeInfo, report_walltime
# Reduction-time keys allowed in diagnostic spec dicts.
# Lowercased so callers can use "Max" or "max" interchangeably.
const _DIAG_ALLOWED_REDUCTIONS = Dict(
    "inst" => (nothing, nothing),       # just dump the variable
    "nothing" => (nothing, nothing),    # also accepts the literal string "nothing"
    "max" => (max, nothing),
    "min" => (min, nothing),
    "average" => ((+), CAD.average_pre_output_hook!),
)

"""
    scheduled_diagnostics_from_specs(specs, Y, t_start, start_date, writers)

Convert YAML-style diagnostic specs into a flat `Vector{ScheduledDiagnostic}`.

A spec's `short_name` may be a list, in which case it expands to one diagnostic per name;
the result is flattened. Errors on an unknown reduction or writer, on a missing `period`,
on combining a list of short names with an `output_name`, and on requesting pressure
coordinates without a pressure writer or with a non-NetCDF writer.

# Arguments

  - `specs`: Iterable of `Dict{String, Any}` diagnostic specs. Each must have `short_name`
    and `period`; `reduction_time`, `writer`, `output_name`, `pressure_coordinates`, and
    `compute_every` are optional. Reduction and writer keys are matched case-insensitively.
  - `Y`: The model state, used only for its float type.
  - `t_start`: Start time of the simulation [s], or an `ITime`. Anchors the schedules.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `writers`: Tuple `(dict, hdf5, netcdf)`, optionally extended with a fourth pressure
    `NetCDFWriter`. The instances are bound to the diagnostics' `output_writer` fields.

# Returns

A `Vector` of `ClimaDiagnostics.ScheduledDiagnostic`.

# Notes

Without an explicit `compute_every`, a reduced diagnostic is computed every step, so the
reduction sees every timestep; an unreduced one is computed only when it is written.
"""
function scheduled_diagnostics_from_specs(
    specs,
    Y,
    t_start,
    start_date,
    writers,
)
    FT = Spaces.undertype(axes(Y.c))

    dict_writer, hdf5_writer, netcdf_writer = writers[1], writers[2], writers[3]
    pressure_netcdf_writer = length(writers) >= 4 ? writers[4] : nothing
    if any(d -> get(d, "pressure_coordinates", false), specs) &&
       isnothing(pressure_netcdf_writer)
        error(
            "diagnostic specs request pressure coordinates, but the writers \
            tuple does not include a pressure NetCDFWriter.",
        )
    end

    allowed_writers = Dict(
        "nothing" => netcdf_writer,
        "dict" => dict_writer,
        "h5" => hdf5_writer,
        "hdf5" => hdf5_writer,
        "nc" => netcdf_writer,
        "netcdf" => netcdf_writer,
    )

    diagnostics_ragged = map(specs) do spec
        short_names = spec["short_name"]
        output_name = get(spec, "output_name", nothing)
        in_pressure_coords = get(spec, "pressure_coordinates", false)

        if short_names isa Vector
            isnothing(output_name) || error(
                "Diagnostics: cannot have multiple short_names while specifying output_name",
            )
        else
            short_names = [short_names]
        end

        map(short_names) do short_name
            reduction_key = lowercase(get(spec, "reduction_time", "nothing"))
            haskey(_DIAG_ALLOWED_REDUCTIONS, reduction_key) ||
                error("reduction $reduction_key not implemented")
            reduction_time_func, pre_output_hook! =
                _DIAG_ALLOWED_REDUCTIONS[reduction_key]

            writer_ext = lowercase(get(spec, "writer", "nothing"))
            haskey(allowed_writers, writer_ext) ||
                error("writer $writer_ext not implemented")
            writer = if in_pressure_coords
                writer_ext in ("netcdf", "nothing") ||
                    error("Writing in pressure coordinates is only \
                    compatible with the NetCDF writer")
                pressure_netcdf_writer
            else
                allowed_writers[writer_ext]
            end

            haskey(spec, "period") ||
                error("period keyword required for diagnostics")

            output_schedule =
                parse_frequency_to_schedule(FT, spec["period"], start_date, t_start)
            compute_schedule =
                parse_frequency_to_schedule(FT, spec["period"], start_date, t_start)

            output_short_name = if isnothing(output_name)
                CAD.descriptive_short_name(
                    CAD.get_diagnostic_variable(short_name),
                    output_schedule,
                    reduction_time_func,
                    pre_output_hook!,
                )
            else
                output_name
            end

            compute_every = if isnothing(reduction_time_func)
                compute_schedule
            elseif !haskey(spec, "compute_every")
                CAD.EveryStepSchedule()
            else
                parse_frequency_to_schedule(
                    FT, spec["compute_every"], start_date, t_start,
                )
            end

            CAD.ScheduledDiagnostic(
                variable = CAD.get_diagnostic_variable(short_name),
                output_schedule_func = output_schedule,
                compute_schedule_func = compute_every,
                reduction_time_func = reduction_time_func,
                pre_output_hook! = pre_output_hook!,
                output_writer = writer,
                output_short_name = output_short_name,
            )
        end
    end

    return collect(Iterators.flatten(diagnostics_ragged))
end

"""
    parse_frequency_to_schedule(
        ::Type{FT},
        frequency_str,
        start_date,
        t_start,
    )

Parse a frequency string into a diagnostics schedule.

Recognizes `"<N>steps"`, which becomes a `DivisorSchedule` firing every `N` steps;
`"<N>months"`; the calendar-aligned aliases `"monthly"`, `"weekly"`, and `"daily"`, which
snap the anchor back to the start of the containing month, week, or day; and anything else
`time_to_seconds` understands, such as `"10mins"` or `"6hours"`. Everything but the step
form yields an `EveryCalendarDtSchedule` anchored at the date of `t_start`.

# Arguments

  - `FT`: Float type used to parse a duration in seconds.
  - `frequency_str`: The frequency, as a string.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.
"""
function parse_frequency_to_schedule(
    ::Type{FT},
    frequency_str,
    start_date,
    t_start,
) where {FT}
    if occursin("steps", frequency_str)
        steps = match(r"^(\d+)steps$", frequency_str)
        isnothing(steps) && error(
            "$(frequency_str) has to be of the form <NUM>steps, e.g. 2steps for 2 steps",
        )
        steps = parse(Int, first(steps))
        return CAD.DivisorSchedule(steps)
    end

    date_last = ClimaUtilities.TimeManager.date(t_start)

    if occursin("months", frequency_str)
        months = match(r"^(\d+)months$", frequency_str)
        isnothing(months) && error(
            "$(frequency_str) has to be of the form <NUM>months, e.g. 2months for 2 months",
        )
        period_dates = Dates.Month(parse(Int, first(months)))
    elseif frequency_str == "monthly"
        period_dates = Dates.Month(1)
        date_last = Dates.firstdayofmonth(date_last)
    elseif frequency_str == "weekly"
        period_dates = Dates.Week(1)
        date_last = date_last - Dates.Day(Dates.dayofweek(date_last) - 1)
    elseif frequency_str == "daily"
        period_dates = Dates.Day(1)
        # Converting to a Date clears the time information (e.g. hours, minutes,
        # seconds, etc)
        date_last = Dates.DateTime(Dates.Date(date_last))
    else
        period_seconds = FT(time_to_seconds(frequency_str))
        period_dates =
            promote_period.(Dates.Second(period_seconds))
    end

    return CAD.EveryCalendarDtSchedule(
        period_dates;
        start_date,
        date_last = date_last,
    )
end

"""
    parse_checkpoint_frequency(period::Number)
    parse_checkpoint_frequency(period_str::AbstractString)

Normalize a user-supplied checkpointing frequency.

A number is read as seconds; a string is either `"<N>months"` or anything
`time_to_seconds` understands, such as `"10days"`. `Inf` passes through unchanged and
means no checkpointing, which is how `checkpoint_callback` and
`validate_checkpoint_diagnostics_consistency` detect that they have nothing to do.

# Returns

A `Dates.Second`, a `Dates.Month`, or `Inf`.
"""
function parse_checkpoint_frequency(period::Number)
    period == Inf && return Inf
    # Treat number as seconds
    return Dates.Second(round(Int, period))
end
function parse_checkpoint_frequency(period_str::AbstractString)
    if occursin("months", period_str)
        months = match(r"^(\d+)months$", period_str)
        isnothing(months) && error(
            "Checkpoint frequency has to be of the form <NUM>months, e.g. \"2months\" for 2 months",
        )
        return Dates.Month(parse(Int, first(months)))
    end
    checkpoint_frequency = time_to_seconds(period_str)
    checkpoint_frequency == Inf && return Inf
    return Dates.Second(round(Int, checkpoint_frequency))
end

"""
    validate_checkpoint_diagnostics_consistency(checkpoint_frequency, periods_reductions)

Warn if the checkpointing frequency is not a multiple of every diagnostic accumulation
period.

A checkpoint taken mid-accumulation cannot restore the partially accumulated diagnostic,
so restarting from it silently changes the affected time means. Making the checkpoint
frequency an integer multiple of all accumulation periods guarantees that every checkpoint
falls on a window boundary. Does nothing when `checkpoint_frequency` is `Inf`.

# Arguments

  - `checkpoint_frequency`: Normalized frequency from `parse_checkpoint_frequency`.
  - `periods_reductions`: Accumulation periods, as returned by `extract_diagnostic_periods`.

# Returns

`nothing`. Inconsistencies are reported with `@warn`, not raised.
"""
function validate_checkpoint_diagnostics_consistency(
    checkpoint_frequency,
    periods_reductions,
)
    if checkpoint_frequency != Inf
        if any(x -> !isdivisible(checkpoint_frequency, x), periods_reductions)
            accum_str = join(promote_period.(collect(periods_reductions)), ", ")
            checkpt_str = promote_period(checkpoint_frequency)
            @warn """The checkpointing frequency \
            (checkpoint_frequency = $checkpt_str) should be an integer \
            multiple of all diagnostics accumulation periods ($accum_str) \
            so simulations can be safely restarted from any checkpoint"""
        end
    end
end

#####
##### Reusable callback builder functions
#####
#
# Every builder below returns a *tuple* of callbacks, empty when the feature is switched
# off, so that callers can splat them together unconditionally. The model-specific ones are
# assembled by `default_model_callbacks` and the rest by `common_callbacks`.

"""
    progress_logging_callback(dt, t_start, t_end)

Build the walltime-reporting callback.

Reports on a `CappedGeometricSeriesSchedule` capped at 5% of the total steps: frequently at
the start of a run, when a user is still checking that it is healthy, then at most twenty
times over the remainder.

# Returns

A one-element tuple holding a `DiscreteCallback`.
"""
function progress_logging_callback(dt, t_start, t_end)
    walltime_info = WallTimeInfo()
    tot_steps = ceil(Int, (t_end - t_start) / dt)
    five_percent_steps = ceil(Int, 0.05 * tot_steps)
    schedule = CappedGeometricSeriesSchedule(five_percent_steps)
    cond = (u, t, integrator) -> schedule(integrator)
    affect! = (integrator) -> report_walltime(walltime_info, integrator)
    return (CTS.DiscreteCallback(cond, affect!),)
end

"""
    nan_checking_callback(check_nan_every::Int)

Build the callback that scans the prognostic state for `NaN`.

Returns an empty tuple when `check_nan_every` is zero or negative, disabling the check.
See `check_nans`.
"""
function nan_checking_callback(check_nan_every::Int)
    if check_nan_every > 0
        return (
            call_every_n_steps((integrator) -> check_nans(integrator), check_nan_every),
        )
    end
    return ()
end

"""
    graceful_exit_callback(output_dir)

Build the callback that lets a user stop a running simulation from the filesystem.

Polls `maybe_graceful_exit` every step and calls `terminate!` when a stop has been
requested, so the run ends through the integrator with its output intact.

# Returns

A one-element tuple holding a `DiscreteCallback`.
"""
function graceful_exit_callback(output_dir)
    return (
        call_every_n_steps(
            terminate!;
            skip_first = true,
            condition = (u, t, integrator) ->
                maybe_graceful_exit(output_dir, integrator),
        ),
    )
end

"""
    checkpoint_callback(checkpoint_frequency, output_dir, start_date, t_start)

Build the callback that writes state checkpoints to `output_dir`.

Fires on a calendar schedule anchored at the date of `t_start`. Returns an empty tuple
when `checkpoint_frequency` is `Inf`, disabling checkpointing. See
`save_state_to_disk_func` for the file naming and contents.
"""
function checkpoint_callback(
    checkpoint_frequency,
    output_dir,
    start_date,
    t_start,
)
    if checkpoint_frequency != Inf
        schedule = CAD.EveryCalendarDtSchedule(
            checkpoint_frequency;
            start_date,
            date_last = ClimaUtilities.TimeManager.date(t_start),
        )
        cond = (u, t, integrator) -> schedule(integrator)
        affect! = (integrator) -> save_state_to_disk_func(integrator, output_dir)
        return (CTS.DiscreteCallback(cond, affect!),)
    end
    return ()
end

"""
    gc_callback(comms_ctx)

Build the callback that forces periodic garbage collection on distributed runs.

Returns an empty tuple on a non-distributed context, leaving Julia's automatic collection
alone. On a distributed context, ranks that collect at different times stall each other at
the next communication, so GC is instead forced on a common step cadence: every
`CLIMAATMOS_GC_NSTEPS` steps, default `1000`, skipping the call at initialization.

# Examples

```shell
# Collect every 200 steps instead of the default 1000.
CLIMAATMOS_GC_NSTEPS=200 mpiexec julia --project=.buildkite .buildkite/ci_driver.jl
```

See `gc_func` for what each collection logs.
"""
function gc_callback(comms_ctx)
    if is_distributed(comms_ctx)
        return (
            call_every_n_steps(
                gc_func,
                parse(Int, get(ENV, "CLIMAATMOS_GC_NSTEPS", "1000")),
                skip_first = true,
            ),
        )
    end
    return ()
end

"""
    conservation_checking_callback()

Build the callback that accumulates boundary energy fluxes for the conservation check.

Runs every step, skipping initialization and including the final step, so the accumulated
flux spans exactly the integrated interval. See `flux_accumulation!`.
"""
function conservation_checking_callback()
    return (
        call_every_n_steps(
            flux_accumulation!;
            skip_first = true,
            call_at_end = true,
        ),
    )
end

"""
    scm_external_forcing_callback()

Build the callback that refreshes single-column external forcing every step.

See `external_driven_single_column!`. Installed by the `default_model_callbacks` method for
`ExternalDrivenTVForcing`.
"""
function scm_external_forcing_callback()
    return (
        call_every_n_steps(
            external_driven_single_column!;
            call_at_end = true,
        ),
    )
end

"""
    scheduled_callback(affect!, dt_str, dt, t_start, t_end[, checkpoint_frequency])

Build a `call_every_dt` callback from a frequency string.

Shared backend of the physics-component callbacks. Converts `dt_str` (e.g. `"6hours"`) to
seconds, wraps it in an `ITime`, and promotes it against the simulation's time
quantities so the arithmetic stays exact.

# Arguments

  - `affect!`: Step function called as `affect!(integrator)`.
  - `dt_str`: Interval between calls, as a frequency string.
  - `dt`, `t_start`, `t_end`: Simulation timestep, start, and end, used for promotion.
  - `checkpoint_frequency = nothing`: When given and finite, warns if the callback period is
    not an even divisor of it. A callback that fires at a different phase after a restart
    makes the run non-reproducible, since its effect persists between calls.

# Returns

A one-element tuple holding a `DiscreteCallback`.
"""
function scheduled_callback(
    affect!,
    dt_str,
    dt,
    t_start,
    t_end,
    checkpoint_frequency = nothing,
)
    dt_seconds_float = time_to_seconds(dt_str)
    dt_seconds = ITime(dt_seconds_float)
    dt_seconds, _, _, _ = promote(dt_seconds, t_start, dt, t_end)

    if !isnothing(checkpoint_frequency) && checkpoint_frequency != Inf
        dt_s = Dates.Second(round(Int, dt_seconds_float))
        if !isdivisible(checkpoint_frequency, dt_s)
            @warn "$(nameof(affect!)) period ($dt_s) is not an even divisor of the checkpoint frequency ($checkpoint_frequency). This simulation will not be reproducible when restarted."
        end
    end

    return (call_every_dt(affect!, dt_seconds),)
end

"""
    radiation_callback(radiation_mode, dt_rad, dt, t_start, t_end, checkpoint_frequency)

Build the callback that runs the RRTMGP radiation solve every `dt_rad`.

Returns an empty tuple for any non-RRTMGP mode. See `rrtmgp_solver_callback!`.
"""
function radiation_callback(
    radiation_mode,
    dt_rad,
    dt,
    t_start,
    t_end,
    checkpoint_frequency,
)
    # `dt_rad` only governs RRTMGP-style radiation. For `rad: held_suarez`
    # (and any other non-RRTMGP mode), the forcing is folded into
    # `remaining_tendency!` and applied every stage; `dt_rad` is ignored.
    radiation_mode isa RRTMGPI.AbstractRRTMGPMode || return ()
    return scheduled_callback(
        rrtmgp_solver_callback!,
        dt_rad,
        dt,
        t_start,
        t_end,
        checkpoint_frequency,
    )
end

"""
    subcol_callback(dt_subcol, dt, t_start, t_end, checkpoint_frequency)

Build the callback that generates the COSP subcolumns every `dt_subcol`.

Unconditional: whether COSP runs at all is decided by `subcol_callback_enabled`. See
`subcol_model_callback!`.
"""
function subcol_callback(
    dt_subcol,
    dt,
    t_start,
    t_end,
    checkpoint_frequency,
)
    return scheduled_callback(
        subcol_model_callback!,
        dt_subcol,
        dt,
        t_start,
        t_end,
        checkpoint_frequency,
    )
end

"""
    subcol_callback_enabled(model::AtmosModel, dt_subcol)

Return whether the COSP subcolumn callback should be installed.

True only when the model configures COSP *and* `dt_subcol` is finite, so `dt_subcol = "Inf"` switches COSP off without changing the model.
"""
subcol_callback_enabled(model::AtmosModel, dt_subcol) =
    !isnothing(model.cosp) && time_to_seconds(dt_subcol) != Inf

"""
    nogw_callback(non_orographic_gravity_wave, dt_nogw, dt, t_start, t_end,
                  checkpoint_frequency)

Build the callback that recomputes non-orographic gravity-wave drag every `dt_nogw`.

Returns an empty tuple when the component is not a `NonOrographicGravityWave`. See
`nogw_model_callback!`.
"""
function nogw_callback(
    non_orographic_gravity_wave,
    dt_nogw,
    dt,
    t_start,
    t_end,
    checkpoint_frequency,
)
    non_orographic_gravity_wave isa NonOrographicGravityWave || return ()
    return scheduled_callback(
        nogw_model_callback!,
        dt_nogw,
        dt,
        t_start,
        t_end,
        checkpoint_frequency,
    )
end

"""
    ogw_callback(orographic_gravity_wave, dt_ogw, dt, t_start, t_end,
                 checkpoint_frequency)

Build the callback that recomputes orographic gravity-wave drag every `dt_ogw`.

Returns an empty tuple when the component is not an `OrographicGravityWave`. See
`ogw_model_callback!`.
"""
function ogw_callback(
    orographic_gravity_wave,
    dt_ogw,
    dt,
    t_start,
    t_end,
    checkpoint_frequency,
)
    orographic_gravity_wave isa OrographicGravityWave || return ()
    return scheduled_callback(
        ogw_model_callback!,
        dt_ogw,
        dt,
        t_start,
        t_end,
        checkpoint_frequency,
    )
end


"""
    default_model_callbacks(model::AtmosModel; dt_subcol = "Inf", kwargs...)
    default_model_callbacks(component; kwargs...)

Assemble the physics callbacks a model needs, by asking each of its components.

The `AtmosModel` method handles COSP itself, then walks every field of the model — all but
`disable_surface_flux_tendency`, which is a flag rather than a component — and
concatenates what each returns. The fallback method returns `()`, so a component
contributes callbacks only if it defines a method. Each component method also decides
whether it is active at all: radiation returns `()` for non-RRTMGP modes, gravity waves
for absent wave models, and so on.

Every callback here exists because its parameterization is too expensive to evaluate each
step. Each is recomputed on its own `dt_*` cadence and held fixed in the cache in between,
so a longer cadence buys speed at the cost of resolving the parameterization in time.

# Arguments

  - `model`: The `AtmosModel`, or one of its components.

# Keyword Arguments

  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `dt`: Simulation timestep [s].
  - `t_start`, `t_end`: Start and end times of the simulation [s].
  - `output_dir`: Output directory.
  - `checkpoint_frequency`: Normalized checkpointing frequency, used to warn about cadences
    that would break restart reproducibility. See `scheduled_callback`.
  - `dt_subcol = "Inf"`: COSP subcolumn cadence. Consumed by the `AtmosModel` method and not
    forwarded; the default disables COSP.
  - `dt_rad = "6hours"`: RRTMGP radiation cadence, on the `AtmosRadiation` method.
  - `dt_nogw = "3hours"`, `dt_ogw = "3hours"`: Gravity-wave cadences, on the
    `AtmosGravityWave` method.

# Returns

A tuple of `DiscreteCallback`, possibly empty.

# Examples

Adding a callback for a new component is a matter of defining one method. Cadences reach
it through `callback_kwargs`:

```julia
import ClimaAtmos as CA

CA.default_model_callbacks(forcing::MyCaseForcing; dt = nothing, t_start, t_end,
    checkpoint_frequency, kwargs...) =
    CA.scheduled_callback(
        my_forcing_callback!, "1hours", dt, t_start, t_end, checkpoint_frequency,
    )

simulation = CA.AtmosSimulation{Float64}(; callback_kwargs = (; dt_rad = "3hours"))
```
"""
function default_model_callbacks(model::AtmosModel;
    dt_subcol = "Inf",
    kwargs...,
)
    callbacks = ()
    if subcol_callback_enabled(model, dt_subcol)
        callbacks = (
            callbacks...,
            subcol_callback(
                dt_subcol,
                kwargs[:dt],
                kwargs[:t_start],
                kwargs[:t_end],
                kwargs[:checkpoint_frequency],
            )...,
        )
    end
    model_component_names =
        filter(x -> x !== :disable_surface_flux_tendency, propertynames(model))
    for property in model_component_names
        component_callbacks =
            default_model_callbacks(getproperty(model, property); kwargs...)
        callbacks = (callbacks..., component_callbacks...)
    end
    return callbacks
end

default_model_callbacks(component; kwargs...) = ()

function default_model_callbacks(radiation::AtmosRadiation;
    dt_rad = "6hours",
    start_date,
    dt,
    t_start,
    t_end,
    checkpoint_frequency,
    kwargs...)
    return radiation_callback(
        radiation.radiation_mode,
        dt_rad,
        dt,
        t_start,
        t_end,
        checkpoint_frequency,
    )
end

# Gravity-wave component callbacks (both orographic and non-orographic)
function default_model_callbacks(gravity_wave::AtmosGravityWave;
    dt_nogw = "3hours",
    dt_ogw = "3hours",
    start_date,
    dt,
    t_start,
    t_end,
    checkpoint_frequency,
    kwargs...)
    return (
        nogw_callback(
            gravity_wave.non_orographic_gravity_wave,
            dt_nogw, dt, t_start, t_end, checkpoint_frequency,
        )...,
        ogw_callback(
            gravity_wave.orographic_gravity_wave,
            dt_ogw, dt, t_start, t_end, checkpoint_frequency,
        )...,
    )
end

default_model_callbacks(scm::SCMSetup; kwargs...) =
    default_model_callbacks(scm.external_forcing; kwargs...)

default_model_callbacks(::ExternalDrivenTVForcing; kwargs...) =
    scm_external_forcing_callback()

"""
    common_callbacks(model, dt, output_dir, start_date, t_start, t_end, comms_ctx,
                     checkpoint_frequency; kwargs...)

Assemble the infrastructure callbacks that are not tied to any physics component.

In order: progress logging, NaN detection, graceful exit, checkpointing, garbage
collection, and conservation checking. Graceful exit is always installed; the rest are
conditional on their keyword arguments, on `checkpoint_frequency` being finite, or on the
context being distributed.

Together with `default_model_callbacks`, this is what `AtmosSimulation` installs when its
`default_callbacks` keyword is `true`, which is the default. Setting `default_callbacks = false` *replaces* both sets with the user's `callbacks` tuple rather than adding to them,
so a simulation configured that way has no checkpointing, NaN checks, or progress logging
unless the user supplies them.

# Arguments

  - `model`: The `AtmosModel`. Currently unused; accepted for signature stability.
  - `dt`: Simulation timestep [s].
  - `output_dir`: Directory for checkpoints and the graceful-exit file.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`, `t_end`: Start and end times of the simulation [s].
  - `comms_ctx`: `ClimaComms` context; garbage collection is forced only if it is
    distributed.
  - `checkpoint_frequency`: Normalized checkpointing frequency, or `Inf` for none.

# Keyword Arguments

  - `log_progress::Bool = true`: Whether to emit periodic walltime reports.
  - `check_nan_every::Int = 1024`: Step cadence of the NaN-detection callback. Set to `0` to
    disable. Scanning the whole state is not free, hence the cadence.
  - `check_conservation::Bool = false`: Whether to accumulate boundary energy fluxes for the
    conservation diagnostics.

Unrecognized keyword arguments are ignored, so the same `callback_kwargs` can be passed to
both this and `default_model_callbacks`.

# Returns

A tuple of `DiscreteCallback`.
"""
function common_callbacks(
    model, dt, output_dir, start_date, t_start, t_end, comms_ctx, checkpoint_frequency;
    log_progress::Bool = true,
    check_nan_every::Int = 1024,
    check_conservation::Bool = false,
    kwargs...,
)
    callbacks = ()

    # Progress logging
    if log_progress
        callbacks = (callbacks..., progress_logging_callback(dt, t_start, t_end)...)
    end

    # NaN checking
    callbacks = (callbacks..., nan_checking_callback(check_nan_every)...)

    # Graceful exit
    callbacks = (callbacks..., graceful_exit_callback(output_dir)...)

    # Checkpointing
    callbacks = (
        callbacks...,
        checkpoint_callback(checkpoint_frequency, output_dir, start_date, t_start)...,
    )

    # Garbage collection
    callbacks = (callbacks..., gc_callback(comms_ctx)...)

    # Conservation checking
    if check_conservation
        callbacks = (callbacks..., conservation_checking_callback()...)
    end
    return callbacks
end
