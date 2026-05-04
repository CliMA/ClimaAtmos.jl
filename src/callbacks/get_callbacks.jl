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

Convert a list of YAML-style diagnostic spec dicts into a flat
`Vector{ScheduledDiagnostic}`. Each spec must contain at least `short_name`
and `period`; supports optional `reduction_time`, `writer`, `output_name`,
`pressure_coordinates`, and `compute_every`.

`writers` is a tuple `(dict, hdf5, netcdf [, pressure_netcdf])` whose
instances are bound to the resulting diagnostics' `output_writer` fields.
Errors if any spec requests pressure coordinates but the writers tuple does
not include a pressure NetCDFWriter.
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

Parse a frequency (e.g. "3months", "2steps", "10mins") into a schedule for
diagnostics.
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

    date_last =
        t_start isa ITime ?
        ClimaUtilities.TimeManager.date(t_start) :
        start_date + Dates.Second(t_start)

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
            CA.promote_period.(Dates.Second(period_seconds))
    end

    return CAD.EveryCalendarDtSchedule(
        period_dates;
        reference_date = start_date,
        date_last = date_last,
    )
end

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

Validate that checkpoint frequency is an integer multiple of all diagnostics accumulation periods.

Warns if inconsistent, which could prevent safe restarts from checkpoints.
"""
function validate_checkpoint_diagnostics_consistency(
    checkpoint_frequency,
    periods_reductions,
)
    if checkpoint_frequency != Inf
        if any(x -> !CA.isdivisible(checkpoint_frequency, x), periods_reductions)
            accum_str = join(CA.promote_period.(collect(periods_reductions)), ", ")
            checkpt_str = CA.promote_period(checkpoint_frequency)
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

function progress_logging_callback(dt, t_start, t_end)
    walltime_info = WallTimeInfo()
    tot_steps = ceil(Int, (t_end - t_start) / dt)
    five_percent_steps = ceil(Int, 0.05 * tot_steps)
    schedule = CappedGeometricSeriesSchedule(five_percent_steps)
    cond = (u, t, integrator) -> schedule(integrator)
    affect! = (integrator) -> report_walltime(walltime_info, integrator)
    return (CTS.DiscreteCallback(cond, affect!),)
end

function nan_checking_callback(check_nan_every::Int)
    if check_nan_every > 0
        return (
            call_every_n_steps((integrator) -> check_nans(integrator), check_nan_every),
        )
    end
    return ()
end

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

function checkpoint_callback(
    checkpoint_frequency,
    output_dir,
    start_date,
    t_start,
)
    if checkpoint_frequency != Inf
        schedule = CAD.EveryCalendarDtSchedule(
            checkpoint_frequency;
            reference_date = start_date,
            date_last = t_start isa ITime ?
                        ClimaUtilities.TimeManager.date(t_start) :
                        start_date + Dates.Second(t_start),
        )
        cond = (u, t, integrator) -> schedule(integrator)
        affect! = (integrator) -> save_state_to_disk_func(integrator, output_dir)
        return (CTS.DiscreteCallback(cond, affect!),)
    end
    return ()
end

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

function conservation_checking_callback()
    return (
        call_every_n_steps(
            flux_accumulation!;
            skip_first = true,
            call_at_end = true,
        ),
    )
end

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

Build a `call_every_dt` callback from a frequency string (e.g. "6hours"),
handling ITime/FT conversion, promotion, and checkpoint validation.
"""
function scheduled_callback(
    affect!,
    dt_str,
    dt,
    t_start,
    t_end,
    checkpoint_frequency = nothing,
)
    FT = typeof(dt) <: ITime ? Float64 : (dt isa AbstractFloat ? typeof(dt) : Float64)
    dt_seconds_float = time_to_seconds(dt_str)
    dt_seconds_val = FT(dt_seconds_float)
    dt_seconds =
        dt isa ITime ? ITime(dt_seconds_float) : dt_seconds_val
    dt_seconds, _, _, _ = promote(dt_seconds, t_start, dt, t_end)

    if !isnothing(checkpoint_frequency) && checkpoint_frequency != Inf
        dt_s = Dates.Second(round(Int, dt_seconds_val))
        if !CA.isdivisible(checkpoint_frequency, dt_s)
            @warn "$(nameof(affect!)) period ($dt_s) is not an even divisor of the checkpoint frequency ($checkpoint_frequency). This simulation will not be reproducible when restarted."
        end
    end

    return (call_every_dt(affect!, dt_seconds),)
end

function radiation_callback(
    radiation_mode,
    dt_rad,
    dt,
    t_start,
    t_end,
    checkpoint_frequency,
)
    radiation_mode isa RRTMGPI.AbstractRRTMGPMode || return ()
    return scheduled_callback(
        rrtmgp_model_callback!,
        dt_rad,
        dt,
        t_start,
        t_end,
        checkpoint_frequency,
    )
end

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

function enforce_physical_constraints_callback(dt)
    return call_every_dt(enforce_physical_constraints_callback!, dt)
end

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
    default_model_callbacks(model::AtmosModel; kwargs...)

Creates the tuple of model callbacks for any AtmosModel by calling
`default_model_callbacks` on each physics component. 

# Arguments
- `model::AtmosModel`: The atmospheric model configuration

# Keyword Arguments
- `start_date`: Simulation start date
- `dt`: Simulation time step 
- `t_start`: Start time
- `t_end`: End time
- `output_dir`: Output directory
- `checkpoint_frequency`: Checkpoint frequency
- Component-specific frequency overrides (dt_rad, dt_nogw, etc.)
"""
function default_model_callbacks(model::AtmosModel; kwargs...)
    callbacks = ()
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
    dt_ogw = "6hours",
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

# Walk into AtmosTurbconv so per-turbconv-model dispatches (e.g. PrognosticEDMFX)
# are reachable via the AtmosModel-level loop over top-level fields.
default_model_callbacks(turbconv::AtmosTurbconv; kwargs...) =
    default_model_callbacks(turbconv.turbconv_model; kwargs...)

# Walk into SCMSetup for the SCM external-forcing callback.
default_model_callbacks(scm::SCMSetup; kwargs...) =
    default_model_callbacks(scm.external_forcing; kwargs...)

# Enforce physical constraints filter for PrognosticEDMFX
default_model_callbacks(turbconv_model::PrognosticEDMFX; dt, kwargs...) =
    (enforce_physical_constraints_callback(dt),)

# Single-column external forcing (ReanalysisTimeVarying / ReanalysisMonthlyAveragedDiurnal
# in YAML both construct ExternalDrivenTVForcing).
default_model_callbacks(::ExternalDrivenTVForcing; kwargs...) =
    scm_external_forcing_callback()

"""
    common_callbacks(model, dt, output_dir, start_date, t_start, t_end, comms_ctx, checkpoint_frequency; kwargs...)

Get commonly used callbacks like progress logging, NaN checking, conservation, etc.
These are not model-specific but are frequently needed across simulations.

# Keyword Arguments
- `log_progress::Bool = true`: Emit periodic progress logging callback.
- `check_nan_every::Int = 1024`: Step cadence for the NaN-detection callback.
  Set to `0` to disable.
- `check_conservation::Bool = false`: Enable the conservation-check callback.
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
