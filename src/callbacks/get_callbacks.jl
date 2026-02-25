function get_diagnostics(
    parsed_args,
    atmos_model,
    Y, p, dt,
    t_start, start_date, output_dir,
)

    FT = Spaces.undertype(axes(Y.c))

    # We either get the diagnostics section in the YAML file, or we return an empty list
    # (which will result in an empty list being created by the map below)
    yaml_diagnostics = get(parsed_args, "diagnostics", [])

    # ALLOWED_REDUCTIONS is the collection of reductions we support. The keys are the
    # strings that have to be provided in the YAML file. The values are tuples with the
    # function that has to be passed to reduction_time_func and the one that has to passed
    # to pre_output_hook!

    # We make "nothing" a string so that we can accept also the word "nothing", in addition
    # to the absence of the value
    #
    # NOTE: Everything has to be lowercase in ALLOWED_REDUCTIONS (so that we can match
    # "max" and "Max")
    ALLOWED_REDUCTIONS = Dict(
        "inst" => (nothing, nothing), # nothing is: just dump the variable
        "nothing" => (nothing, nothing),
        "max" => (max, nothing),
        "min" => (min, nothing),
        "average" => ((+), CAD.average_pre_output_hook!),
    )

    dict_writer = CAD.DictWriter()
    hdf5_writer = CAD.HDF5Writer(output_dir)

    if !isnothing(parsed_args["netcdf_interpolation_num_points"])
        num_netcdf_points =
            tuple(parsed_args["netcdf_interpolation_num_points"]...)
    else
        # From the given space, calculate the number of diagnostic grid points if not
        # specified by the user
        num_netcdf_points = default_netcdf_points(axes(Y.c), parsed_args)
    end

    z_sampling_method =
        parsed_args["netcdf_output_at_levels"] ? CAD.LevelsMethod() :
        CAD.FakePressureLevelsMethod()

    # The start_date keyword was added in v0.2.9. For prior versions, the diagnostics will
    # not contain the date
    maybe_add_start_date =
        pkgversion(CAD.ClimaDiagnostics) >= v"0.2.9" ? (; start_date) : (;)

    netcdf_writer = CAD.NetCDFWriter(
        axes(Y.c),
        output_dir,
        num_points = num_netcdf_points;
        z_sampling_method,
        sync_schedule = CAD.EveryStepSchedule(),
        init_time = t_start,
        maybe_add_start_date...,
    )

    # Create NetCDF writer for diagnostics in pressure coordinates if they
    # exist
    write_in_pressure_coords = any(yaml_diagnostics) do yaml_diag
        get(yaml_diag, "pressure_coordinates", false)
    end
    pressure_netcdf_writer = nothing
    if write_in_pressure_coords
        z_sampling_method = ClimaDiagnostics.Writers.RealPressureLevelsMethod(
            p.precomputed.ᶜp,
            t_start,
        )
        pressure_space = ClimaDiagnostics.Writers.pressure_space(z_sampling_method)
        pressure_netcdf_writer = CAD.NetCDFWriter(
            pressure_space,
            output_dir,
            num_points = num_netcdf_points;
            z_sampling_method,
            sync_schedule = CAD.EveryStepSchedule(),
            init_time = t_start,
            maybe_add_start_date...,
        )
    end

    writers = (dict_writer, hdf5_writer, netcdf_writer)
    isnothing(pressure_netcdf_writer) || (writers = (writers..., pressure_netcdf_writer))

    # The default writer is netcdf
    ALLOWED_WRITERS = Dict(
        "nothing" => netcdf_writer,
        "dict" => dict_writer,
        "h5" => hdf5_writer,
        "hdf5" => hdf5_writer,
        "nc" => netcdf_writer,
        "netcdf" => netcdf_writer,
    )

    diagnostics_ragged = map(yaml_diagnostics) do yaml_diag
        short_names = yaml_diag["short_name"]
        output_name = get(yaml_diag, "output_name", nothing)
        in_pressure_coords = get(yaml_diag, "pressure_coordinates", false)

        if short_names isa Vector
            isnothing(output_name) || error(
                "Diagnostics: cannot have multiple short_names while specifying output_name",
            )
        else
            short_names = [short_names]
        end

        map(short_names) do short_name
            # Return "nothing" if "reduction_time" is not in the YAML block
            #
            # We also normalize everything to lowercase, so that can accept "max" but
            # also "Max"
            reduction_time_yaml =
                lowercase(get(yaml_diag, "reduction_time", "nothing"))

            if !haskey(ALLOWED_REDUCTIONS, reduction_time_yaml)
                error("reduction $reduction_time_yaml not implemented")
            else
                reduction_time_func, pre_output_hook! =
                    ALLOWED_REDUCTIONS[reduction_time_yaml]
            end

            writer_ext = lowercase(get(yaml_diag, "writer", "nothing"))

            if !haskey(ALLOWED_WRITERS, writer_ext)
                error("writer $writer_ext not implemented")
            else
                writer = if in_pressure_coords
                    writer_ext in ("netcdf", "nothing") ||
                        error("Writing in pressure coordinates is only \
                        compatible with the NetCDF writer")
                    pressure_netcdf_writer
                else
                    ALLOWED_WRITERS[writer_ext]
                end
            end

            haskey(yaml_diag, "period") ||
                error("period keyword required for diagnostics")

            period_str = yaml_diag["period"]
            output_schedule =
                parse_frequency_to_schedule(FT, period_str, start_date, t_start)
            compute_schedule =
                parse_frequency_to_schedule(FT, period_str, start_date, t_start)

            if isnothing(output_name)
                output_short_name = CAD.descriptive_short_name(
                    CAD.get_diagnostic_variable(short_name),
                    output_schedule,
                    reduction_time_func,
                    pre_output_hook!,
                )
            end

            if isnothing(reduction_time_func)
                compute_every = compute_schedule
            elseif !("compute_every" in keys(yaml_diag))
                compute_every = CAD.EveryStepSchedule()
            else
                compute_every_str = yaml_diag["compute_every"]
                compute_every =
                    parse_frequency_to_schedule(FT, compute_every_str, start_date, t_start)
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

    # Flatten the array of arrays of diagnostics
    diagnostics = vcat(diagnostics_ragged...)

    if parsed_args["output_default_diagnostics"]
        diagnostics = [
            CAD.default_diagnostics(
                atmos_model,
                dt isa ITime ?
                ITime(time_to_seconds(parsed_args["t_end"])) - t_start :
                FT(time_to_seconds(parsed_args["t_end"]) - t_start),
                start_date,
                t_start;
                output_writer = netcdf_writer,
                topography = has_topography(axes(Y.c)),
            )...,
            diagnostics...,
        ]
    end
    diagnostics = collect(diagnostics)

    periods_reductions = extract_diagnostic_periods(diagnostics)
    periods_str = join(CA.promote_period.(periods_reductions), ", ")
    @info "Saving accumulated diagnostics to disk with frequency: $(periods_str)"

    for writer in writers
        writer_str = nameof(typeof(writer))
        diags_with_writer =
            filter((x) -> getproperty(x, :output_writer) == writer, diagnostics)
        diags_outputs = [
            getproperty(diag, :output_short_name) for diag in diags_with_writer
        ]
        @info "$writer_str: $diags_outputs"
    end

    return diagnostics, writers, periods_reductions
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

    if occursin("months", frequency_str)
        months = match(r"^(\d+)months$", frequency_str)
        isnothing(months) && error(
            "$(frequency_str) has to be of the form <NUM>months, e.g. 2months for 2 months",
        )
        period_dates = Dates.Month(parse(Int, first(months)))
    else
        period_seconds = FT(time_to_seconds(frequency_str))
        period_dates =
            CA.promote_period.(Dates.Second(period_seconds))
    end

    date_last =
        t_start isa ITime ?
        ClimaUtilities.TimeManager.date(t_start) :
        start_date + Dates.Second(t_start)
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
    return (SciMLBase.DiscreteCallback(cond, affect!),)
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
        return (SciMLBase.DiscreteCallback(cond, affect!),)
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

function cloud_fraction_callback(
    dt_cloud_fraction,
    dt,
    t_start,
    t_end;
)
    FT = typeof(dt) <: ITime ? Float64 : (dt isa AbstractFloat ? typeof(dt) : Float64)
    dt_cf_seconds =
        dt isa ITime ?
        ITime(time_to_seconds(dt_cloud_fraction)) :
        FT(time_to_seconds(dt_cloud_fraction))
    dt_cf_seconds, _, _, _ = promote(dt_cf_seconds, t_start, dt, t_end)
    return (call_every_dt(cloud_fraction_model_callback!, dt_cf_seconds),)
end

function radiation_callback(
    radiation_mode,
    dt_rad,
    dt,
    t_start,
    t_end,
    checkpoint_frequency;
)
    radiation_mode isa RRTMGPI.AbstractRRTMGPMode || return ()

    # Determine float type: use Float64 if dt is ITime (since float(ITime) -> Float64),
    # otherwise preserve the float type of dt or default to Float64
    FT = typeof(dt) <: ITime ? Float64 : (dt isa AbstractFloat ? typeof(dt) : Float64)

    # Convert dt_rad string to seconds (once)
    dt_rad_seconds_float = time_to_seconds(dt_rad)

    # Compute float value for validation BEFORE promotion
    # We need this separate value because after promotion with ITime, we can't easily
    # extract the numeric value for rounding to Int
    dt_rad_seconds_val = FT(dt_rad_seconds_float)

    # Create dt_rad_seconds matching the type of dt (ITime if dt is ITime, else FT)
    # This ensures type consistency before promotion
    dt_rad_seconds =
        dt isa ITime ?
        ITime(dt_rad_seconds_float) :
        dt_rad_seconds_val

    # Promote dt_rad_seconds with other time values to ensure all have compatible types
    # (e.g., if dt is ITime, all promoted values will be ITime with matching periods)
    dt_rad_seconds, _, _, _ = promote(dt_rad_seconds, t_start, dt, t_end)

    # Validation against checkpoint frequency using the float value (before promotion)
    dt_rad_s = Dates.Second(round(Int, dt_rad_seconds_val))
    if checkpoint_frequency != Inf &&
       !CA.isdivisible(checkpoint_frequency, dt_rad_s)
        @warn "Radiation period ($(dt_rad_s)) is not an even divisor of the checkpoint frequency ($checkpoint_frequency)"
        @warn "This simulation will not be reproducible when restarted"
    end

    return (call_every_dt(rrtmgp_model_callback!, dt_rad_seconds),)
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

    # Determine float type: use Float64 if dt is ITime (since float(ITime) -> Float64),
    # otherwise preserve the float type of dt or default to Float64
    FT = typeof(dt) <: ITime ? Float64 : (dt isa AbstractFloat ? typeof(dt) : Float64)

    # Convert dt_nogw string to seconds (once)
    dt_nogw_seconds_float = time_to_seconds(dt_nogw)

    # Compute float value for validation BEFORE promotion
    # We need this separate value because after promotion with ITime, we can't easily
    # extract the numeric value for rounding to Int
    dt_nogw_seconds_val = FT(dt_nogw_seconds_float)

    # Create dt_nogw_seconds matching the type of dt (ITime if dt is ITime, else FT)
    # This ensures type consistency before promotion
    dt_nogw_seconds =
        dt isa ITime ?
        ITime(dt_nogw_seconds_float) :
        dt_nogw_seconds_val

    # Promote dt_nogw_seconds with other time values to ensure all have compatible types
    # (e.g., if dt is ITime, all promoted values will be ITime with matching periods)
    dt_nogw_seconds, _, _, _ = promote(dt_nogw_seconds, t_start, dt, t_end)

    # Validation against checkpoint frequency using the float value (before promotion)
    dt_nogw_s = Dates.Second(round(Int, dt_nogw_seconds_val))
    if checkpoint_frequency != Inf &&
       !CA.isdivisible(checkpoint_frequency, dt_nogw_s)
        @warn "Non-orographic gravity wave period ($(dt_nogw_s)) is not an even divisor of the checkpoint frequency ($checkpoint_frequency)"
        @warn "This simulation will not be reproducible when restarted"
    end

    return (call_every_dt(nogw_model_callback!, dt_nogw_seconds),)
end

function edmfx_filter_callback(
    dt,
)
    return call_every_dt(edmfx_filter_callback!, dt)
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

    # Determine float type: use Float64 if dt is ITime (since float(ITime) -> Float64),
    # otherwise preserve the float type of dt or default to Float64
    FT = typeof(dt) <: ITime ? Float64 : (dt isa AbstractFloat ? typeof(dt) : Float64)

    # Convert dt_ogw string to seconds (once)
    dt_ogw_seconds_float = time_to_seconds(dt_ogw)

    # Compute float value for validation BEFORE promotion
    dt_ogw_seconds_val = FT(dt_ogw_seconds_float)

    # Create dt_ogw_seconds matching the type of dt (ITime if dt is ITime, else FT)
    dt_ogw_seconds =
        dt isa ITime ?
        ITime(dt_ogw_seconds_float) :
        dt_ogw_seconds_val

    # Promote dt_ogw_seconds with other time values to ensure all have compatible types
    dt_ogw_seconds, _, _, _ = promote(dt_ogw_seconds, t_start, dt, t_end)

    # Validation against checkpoint frequency using the float value (before promotion)
    dt_ogw_s = Dates.Second(round(Int, dt_ogw_seconds_val))
    if checkpoint_frequency != Inf &&
       !CA.isdivisible(checkpoint_frequency, dt_ogw_s)
        @warn "Orographic gravity wave period ($(dt_ogw_s)) is not an even divisor of the checkpoint frequency ($checkpoint_frequency)"
        @warn "This simulation will not be reproducible when restarted"
    end

    return (call_every_dt(ogw_model_callback!, dt_ogw_seconds),)
end


function get_callbacks(config, sim_info, atmos, params, Y, p)
    (; parsed_args, comms_ctx) = config
    FT = eltype(params)
    (; dt, output_dir, start_date, t_start, t_end) = sim_info

    callbacks = ()

    # Progress logging
    if parsed_args["log_progress"]
        callbacks = (callbacks..., progress_logging_callback(dt, t_start, t_end)...)
    end

    # NaN checking
    check_nan_every = parsed_args["check_nan_every"]
    callbacks = (callbacks..., nan_checking_callback(check_nan_every)...)

    # Graceful exit
    callbacks = (callbacks..., graceful_exit_callback(output_dir)...)

    # Checkpointing
    checkpoint_frequency = parse_checkpoint_frequency(parsed_args["dt_save_state_to_disk"])
    callbacks = (
        callbacks...,
        checkpoint_callback(
            checkpoint_frequency,
            output_dir,
            start_date,
            t_start,
        )...,
    )

    # Garbage collection
    callbacks = (callbacks..., gc_callback(comms_ctx)...)

    # Conservation checking
    if parsed_args["check_conservation"]
        callbacks = (callbacks..., conservation_checking_callback()...)
    end

    # External forcing
    if parsed_args["external_forcing"] in
       ["ReanalysisTimeVarying", "ReanalysisMonthlyAveragedDiurnal"] &&
       parsed_args["config"] == "column"
        callbacks = (callbacks..., scm_external_forcing_callback()...)
    end



    # Radiation
    callbacks = (
        callbacks...,
        radiation_callback(
            atmos.radiation_mode,
            parsed_args["dt_rad"],
            dt,
            t_start,
            t_end,
            checkpoint_frequency,
        )...,
    )

    # Non-orographic gravity wave
    callbacks = (
        callbacks...,
        nogw_callback(
            atmos.non_orographic_gravity_wave,
            parsed_args["dt_nogw"],
            dt,
            t_start,
            t_end,
            checkpoint_frequency,
        )...,
    )

    # Orographic gravity wave
    callbacks = (
        callbacks...,
        ogw_callback(
            atmos.orographic_gravity_wave,
            parsed_args["dt_ogw"],
            dt,
            t_start,
            t_end,
            checkpoint_frequency,
        )...,
    )

    # EDMFX filter
    callbacks = (
        callbacks...,
        edmfx_filter_callback(dt),
    )

    return callbacks
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


function default_model_callbacks(component; kwargs...)
    return ()
end

function default_model_callbacks(radiation::AtmosRadiation;
    dt_rad,
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
        checkpoint_frequency;
    )
end

# Gravity wave component callbacks
function default_model_callbacks(gravity_wave::AtmosGravityWave;
    dt_nogw = "3hours",
    start_date,
    dt,
    t_start,
    t_end,
    checkpoint_frequency,
    kwargs...)
    return nogw_callback(
        gravity_wave.non_orographic_gravity_wave,
        dt_nogw,
        dt,
        t_start,
        t_end,
        checkpoint_frequency;
    )
end

# EDMFX filter callbacks
function default_model_callbacks(turbconv_model::PrognosticEDMFX;
    dt)
    return edmfx_filter_callback(dt)
end

function default_model_callbacks(water::AtmosWater;
    dt_cloud_fraction,
    start_date,
    dt,
    t_start,
    t_end,
    kwargs...)
    if !isnothing(water.microphysics_model)
        return cloud_fraction_callback(
            dt_cloud_fraction,
            dt,
            t_start,
            t_end,
        )
    end
    return ()
end

"""
    common_callbacks(model, dt, output_dir, start_date, t_start, t_end, comms_ctx; kwargs...)

Get commonly used callbacks like progress logging, NaN checking, conservation, etc.
These are not model-specific but are frequently needed across simulations.

# Keyword Arguments
- `progress_logging = false`: Enable progress reporting
- `checkpoint_frequency`: Frequency for saving state to disk
- `external_forcing_column = false`: Enable external forcing for single column
"""
function common_callbacks(
    model, dt, output_dir, start_date, t_start, t_end, comms_ctx, checkpoint_frequency,
)
    callbacks = ()

    # Progress logging
    callbacks = (callbacks..., progress_logging_callback(dt, t_start, t_end)...)

    # NaN checking
    callbacks = (callbacks..., nan_checking_callback(1024)...)

    # Graceful exit
    callbacks = (callbacks..., graceful_exit_callback(output_dir)...)

    # Checkpointing
    callbacks = (
        callbacks...,
        checkpoint_callback(checkpoint_frequency, output_dir, start_date, t_start)...,
    )

    # Garbage collection
    callbacks = (callbacks..., gc_callback(comms_ctx)...)

    return callbacks
end
