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
    dt_rad = "1h",
    start_date = nothing,
    dt = nothing,
    t_start = 0,
    t_end = nothing,
    checkpoint_frequency = "Inf",
    kwargs...)
    radiation.radiation_mode isa RRTMGPI.AbstractRRTMGPMode || return ()
    # Parse time parameters
    FT = typeof(dt) <: ITime ? Float64 : (dt isa AbstractFloat ? typeof(dt) : Float64)
    dt_rad_seconds =
        dt isa ITime ?
        ITime(time_to_seconds(dt_rad)) :
        FT(time_to_seconds(dt_rad))
    dt_rad_seconds, _, _, _ = promote(dt_rad_seconds, t_start, dt, t_end)

    # Validation against checkpoint frequency  
    dt_save_state_to_disk_dates =
        checkpoint_frequency_from_parsed_args(checkpoint_frequency)
    dt_rad_ms = Dates.Millisecond(1_000 * float(dt_rad_seconds))
    if dt_save_state_to_disk_dates != Inf &&
       !isdivisible(dt_save_state_to_disk_dates, dt_rad_ms)
        @warn "Radiation period ($(dt_rad_ms)) is not an even divisor of the checkpoint frequency ($dt_save_state_to_disk_dates)"
        @warn "This simulation will not be reproducible when restarted"
    end

    @info "Auto-enabled radiation callback: dt_rad = $dt_rad"
    return (call_every_dt(rrtmgp_model_callback!, dt_rad_seconds),)
end

# Gravity wave component callbacks
function default_model_callbacks(gravity_wave::AtmosGravityWave;
    dt_nogw = "3hours",
    start_date = nothing,
    dt = nothing,
    t_start = 0,
    t_end = nothing,
    checkpoint_frequency = "Inf",
    kwargs...)
    if gravity_wave.non_orographic_gravity_wave isa NonOrographicGravityWave
        # Parse time parameters
        FT = typeof(dt) <: ITime ? Float64 : (dt isa AbstractFloat ? typeof(dt) : Float64)
        dt_nogw_seconds =
            dt isa ITime ?
            ITime(time_to_seconds(dt_nogw)) :
            FT(time_to_seconds(dt_nogw))
        dt_nogw_seconds, _, _, _ = promote(dt_nogw_seconds, t_start, dt, t_end)

        # Validation against checkpoint frequency
        dt_save_state_to_disk_dates =
            checkpoint_frequency_from_parsed_args(checkpoint_frequency)
        dt_nogw_ms = Dates.Millisecond(1_000 * float(dt_nogw_seconds))
        if dt_save_state_to_disk_dates != Inf &&
           !isdivisible(dt_save_state_to_disk_dates, dt_nogw_ms)
            @warn "Non-orographic gravity wave period ($(dt_nogw_ms)) is not an even divisor of the checkpoint frequency ($dt_save_state_to_disk_dates)"
            @warn "This simulation will not be reproducible when restarted"
        end

        @info "Auto-enabled non-orographic gravity wave callback: dt_nogw = $dt_nogw"
        return (call_every_dt(nogw_model_callback!, dt_nogw_seconds),)
    end
    return ()
end

function default_model_callbacks(water::AtmosWater;
    dt_cloud_fraction = "3hours",
    call_cloud_diagnostics_per_stage = false,
    start_date = nothing,
    dt = nothing,
    t_start = 0,
    t_end = nothing,
    kwargs...)
    if !call_cloud_diagnostics_per_stage && !isnothing(water.moisture_model)
        # Parse time parameters
        FT = typeof(dt) <: ITime ? Float64 : (dt isa AbstractFloat ? typeof(dt) : Float64)
        dt_cf_seconds =
            dt isa ITime ?
            ITime(time_to_seconds(dt_cloud_fraction)) :
            FT(time_to_seconds(dt_cloud_fraction))
        dt_cf_seconds, _, _, _ = promote(dt_cf_seconds, t_start, dt, t_end)

        @info "Auto-enabled cloud fraction callback: dt_cloud_fraction = $dt_cloud_fraction"
        return (call_every_dt(cloud_fraction_model_callback!, dt_cf_seconds),)
    end
    return ()
end

# backward compatibility for deprecated function
function default_model_callbacks(
    model::AtmosModel,
    sim_info,
    params,
    Y,
    p;
    kwargs...,
)
    (; dt, output_dir, start_date, t_start, t_end) = sim_info
    return default_model_callbacks(model;
        start_date,
        dt,
        t_start,
        t_end,
        output_dir,
        kwargs...)
end

"""
    common_callbacks(dt, output_dir, start_date, t_start, t_end, comms_ctx; kwargs...)

Get commonly used callbacks like progress logging, NaN checking, conservation, etc.
These are not model-specific but are frequently needed across simulations.

# Keyword Arguments
- `progress_logging = false`: Enable progress reporting
- `check_nan_every = 0`: Check for NaNs every N steps (0 = disabled)
- `check_conservation = false`: Enable conservation checking
- `checkpoint_frequency = "Inf"`: Frequency for saving state to disk
- `external_forcing_column = false`: Enable external forcing for single column
"""
function common_callbacks(
    dt, output_dir, start_date, t_start, t_end, comms_ctx;
    progress_logging = false,
    check_nan_every = 0,
    check_conservation = false,
    checkpoint_frequency = "Inf",
    external_forcing_column = false,
)

    callbacks = ()
    # TODO: Unify this with `get_callbacks.jl`

    if progress_logging
        @info "Progress logging enabled"
        walltime_info = WallTimeInfo()
        tot_steps = ceil(Int, (t_end - t_start) / dt)
        five_percent_steps = ceil(Int, 0.05 * tot_steps)
        schedule = CappedGeometricSeriesSchedule(five_percent_steps)
        cond = (u, t, integrator) -> schedule(integrator)
        affect! = (integrator) -> report_walltime(walltime_info, integrator)
        callbacks = (callbacks..., SciMLBase.DiscreteCallback(cond, affect!))
    end

    if check_nan_every > 0
        @info "Checking NaNs in the state every $(check_nan_every) steps"
        callbacks = (
            callbacks...,
            call_every_n_steps((integrator) -> check_nans(integrator), check_nan_every),
        )
    end
    callbacks = (
        callbacks...,
        call_every_n_steps(
            terminate!;
            skip_first = true,
            condition = (u, t, integrator) ->
                maybe_graceful_exit(output_dir, integrator),
        ),
    )

    dt_save_state_to_disk_dates =
        checkpoint_frequency_from_parsed_args(checkpoint_frequency)
    if dt_save_state_to_disk_dates != Inf
        schedule = CAD.EveryCalendarDtSchedule(
            dt_save_state_to_disk_dates;
            reference_date = start_date,
            date_last = t_start isa ITime ?
                        ClimaUtilities.TimeManager.date(t_start) :
                        start_date + Dates.Second(t_start),
        )
        cond = (u, t, integrator) -> schedule(integrator)
        affect! = (integrator) -> save_state_to_disk_func(integrator, output_dir)
        callbacks = (callbacks..., SciMLBase.DiscreteCallback(cond, affect!))
    end

    # Garbage collection, only enable for distributed simulations
    if is_distributed(comms_ctx)
        callbacks = (
            callbacks...,
            call_every_n_steps(
                gc_func,
                parse(Int, get(ENV, "CLIMAATMOS_GC_NSTEPS", "1000")),
                skip_first = true,
            ),
        )
    end

    if check_conservation
        callbacks = (
            callbacks...,
            call_every_n_steps(flux_accumulation!;
                skip_first = true, call_at_end = true,
            ),
        )
    end

    if external_forcing_column
        callbacks = (
            callbacks...,
            call_every_n_steps(
                external_driven_single_column!; call_at_end = true,
            ),
        )
    end

    return callbacks
end
