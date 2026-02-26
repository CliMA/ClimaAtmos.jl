import ClimaCore: Grids
import ClimaUtilities.TimeManager: ITime
import ClimaAtmos.Diagnostics as CAD
import .InitialConditions as ICs

struct AtmosSimulation{TT, S1 <: AbstractString, S2 <: AbstractString, OW, OD}
    job_id::S1
    output_dir::S2
    start_date::DateTime
    t_end::TT
    output_writers::OW
    integrator::OD
end

ClimaComms.context(sim::AtmosSimulation) =
    ClimaComms.context(sim.integrator.u.c)
ClimaComms.device(sim::AtmosSimulation) = ClimaComms.device(sim.integrator.u.c)


function setup_diagnostics_and_writers(
    use_default_diagnostics,
    diagnostics,
    model,
    Y,
    p,
    dt,
    t_start,
    t_end,
    start_date,
    output_dir,
)
    all_diagnostics = []
    writers = ()

    # Helper function to create default NetCDF writer
    function default_nc_writer(; netcdf_interpolation_method = "bilinear")
        parsed_args = Dict("netcdf_interpolation_method" => netcdf_interpolation_method)
        maybe_horizontal_method = netcdf_horizontal_method_kw(parsed_args) # `maybe` for version compatibility
        return CAD.NetCDFWriter(
            axes(Y.c),
            output_dir,
            num_points = ClimaDiagnostics.Writers.default_num_points(axes(Y.c));
            z_sampling_method = CAD.LevelsMethod(),  # TODO: Could make this configurable
            sync_schedule = CAD.EveryStepSchedule(),
            init_time = t_start,
            start_date,
            maybe_horizontal_method...,
        )
    end

    # Add default diagnostics if enabled
    if use_default_diagnostics
        sim_duration = t_end isa ITime ? t_end - dt : t_end
        netcdf_writer = default_nc_writer()
        default_diag_list = CAD.default_diagnostics(
            model,
            sim_duration,
            start_date,
            t_start;
            output_writer = netcdf_writer,
            topography = has_topography(axes(Y.c)),
        )
        append!(all_diagnostics, default_diag_list)
        @info "Added $(length(default_diag_list)) default ClimaAtmos diagnostics"
        # Create default writers tuple (matches get_diagnostics pattern)
        writers = (
            CAD.DictWriter(),
            CAD.HDF5Writer(output_dir),
            netcdf_writer,
        )
    end

    # Add user-provided diagnostics
    if !isempty(diagnostics)
        if diagnostics isa AbstractVector &&
           all(d -> d isa CAD.ScheduledDiagnostic, diagnostics)
            # User provided ScheduledDiagnostic objects directly
            append!(all_diagnostics, diagnostics)
            @info "Added $(length(diagnostics)) user-provided ScheduledDiagnostic objects"

            # Create default writers if not already created (from default_diagnostics)
            if isempty(writers)
                writers = (
                    CAD.DictWriter(),
                    CAD.HDF5Writer(output_dir),
                    default_nc_writer(),
                )
            end
        else
            # YAML-style diagnostics: get_diagnostics will return its own writers
            diag_config = Dict(
                "diagnostics" => diagnostics,
                "netcdf_interpolation_num_points" => nothing,
                "netcdf_output_at_levels" => false,
                "netcdf_interpolation_method" => "bilinear",
                "output_default_diagnostics" => false,
            )
            user_scheduled_diagnostics, user_writers, _ = get_diagnostics(
                diag_config,
                model,
                Y, p, dt,
                t_start, start_date,
                output_dir,
            )
            append!(all_diagnostics, user_scheduled_diagnostics)
            # Use writers from get_diagnostics (they match the user's config)
            writers = user_writers
            @info "Added $(length(user_scheduled_diagnostics)) user-provided YAML-style diagnostics"
        end
    end

    # Extract accumulation periods from all diagnostics
    periods_reductions = extract_diagnostic_periods(all_diagnostics)
    if !isempty(periods_reductions)
        periods_str = join(promote_period.(periods_reductions), ", ")
        @info "Saving accumulated diagnostics to disk with frequency: $(periods_str)"
    end

    return all_diagnostics, writers, periods_reductions
end

"""
    AtmosSimulation(config::AtmosConfig)
    AtmosSimulation(; kwargs...)

Construct a simulation.
"""
function AtmosSimulation(config::AtmosConfig)
    return get_simulation(config)
end

function AtmosSimulation{FT}(;
    model = AtmosModel(),
    params::Parameters.ClimaAtmosParameters = ClimaAtmosParameters(FT),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(),
    grid::Grids.AbstractGrid = SphereGrid(FT; radius = CAP.planet_radius(params), context),
    initial_condition::ICs.InitialCondition = InitialConditions.DecayingProfile(),
    dt = 600,
    start_date = DateTime(2010, 1, 1),
    t_start = 0,
    t_end = 86400 * 10,  # 10 days
    ode_config = CTS.IMEXAlgorithm(
        CTS.ARS343(),
        CTS.NewtonsMethod(;
            max_iters = 1,
            update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
        ),
    ),
    surface_setup = SurfaceConditions.DefaultExchangeCoefficients(),
    itime = false,
    job_id = "atmos_sim",
    output_dir = nothing,
    output_dir_style = "activelink",  # TODO: Should this be an actual type?
    restart_file = nothing,
    detect_restart_file = false,
    tracers = [], # TODO: set these from the model
    # Callbacks
    default_callbacks = true,   # Enable common simulation callbacks  
    callbacks = (),             # User-provided additional callbacks
    callback_kwargs = (),       # Kwargs for default_callbacks
    # Diagnostics
    default_diagnostics = true, # Enable standard ClimaAtmos diagnostics
    diagnostics = (),           # User-provided diagnostics (YAML string format or ScheduledDiagnostics )
    # Numerics
    use_dense_jacobian = false,
    use_auto_jacobian = false,
    approximate_linear_solve_iters = 1,
    auto_jacobian_padding_bands = 0,
    debug_jacobian = false,
    # Misc 
    checkpoint_frequency = Inf,
    log_to_file = false,
) where {FT}
    # Set up output directory and restart file detection
    output_dir, restart_file = setup_output_dir(
        job_id, output_dir, output_dir_style,
        detect_restart_file, restart_file, context,
    )

    if !isnothing(restart_file)
        # Handle restart: validates t_start, loads state, logs info, extracts spaces
        (Y, t_start, spaces) = handle_restart(
            restart_file, t_start, start_date, model, context, itime, FT,
        )
        # t_start is already converted from restart file, but we still need to convert dt and t_end
        if itime
            # convert time string to seconds
            to_seconds(t) = t isa AbstractString ? time_to_seconds(t) : Float64(t)
            dt_seconds = to_seconds(dt)
            t_end_seconds = to_seconds(t_end)
            dt = ITime(dt_seconds)
            t_end = ITime(t_end_seconds, epoch = start_date)
            # Promote with t_start to ensure all have compatible types
            (dt, t_start, t_end, _) = promote(dt, t_start, t_end, ITime(0))
        else
            # convert time string to FT
            to_ft(t) = t isa AbstractString ? FT(time_to_seconds(t)) : FT(t)
            dt = to_ft(dt)
            t_end = to_ft(t_end)
        end
    else
        dt, t_start, t_end = convert_time_args(dt, t_start, t_end, itime, start_date, FT)
        spaces = get_spaces(grid)
        Y = ICs.atmos_state(
            initial_condition(params), model,
            spaces.center_space,
            spaces.face_space,
        )
        InitialConditions.overwrite_initial_conditions!(
            initial_condition, Y, params.thermodynamics_params,
        )
    end

    # TODO: Add steady state velocity
    steady_state_velocity = nothing
    time_varying_trace_gas_names = ()
    p = build_cache(
        Y, model, params, surface_setup, dt, start_date, tracers,
        time_varying_trace_gas_names, steady_state_velocity,
        nothing,  # vwb_species - not available in this context
    )

    # Combine all callbacks
    discrete_callbacks = if default_callbacks
        checkpoint_frequency = parse_checkpoint_frequency(checkpoint_frequency)
        (
            default_model_callbacks(
                model;
                start_date, dt, t_start, t_end, output_dir, checkpoint_frequency,
                callback_kwargs...,
            )...,
            common_callbacks(
                model, dt, output_dir, start_date, t_start, t_end, context,
                checkpoint_frequency,
            )...)
    else
        callbacks
    end
    continuous_callbacks = ()
    callback_set = SciMLBase.CallbackSet(continuous_callbacks, discrete_callbacks)

    integrator_args, integrator_kwargs = args_integrator(
        Y, p, (t_start, t_end), ode_config,
        callback_set,
        use_dense_jacobian, use_auto_jacobian, auto_jacobian_padding_bands,
        approximate_linear_solve_iters, debug_jacobian,
        nothing,
        dt,
    )

    integrator = SciMLBase.init(integrator_args...; integrator_kwargs...)

    # Set up diagnostics and writers
    all_diagnostics, writers, periods_reductions = setup_diagnostics_and_writers(
        default_diagnostics, diagnostics, model,
        Y, p, dt,
        t_start, t_end, start_date,
        output_dir,
    )

    # Validate checkpoint-diagnostics consistency
    validate_checkpoint_diagnostics_consistency(
        checkpoint_frequency, periods_reductions,
    )

    # Wrap integrator with diagnostics if any diagnostics are present
    if !isempty(all_diagnostics)
        integrator = ClimaDiagnostics.IntegratorWithDiagnostics(
            integrator,
            all_diagnostics,
        )
        @info "Initialized $(length(all_diagnostics)) total diagnostics"
    end

    reset_graceful_exit(output_dir)

    if log_to_file
        logger = ClimaComms.FileLogger(context, output_dir)
        Logging.global_logger(logger)
    end

    return AtmosSimulation(
        job_id, output_dir, start_date, t_end, writers, integrator,
    )
end
