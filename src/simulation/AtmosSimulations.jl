include("steady_state_velocity.jl")

struct AtmosSimulation{TT, S1 <: AbstractString, S2 <: AbstractString, OW, OD}
    job_id::S1
    output_dir::S2
    start_date::DateTime
    t_end::TT
    output_writers::OW
    integrator::OD
end

Base.summary(::AtmosSimulation) = "AtmosSimulation"

ClimaComms.context(sim::AtmosSimulation) =
    ClimaComms.context(sim.integrator.u.c)
ClimaComms.device(sim::AtmosSimulation) = ClimaComms.device(sim.integrator.u.c)

function Base.show(io::IO, sim::AtmosSimulation)
    device_type = nameof(typeof(ClimaComms.device(sim)))
    return print(
        io,
        "Simulation $(sim.job_id)\n",
        "├── Running on: $(device_type)\n",
        "├── Output folder: $(sim.output_dir)\n",
        "├── Start date: $(sim.start_date)\n",
        "├── Current time: $(sim.integrator.t) seconds\n",
        "└── Stop time: $(sim.t_end) seconds",
    )
end

"""
    AtmosSimulation(config::AtmosConfig)
    AtmosSimulation(; kwargs...)

Construct a simulation.
"""
function AtmosSimulation(config::AtmosConfig)
    return get_simulation(config)
end

#= TODOs
Unimportant differences compared to get_simulation:
- get_simulation uses config-based approach with parsed_args, AtmosSimulation uses direct parameters
- get_simulation handles output directory generation with styles (activelink, removepreexisting)
- get_simulation has parameter logging (TOML, YAML) and file output
- get_simulation has checkpoint frequency validation against diagnostics
- get_simulation supports tracers from parsed_args vs direct tracers parameter
future work:
- [ ] Improve creation of the ode configuration
- [ ] See what else can be unified between the two constructors
- [ ] Add unit tests for the AtmosSimulation constructor
=#

function AtmosSimulation{FT}(;
    model = AtmosModel(),
    domain::AtmosDomain = SphereDomain{FT}(),
    params::Parameters.ClimaAtmosParameters = ClimaAtmosParameters(FT),
    initial_condition::ICs.InitialCondition = InitialConditions.DecayingProfile(),
    comms_ctx::ClimaComms.AbstractCommsContext = ClimaComms.context(),
    dt = 600,
    start_date = DateTime(2010, 1, 1),
    t_start = 0,
    t_end = 24 * 10 * 60 * 60,  # 10 days
    ode_algo = CTS.ARS343(),
    surface_setup = SurfaceConditions.DefaultExchangeCoefficients(),
    itime = false,
    job_id = "atmos_sim",
    output_dir = nothing,
    output_dir_style = "activelink",
    restart_file = nothing,
    detect_restart_file = false,
    tracers = [], # TODO: set these from the model
    discrete_hydrostatic_balance = false,

    # Callback configuration
    model_callbacks = true,           # Enable automatic model-based callbacks
    default_callbacks = true,         # Enable common simulation callbacks  
    callbacks = (),                   # User-provided additional callbacks
    progress_logging = false,         # Enable progress reporting
    check_nan_every = 0,              # Check for NaNs every N steps (0 = disabled)  
    check_conservation = false,       # Enable conservation diagnostics
    checkpoint_frequency = "Inf",     # Frequency for saving state to disk
    dt_rad = "1h",                    # Radiation callback frequency
    dt_nogw = "3hours",               # Non-orographic gravity wave frequency
    dt_cloud_fraction = "3hours",     # Cloud fraction callback frequency
    call_cloud_diagnostics_per_stage = false,  # TODO: add netcdf_interpolation_num_points, netcdf_output_at_levels

    # Diagnostic configuration
    default_diagnostics = false,     # Enable standard ClimaAtmos diagnostics
    diagnostics = (),                # User-provided diagnostics (YAML format or ScheduledDiagnostic objects)

    
    # Numerics
    check_steady_state = false,
    use_dense_jacobian = false,
    use_auto_jacobian = false,
    approximate_linear_solve_iters = 1,
    debug_jacobian = false,
    update_jacobian_every = "solve",
    max_newton_iters_ode = 1,
    use_krylov_method = false,
    use_dynamic_krylov_rtol = false,
    eisenstat_walker_forcing_alpha = 2.0,
    krylov_rtol = 0.1,
    use_newton_rtol = false,
    newton_rtol = 1.0e-5,
) where {FT}
    # Set up output directory and restart file detection
    output_dir, restart_file = setup_output_dir(
        job_id, output_dir, output_dir_style,
        detect_restart_file, restart_file, comms_ctx,
    )

    isnothing(restart_file) || @info "Restarting simulation from file $restart_file"

    if !isnothing(restart_file)
        (Y, t_start) = get_state_restart(
            restart_file, start_date, hash(model), comms_ctx, itime,
        )
        spaces = get_spaces_restart(Y)
    else
        spaces = get_spaces(domain, params, comms_ctx)
        Y = ICs.atmos_state(
            initial_condition(params), model,
            spaces.center_space,
            spaces.face_space,
        )
        CA.InitialConditions.overwrite_initial_conditions!(
            initial_condition, Y, params.thermodynamics_params,
        )
    end

    steady_state_velocity = get_steady_state_velocity(
        domain, initial_condition, params, Y; check_steady_state,
    )

    p = build_cache(
        Y, model, params, surface_setup, dt, start_date, tracers,
        steady_state_velocity,
    )
    discrete_hydrostatic_balance &&
        set_discrete_hydrostatic_balanced_state!(Y, p)

    ode_name = nameof(typeof(ode_algo))
    ode_config = ode_configuration(FT, ode_name, update_jacobian_every,
        max_newton_iters_ode, use_krylov_method, use_dynamic_krylov_rtol,
        eisenstat_walker_forcing_alpha, krylov_rtol, use_newton_rtol, newton_rtol)

    model_callback_tuple = if model_callbacks
        default_model_callbacks(
            model;
            start_date, dt, t_start, t_end, output_dir = output_dir,
            dt_rad, dt_nogw, dt_cloud_fraction,
            call_cloud_diagnostics_per_stage,
            checkpoint_frequency,
        )
    else
        ()
    end

    default_callback_tuple = if default_callbacks
        common_callbacks(
            dt, output_dir, start_date, t_start, t_end, comms_ctx;
            progress_logging, check_nan_every, check_conservation, checkpoint_frequency,
            external_forcing_column = false,  # TODO: detect from model
        )
    else
        ()
    end

    # Combine all callbacks
    continuous_callbacks = ()
    discrete_callbacks = (model_callback_tuple..., default_callback_tuple..., callbacks...)
    callback_set = SciMLBase.CallbackSet(continuous_callbacks, discrete_callbacks)

    integrator_args, integrator_kwargs = args_integrator(
        Y, p, (t_start, t_end), ode_config,
        callback_set,
        use_dense_jacobian, use_auto_jacobian,
        approximate_linear_solve_iters, debug_jacobian,
    )
    integrator = SciMLBase.init(integrator_args...; integrator_kwargs...)

    # Initialize diagnostics with logical grouping
    all_diagnostics = []
    writers = ()

    if default_diagnostics
        # Calculate simulation duration
        sim_duration = t_end isa ITime ? t_end - dt : t_end

        default_diag_list = CAD.default_diagnostics(
            model,
            sim_duration,
            start_date;
            output_writer = CAD.NetCDFWriter(output_dir),
        )
        append!(all_diagnostics, default_diag_list)
        @info "Added $(length(default_diag_list)) default ClimaAtmos diagnostics"
    end

    # Add user-provided diagnostics
    if !isempty(diagnostics)
        if diagnostics isa AbstractVector &&
           all(d -> d isa CAD.ScheduledDiagnostic, diagnostics)

            append!(all_diagnostics, diagnostics)
            @info "Added $(length(diagnostics)) user-provided ScheduledDiagnostic objects"
        else
            user_scheduled_diagnostics, user_writers, _ = get_diagnostics(
                diagnostics isa Dict ? diagnostics : Dict("diagnostics" => diagnostics),
                model,
                Y, p, dt,
                t_start, start_date,
                output_dir,
            )
            append!(all_diagnostics, user_scheduled_diagnostics)
            writers = user_writers
            @info "Added $(length(user_scheduled_diagnostics)) user-provided YAML-style diagnostics"
        end
    end

    # Set up diagnostics integration if any diagnostics are present
    if !isempty(all_diagnostics)
        integrator = ClimaDiagnostics.IntegratorWithDiagnostics(
            integrator,
            all_diagnostics,
        )
        if isempty(writers)
            dict_writer = CAD.DictWriter()
            hdf5_writer = CAD.HDF5Writer(output_dir)
            netcdf_writer = CAD.NetCDFWriter(output_dir)
            writers = (dict_writer, hdf5_writer, netcdf_writer)
        end
        @info "Initialized $(length(all_diagnostics)) total diagnostics"
    end

    reset_graceful_exit(output_dir)

    return AtmosSimulation(
        job_id, output_dir, start_date, t_end, writers, integrator,
    )
end
