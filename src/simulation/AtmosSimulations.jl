import ClimaCore
import ClimaCore: Grids
import ClimaUtilities.TimeManager: ITime
import ClimaAtmos.Diagnostics as CAD
import .Setups

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
    diagnostics_config::CAD.DiagnosticsConfig,
    model,
    Y,
    p,
    dt,
    t_start,
    t_end,
    start_date,
    output_dir,
)
    (; default, additional, interpolation_num_points, output_at_levels) =
        diagnostics_config

    all_diagnostics = []

    num_points = if isnothing(interpolation_num_points)
        ClimaDiagnostics.Writers.default_num_points(axes(Y.c))
    else
        tuple(interpolation_num_points...)
    end
    z_sampling_method =
        output_at_levels ? CAD.LevelsMethod() : CAD.FakePressureLevelsMethod()
    horizontal_method = ClimaCore.Remapping.BilinearRemapping()

    # Build writers once. All paths below bind diagnostics to these instances.
    netcdf_writer = CAD.NetCDFWriter(
        axes(Y.c),
        output_dir;
        num_points,
        z_sampling_method,
        horizontal_method,
        sync_schedule = CAD.EveryStepSchedule(),
        init_time = t_start,
        start_date,
    )
    writers = (CAD.DictWriter(), CAD.HDF5Writer(output_dir), netcdf_writer)

    # Add default diagnostics if enabled
    if default
        sim_duration = t_end isa ITime ? t_end - dt : t_end
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
    end

    # Add user-provided diagnostics
    if !isempty(additional)
        normalized = map(CAD.normalize_diag_entry, additional)
        prebuilt = filter(d -> d isa CAD.ScheduledDiagnostic, normalized)
        dict_specs = filter(d -> d isa AbstractDict, normalized)

        if !isempty(prebuilt)
            append!(all_diagnostics, prebuilt)
            @info "Added $(length(prebuilt)) user-provided ScheduledDiagnostic objects"
        end

        if !isempty(dict_specs)
            # If any dict spec requests pressure coordinates, build the pressure
            # writer here and extend the shared writers tuple before delegating.
            if any(d -> get(d, "pressure_coordinates", false), dict_specs) &&
               length(writers) < 4
                pressure_z_sampling = ClimaDiagnostics.Writers.RealPressureLevelsMethod(
                    p.precomputed.ᶜp, t_start,
                )
                pressure_space =
                    ClimaDiagnostics.Writers.pressure_space(pressure_z_sampling)
                pressure_netcdf_writer = CAD.NetCDFWriter(
                    pressure_space,
                    output_dir;
                    num_points,
                    z_sampling_method = pressure_z_sampling,
                    horizontal_method,
                    sync_schedule = CAD.EveryStepSchedule(),
                    init_time = t_start,
                    start_date,
                )
                writers = (writers..., pressure_netcdf_writer)
            end

            user_scheduled_diagnostics = scheduled_diagnostics_from_specs(
                dict_specs, Y, t_start, start_date, writers,
            )
            append!(all_diagnostics, user_scheduled_diagnostics)
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

Construct a simulation from a YAML-based configuration.
Construct an atmospheric simulation with the default floating-point type `Float32`.
Equivalent to `AtmosSimulation{Float32}(; kwargs...)`.
"""
AtmosSimulation(config::AtmosConfig) = get_simulation(config)

"""
    AtmosSimulation(; kwargs...)

Construct an atmospheric simulation with the default floating-point type `Float32`.
Equivalent to `AtmosSimulation{Float32}(; kwargs...)`.
"""
AtmosSimulation(; kwargs...) = AtmosSimulation{Float32}(; kwargs...)

"""
    AtmosSimulation{FT}(; kwargs...) where {FT}

Construct an atmospheric simulation with floating-point type `FT` (default: Float32).

## Keyword Arguments

### Model and domain
- `model::AtmosModel = AtmosModel()`: Physics and parameterization configuration.
- `params::ClimaAtmosParameters = ClimaAtmosParameters(FT)`: Physical parameters.
- `grid::AbstractGrid = SphereGrid(FT; ...)`: Computational grid.
  Use [`ColumnGrid`](@ref), [`BoxGrid`](@ref), [`PlaneGrid`](@ref), or [`SphereGrid`](@ref).
- `setup = Setups.DecayingProfile(; perturb=true, params)`: Setup defining the
  initial state. See [Setups](@ref "Setups") for available options.
- `surface_setup = DefaultExchangeCoefficients()`: Surface exchange parameterization.

### Time
- `dt = 600`: Timestep in seconds.
- `t_start = 0`: Start time in seconds.
- `t_end = 864000`: End time in seconds (default: 10 days).
- `start_date = DateTime(2010, 1, 1)`: Calendar reference date.

### Output
- `job_id::String = "atmos_sim"`: Run identifier, used in output directory naming.
- `output_dir = nothing`: Output directory path. Auto-generated from `job_id` if `nothing`.
- `output_dir_style = "activelink"`: Output directory organization style.
- `checkpoint_frequency = Inf`: How often to save restart checkpoints (seconds).
- `log_to_file::Bool = false`: Write log output to a file in `output_dir`.

### Diagnostics
- `diagnostics::DiagnosticsConfig = DiagnosticsConfig()`: Specification of which
  diagnostics the simulation produces and how their NetCDF output is shaped.
  See [`DiagnosticsConfig`](@ref).

### Callbacks
- `default_callbacks::Bool = true`: Enable common simulation callbacks.
- `callbacks = ()`: Additional user-provided callbacks.
- `callback_kwargs = ()`: Extra keyword arguments forwarded to default callbacks.

### Restarts
- `restart_file = nothing`: Path to a restart file to resume from.
- `detect_restart_file::Bool = false`: Automatically detect the latest restart file in
  a structured output directory.

### Numerics
- `ode_config`: ODE solver algorithm. Default: `IMEXAlgorithm(ARS343(), NewtonsMethod(...))`.
- `jacobian::JacobianAlgorithm = ManualSparseJacobian(; approximate_solve_iters = 1)`:
  Jacobian algorithm for the implicit solve. Use [`ManualSparseJacobian`](@ref),
  [`AutoSparseJacobian`](@ref), or [`AutoDenseJacobian`](@ref).
- `debug_jacobian::Bool = false`: Enable Jacobian debugging output.
- `tracers = []`: Additional tracer species.

## Example

```julia
import ClimaAtmos as CA

# Minimal: 1-day global simulation with defaults
simulation = CA.AtmosSimulation{Float64}(; t_end = 86400)
CA.solve_atmos!(simulation)

# Single-column BOMEX case
simulation = CA.AtmosSimulation{Float64}(;
    grid = CA.ColumnGrid(Float64; z_elem = 60, z_max = 3000.0),
    setup = CA.Setups.Bomex(),
    dt = 5,
    t_end = 3600 * 6,
)
```
"""
function AtmosSimulation{FT}(;
    model = AtmosModel(),
    params::Parameters.ClimaAtmosParameters = ClimaAtmosParameters(FT),
    context::ClimaComms.AbstractCommsContext = ClimaComms.context(),
    grid::Grids.AbstractGrid = SphereGrid(FT; radius = CAP.planet_radius(params), context),
    setup = Setups.DecayingProfile(; perturb = true, params),
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
    steady_state_velocity = nothing, # Predicted steady-state velocity for diagnostics
    job_id = "atmos_sim",
    output_dir = nothing,
    output_dir_style = "activelink",  # TODO: Should this be an actual type?
    restart_file = nothing,
    detect_restart_file = false,
    aerosol_names = [], # TODO: set from the model
    time_varying_trace_gases = (),
    # Callbacks
    default_callbacks = true,   # Enable common simulation callbacks  
    callbacks = (),             # User-provided additional callbacks
    callback_kwargs = (),       # Kwargs for default_callbacks
    # Diagnostics
    diagnostics = CAD.DiagnosticsConfig(),
    # Numerics
    jacobian::JacobianAlgorithm = ManualSparseJacobian(approximate_solve_iters = 1),
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
            restart_file, t_start, start_date, model, context,
        )
        # t_start is already converted from restart file, but we still need to convert dt and t_end
        to_seconds(t) = t isa AbstractString ? time_to_seconds(t) : Float64(t)
        dt = ITime(to_seconds(dt))
        t_end = ITime(to_seconds(t_end), epoch = start_date)
        # Promote with t_start to ensure all have compatible types
        (dt, t_start, t_end, _) = promote(dt, t_start, t_end, ITime(0))
    else
        dt, t_start, t_end = convert_time_args(dt, t_start, t_end, start_date)
        spaces = get_spaces(grid)
        Y = Setups.initial_state(
            setup, params, model,
            spaces.center_space,
            spaces.face_space,
        )
        Setups.overwrite_initial_state!(
            setup, Y, params.thermodynamics_params,
        )
    end

    p = build_cache(
        Y, model, params, surface_setup, dt, start_date, aerosol_names,
        time_varying_trace_gases, steady_state_velocity,
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
                checkpoint_frequency;
                callback_kwargs...,
            )...)
    else
        callbacks
    end
    continuous_callbacks = ()
    callback_set = CTS.CallbackSet(continuous_callbacks, discrete_callbacks)

    integrator_args, integrator_kwargs = args_integrator(
        Y, p, (t_start, t_end), ode_config,
        callback_set,
        jacobian, debug_jacobian,
        nothing,
        dt,
    )

    integrator = CTS.init(integrator_args...; integrator_kwargs...)

    all_diagnostics, writers, periods_reductions = setup_diagnostics_and_writers(
        diagnostics, model,
        Y, p, dt,
        t_start, t_end, start_date,
        output_dir,
    )

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
