import ClimaCore
import ClimaCore: Grids
import ClimaUtilities.TimeManager: ITime
import ClimaAtmos.Diagnostics as CAD
import .Setups

"""
    AtmosSimulation

A configured atmospheric simulation: an initialized time-stepping integrator together
with the output bookkeeping needed to run it and write its diagnostics.

Build one with the keyword constructor `AtmosSimulation{FT}(; ...)` (or
`AtmosSimulation(; ...)` for `Float32`), or from a configuration with
`AtmosSimulation(config)`. Run it with `solve_atmos!`.

# Fields

  - `job_id`: Run identifier, also used to name the output directory.
  - `output_dir`: Directory that receives diagnostics, checkpoints, and logs.
  - `start_date`: Calendar date corresponding to the simulation start.
  - `t_end`: End time of the simulation [s].
  - `output_writers`: Diagnostic writers, closed by `solve_atmos!` when the run
    finishes.
  - `integrator`: The ClimaTimeSteppers integrator holding the state, cache, and callbacks.
"""
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

"""
    setup_diagnostics_and_writers(diagnostics_config, model, Y, p, dt, t_start, t_end,
                                  start_date, output_dir; verbose = false)

Build the scheduled diagnostics and the writers they output to.

The default diagnostics of `model` are added when `diagnostics_config.default` is set,
and the entries of `diagnostics_config.additional` are added as well: those are either
prebuilt `ScheduledDiagnostic`s or dictionary specifications, the latter translated by
`scheduled_diagnostics_from_specs`. A NetCDF, an HDF5, and an in-memory dictionary
writer are always created; a second NetCDF writer on pressure levels is added when a
specification requests pressure coordinates.

# Returns

`(all_diagnostics, writers, periods_reductions)`: the scheduled diagnostics, the writers
they bind to, and the accumulation periods used by
`validate_checkpoint_diagnostics_consistency`.
"""
function setup_diagnostics_and_writers(
    diagnostics_config::CAD.DiagnosticsConfig,
    model,
    Y,
    p,
    dt,
    t_start,
    t_end,
    start_date,
    output_dir;
    verbose = false,
)
    (;
        default,
        additional,
        interpolation_num_points,
        output_at_levels,
        debug_tendency,
    ) = diagnostics_config

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
        sim_duration = t_end - dt
        default_diag_list = CAD.default_diagnostics(
            model,
            sim_duration,
            start_date,
            t_start;
            output_writer = netcdf_writer,
            topography = has_topography(axes(Y.c)),
        )
        append!(all_diagnostics, default_diag_list)
        verbose &&
            @info "Added $(length(default_diag_list)) default ClimaAtmos diagnostics"
    end

    # Add debug tendency diagnostics if enabled
    if debug_tendency
        sim_duration = t_end - dt
        tendency_diag_list = CAD.tendency_debug_default_diagnostics(
            netcdf_writer,
            sim_duration,
            start_date,
            t_start,
        )
        append!(all_diagnostics, tendency_diag_list)
        verbose &&
            @info "Added $(length(tendency_diag_list)) debug tendency diagnostics"
    end

    # Add user-provided diagnostics
    if !isempty(additional)
        normalized = map(CAD.normalize_diag_entry, additional)
        prebuilt = filter(d -> d isa CAD.ScheduledDiagnostic, normalized)
        dict_specs = filter(d -> d isa AbstractDict, normalized)

        if !isempty(prebuilt)
            append!(all_diagnostics, prebuilt)
            verbose &&
                @info "Added $(length(prebuilt)) user-provided ScheduledDiagnostic objects"
        end

        if !isempty(dict_specs)
            # If any dict spec requests pressure coordinates, build the pressure
            # writer here and extend the shared writers tuple.
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
            verbose &&
                @info "Added $(length(user_scheduled_diagnostics)) user-provided YAML-style diagnostics"
        end
    end

    # Extract accumulation periods from all diagnostics
    periods_reductions = CAD.extract_diagnostic_periods(all_diagnostics)
    if !isempty(periods_reductions)
        periods_str = join(promote_period.(periods_reductions), ", ")
        verbose &&
            @info "Saving accumulated diagnostics to disk with frequency: $(periods_str)"
    end

    return all_diagnostics, writers, periods_reductions
end

"""
    convert_time_args(dt, t_start, t_end, start_date)

Convert `dt`, `t_start`, and `t_end` to `ITime`s of a common type, with `start_date` as
the epoch of the two times.

Each input may be a number of seconds or a time string such as `"1hours"` (see
`time_to_seconds`).
"""
function convert_time_args(dt, t_start, t_end, start_date)
    dt = ITime(time_to_seconds(dt))
    t_start = ITime(time_to_seconds(t_start), epoch = start_date)
    t_end = ITime(time_to_seconds(t_end), epoch = start_date)
    # ITime(0) is added for backward compatibility (since t_start used to always be 0)
    (dt, t_start, t_end, _) = promote(dt, t_start, t_end, ITime(0))
    return (dt, t_start, t_end)
end

"""
    AtmosSimulation(config::AtmosConfig)

Construct a simulation from a configuration, with the float type taken from `config`.

Equivalent to `get_simulation(config)`, which also writes the parameter manifest and
config snapshot into the output directory.
"""
AtmosSimulation(config::AtmosConfig) = get_simulation(config)

"""
    AtmosSimulation(; kwargs...)

Construct an atmospheric simulation with the default float type `Float32`.

Equivalent to `AtmosSimulation{Float32}(; kwargs...)`.
"""
AtmosSimulation(; kwargs...) = AtmosSimulation{Float32}(; kwargs...)

"""
    AtmosSimulation{FT}(; kwargs...) where {FT}

Construct an atmospheric simulation with float type `FT`.

Builds (or restarts) the state, the cache, the callbacks, the diagnostics, and the
time-stepping integrator, and resolves the output directory. This is the primary
entry point for simulations written as scripts; configuration-driven runs go through
`get_simulation` instead.

# Keyword Arguments

  - `model = AtmosModel()`: Physics and parameterization configuration.
  - `params = ClimaAtmosParameters(FT; microphysics_model = model.microphysics_model)`:
    Physical parameters. Built from `model` by default, so only the parameter sets
    the model needs are loaded and the microphysics process options set on the
    model take effect.
  - `context = ClimaComms.context()`: Communications context (device and MPI).
  - `grid = SphereGrid(FT; radius = CAP.planet_radius(params), context)`: Computational
    grid. Use [`ColumnGrid`](@ref), [`BoxGrid`](@ref), [`PlaneGrid`](@ref), or
    [`SphereGrid`](@ref).
  - `setup = Setups.DecayingProfile(; perturb = true, params)`: Setup defining the initial
    state, and, for single-column cases, the forcings. See [Setups](@ref "Setups").
  - `dt = 600`: Timestep [s], or a string such as `"10mins"`.
  - `start_date = DateTime(2010, 1, 1)`: Calendar date of the simulation start.
  - `t_start = 0`: Start time [s]. Ignored, with a warning, when restarting.
  - `t_end = 86400 * 10`: End time [s], 10 days by default.
  - `ode_config`: Time-stepping algorithm. Defaults to `IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 1, update_j = UpdateEvery(NewNewtonIteration)))`.
  - `steady_state_velocity = nothing`: Analytic steady-state velocity used by diagnostics,
    either a precomputed field or a callable `(Y, params) -> velocity` evaluated once `Y`
    exists.
  - `job_id = "atmos_sim"`: Run identifier, used in output directory naming.
  - `output_dir = nothing`: Output directory. Defaults to `output/<job_id>`, or `<job_id>`
    when the `CI` environment variable is set.
  - `output_dir_style = "activelink"`: How the output directory is managed;
    `"activelink"` keeps numbered directories with a symlink to the active one,
    `"removepreexisting"` deletes previous output.
  - `restart_file = nothing`: Restart file to resume from.
  - `detect_restart_file = false`: Pick up the most recent restart file in the output
    directory structure; only available with `output_dir_style = "activelink"`.
  - `aerosol_names = []`: Prescribed aerosol species to read from file.
  - `time_varying_trace_gases = ()`: Trace gases read from a time-varying file.
  - `vertical_water_borrowing_species = nothing`: Species the vertical water borrowing
    constraint may draw from.
  - `default_callbacks = true`: Add the default model and common callbacks. When `false`,
    only `callbacks` is used.
  - `callbacks = ()`: User-provided callbacks, used only when `default_callbacks` is
    `false`.
  - `callback_kwargs = ()`: Extra keyword arguments forwarded to the default callbacks.
  - `diagnostics = DiagnosticsConfig()`: Which diagnostics to produce and how to write
    them. See [`DiagnosticsConfig`](@ref).
  - `jacobian = ManualSparseJacobian(; approximate_solve_iters = 1)`: Jacobian algorithm
    for the implicit solve. Use [`ManualSparseJacobian`](@ref),
    [`AutoSparseJacobian`](@ref), or [`AutoDenseJacobian`](@ref).
  - `debug_jacobian = false`: Print Jacobian diagnostics while solving.
  - `update_cache_every = "stage"`: When the cache is refreshed, `"stage"` or `"step"`.
  - `update_constrain_state_every = "step"`: When state constraints are applied,
    `"stage"`, `"step"`, or `"dss"`.
  - `checkpoint_frequency = Inf`: How often to write restart checkpoints; a number of
    seconds, a time string, or `"<N>months"`. `Inf` disables checkpointing.
  - `log_to_file = false`: Send log output to a file in the output directory.
  - `verbose = false`: Log progress while building the simulation (root process only).

# Returns

An [`AtmosSimulation`](@ref), ready to be passed to `solve_atmos!`.

# Examples

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
    params::Parameters.ClimaAtmosParameters = ClimaAtmosParameters(
        FT; model.microphysics_model,
    ),
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
    steady_state_velocity = nothing, # Predicted steady-state velocity for diagnostics
    job_id = "atmos_sim",
    output_dir = nothing,
    output_dir_style = "activelink",  # TODO: Should this be an actual type?
    restart_file = nothing,
    detect_restart_file = false,
    aerosol_names = [], # TODO: set from the model
    time_varying_trace_gases = (),
    vertical_water_borrowing_species = nothing,
    # Callbacks
    default_callbacks = true,   # Enable common simulation callbacks
    callbacks = (),             # User-provided additional callbacks
    callback_kwargs = (),       # Kwargs for default_callbacks
    # Diagnostics
    diagnostics = CAD.DiagnosticsConfig(),
    # Numerics
    jacobian::JacobianAlgorithm = ManualSparseJacobian(approximate_solve_iters = 1),
    debug_jacobian = false,
    update_cache_every = "stage",
    update_constrain_state_every = "step",
    # Misc
    checkpoint_frequency = Inf,
    log_to_file = false,
    verbose = false,
) where {FT}
    # Log only on root process
    verbose = ClimaComms.iamroot(context) && verbose

    # Set up output directory and restart file detection
    output_dir, restart_file = setup_output_dir(
        job_id, output_dir, output_dir_style,
        detect_restart_file, restart_file, context,
    )

    if !isnothing(restart_file)
        # Handle restart: validates t_start, loads state, logs info, extracts spaces
        (Y, t_start, spaces) = @timed_log verbose "Loaded restart file" handle_restart(
            restart_file, t_start, start_date, model, context; verbose,
        )
        # t_start is already converted from restart file, but we still need to convert dt and t_end
        dt = ITime(time_to_seconds(dt))
        t_end = ITime(time_to_seconds(t_end), epoch = start_date)
        # Promote with t_start to ensure all have compatible types
        (dt, t_start, t_end, _) = promote(dt, t_start, t_end, ITime(0))
    else
        dt, t_start, t_end = convert_time_args(dt, t_start, t_end, start_date)
        spaces = get_spaces(grid)
        @timed_log verbose "Initialized state" begin
            Y = Setups.initial_state(
                setup, params, model,
                spaces.center_space,
                spaces.face_space,
            )
            Setups.overwrite_initial_state!(
                setup, Y, params.thermodynamics_params,
            )
        end
    end

    # Resolve steady_state_velocity: accept nothing, a precomputed velocity field,
    # or a callable `(Y, params) -> velocity` that needs Y to be built first.
    resolved_steady_state_velocity =
        steady_state_velocity isa Function ? steady_state_velocity(Y, params) :
        steady_state_velocity

    p = @timed_log verbose "Built cache" build_cache(
        Y, model, params, dt, start_date, aerosol_names,
        time_varying_trace_gases, resolved_steady_state_velocity,
        vertical_water_borrowing_species,
    )

    # Combine all callbacks
    discrete_callbacks = @timed_log verbose "Assembled callbacks" if default_callbacks
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
    callback_set = CTS.CallbackSet(discrete_callbacks...)

    integrator_args, integrator_kwargs = args_integrator(
        Y, p, (t_start, t_end), ode_config,
        callback_set,
        jacobian, debug_jacobian,
        model.prescribed_flow,
        dt,
        update_cache_every,
        update_constrain_state_every;
        verbose,
    )

    integrator = @timed_log verbose "Initialized integrator" CTS.init(
        integrator_args...;
        integrator_kwargs...,
    )

    all_diagnostics, writers, periods_reductions =
        @timed_log verbose "Set up diagnostics" setup_diagnostics_and_writers(
            diagnostics, model,
            Y, p, dt,
            t_start, t_end, start_date,
            output_dir;
            verbose,
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
        verbose && @info "Initialized $(length(all_diagnostics)) total diagnostics"
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
