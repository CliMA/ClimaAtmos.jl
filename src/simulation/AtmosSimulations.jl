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

const default_FT = Float32

#= TODOs
- [x] Document discrepancies between get_simulation and AtmosSimulation constructors
- Unimportant differences:
  - get_simulation uses config-based approach with parsed_args, AtmosSimulation uses direct parameters
  - get_simulation handles output directory generation with styles (activelink, removepreexisting)
  - get_simulation has parameter logging (TOML, YAML) and file output
  - get_simulation has checkpoint frequency validation against diagnostics
  - get_simulation supports distributed computing contexts
  - get_simulation has graceful exit handling
  - get_simulation uses ode_configuration() vs direct ode_algo parameter
  - get_simulation supports tracers from parsed_args vs direct tracers parameter
Important differences (future work):
  - get_simulation has extensive logging and timing (@timed_str, @info)
  - get_simulation supports restart file detection and auto-detection
  - get_simulation supports ITime vs regular time handling
  - get_simulation has more complex steady_state_velocity calculation
  - get_simulation has more sophisticated callback handling (continuous vs discrete)
  - get_simulation uses enable_diagnostics flag vs direct diagnostics parameter
- [ ] Improve creation of the ode configuration
- [ ] Handle diagnostics and callbacks in separate functions, keep current behavior, add default diagnostics 
- [ ] Ensure FT is propagated consistently with default structs (params, domain)
- [x] Deal with domain not having topography
- [ ] Ensure output writers are set up correctly
- [ ] Ensure same defaults between constructors
- [ ] See what else can be unified between the two constructors
- [ ] Add unit tests for the AtmosSimulation constructor, with informative error messages
- [ ] Add documentation
=#

function AtmosSimulation{FT}(;
    model::AtmosModel = AtmosModel(),
    domain::AtmosDomain = SphereDomain{FT}(),
    params::Parameters.ClimaAtmosParameters = ClimaAtmosParameters(FT),
    initial_condition::ICs.InitialCondition = InitialConditions.DecayingProfile(),
    comms_ctx::ClimaComms.AbstractCommsContext = ClimaComms.context(),
    dt = 600,
    t_start = 0,
    t_end = 24 * 10 * 60 * 60,  # 10 days
    ode_algo = CTS.ARS343(),
    surface_setup = SurfaceConditions.DefaultExchangeCoefficients(),
    job_id = "atmos_sim",
    output_dir = "output",
    restart_file = nothing,
    start_date = DateTime(2010, 1, 1),
    tracers = [],
    callbacks = (),
    diagnostics = (),
    discrete_hydrostatic_balance = false,
    itime = false,
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
    callback_set = SciMLBase.CallbackSet(callbacks...)
    integrator_args, integrator_kwargs = args_integrator(
        Y, p, (t_start, t_end), ode_config,
        callback_set,
        use_dense_jacobian, use_auto_jacobian, 
        approximate_linear_solve_iters, debug_jacobian,
    )
    integrator = SciMLBase.init(integrator_args...; integrator_kwargs...)

    # Initialize diagnostics
    # TODO: Move this to a separate function, add the same checks and behavior 
    # as in get_diagnostics
    if !isempty(diagnostics)
        scheduled_diagnostics, writers, _ = get_diagnostics(
            diagnostics,
            model,
            Y,
            p,
            t_start,
            start_date,
            output_dir,
        )
        integrator = ClimaDiagnostics.IntegratorWithDiagnostics(
            integrator,
            scheduled_diagnostics,
        )
    else
        writers = ()
    end

    reset_graceful_exit(output_dir)

    return AtmosSimulation(
        job_id,
        output_dir,
        start_date,
        t_end,
        writers,
        integrator,
    )
end
