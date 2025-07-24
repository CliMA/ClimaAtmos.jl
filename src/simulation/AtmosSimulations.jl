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
- [ ] Document discrepancies between get_simulation and AtmosSimulation constructors
- [ ] Handle diagnostics and callbacks in separate functions, keep current behavior
- [ ] Ensure FT is propagated consistently with default structs (params, domain)
- [ ] Deal with domain not having topography
- [ ] Ensure output writers are set up correctly
- [ ] Ensure same defaults between constructors
- [ ] See what else can be unified between the two constructors
- [ ] Add unit tests for the constructors, with informative error messages
- [ ] Add documentation
=#

function AtmosSimulation(;
    model::AtmosModel,
    domain::AbstractDomain = SphereDomain{default_FT}(),
    params = ClimaAtmosParameters(default_FT),
    initial_condition = DecayingProfile(params),
    comms_ctx = ClimaComms.SingletonCommsContext(),
    dt = 600,
    t_start = 0,
    t_end = 24 * 10 * 60 * 60,  # 10 days
    ode_algo = CTS.ARS343(),
    surface_setup = DefaultExchangeCoefficients(params),
    job_id = "atmos_sim",
    output_dir = "output",
    restart_file = nothing,
    start_date = DateTime(2010, 1, 1),
    tracers = [],
    callbacks = (),
    diagnostics = (),
    discrete_hydrostatic_balance = false,
    itime = true,
)
    if !isnothing(restart_file)
        (Y, t_start) = get_state_restart(
            restart_file,
            start_date,
            hash(model),
            comms_ctx,
            itime,
        )
        spaces = get_spaces_restart(Y)
    else
        spaces = get_spaces(domain, params, comms_ctx)
        Y = ICs.atmos_state(
            initial_condition(params),
            model,
            spaces.center_space,
            spaces.face_space,
        )
        CA.InitialConditions.overwrite_initial_conditions!(
            initial_condition,
            Y,
            params.thermodynamics_params,
        )
    end
    # TODO: Deal with domain not having topography
    steady_state_velocity = get_steady_state_velocity(
        params,
        Y,
        domain.topography,
        initial_condition,
        domain.mesh_warp_type,
    )

    p = build_cache(
        Y,
        model,
        params,
        surface_setup,
        dt,
        start_date,
        tracers,
        steady_state_velocity,
    )
    discrete_hydrostatic_balance &&
        set_discrete_hydrostatic_balanced_state!(Y, p)

    # TODO: Move this to a separate function, add the same checks and behavior 
    # as in get_callbacks
    callback_set = SciMLBase.CallbackSet(callbacks...)

    integrator_args, integrator_kwargs =
        args_integrator(Y, p, (t_start, t_end), ode_algo, callback_set)
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
