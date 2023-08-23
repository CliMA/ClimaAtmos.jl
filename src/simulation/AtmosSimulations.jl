import ClimaComms
import ClimaCore: Spaces
import ClimaTimeSteppers as CTS
import ..InitialConditions as ICs

"""
    AtmosSimulation

    Container for an atmospheric model and an integrator (which contains the state and a lot
    of other information). The state (and the other variables) are updated as the simulation
    moves forward. `AtmosSimulation` provides convenient methods to obtain useful
    information, such as `.state` to retrieve the state.

"""
struct AtmosSimulation{AM, OD}
    integrator::OD
end

function Base.getproperty(sim::AtmosSimulation, v::Symbol)
    if v == :state
        return sim.integrator.u
    elseif v == :callbacks
        return sim.integrator.callback
    elseif v == :comms_ctx
        return ClimaComms.context(axes(sim.integrator.u.c))
    elseif v == :float_type
        return Spaces.undertype(axes(sim.integrator.c))
    elseif v == :time
        return sim.integrator.t
    elseif v == :atmos_model
        return sim.integrator.p.atmos
    else
        return getfield(sim, v)
    end
end

"""
    AtmosSimulation(atmos_model::AtmosModel, Δt;
                    timestepper::AtmosTimeStepper = AtmosTimeStepper(ClimaTimeSteppers.ARS343()),
                    callbacks = [],
                    stop_time = Inf,
                    Δt_save_solution = Inf,
                    output_dir = "output",
                    )

Construct a `AtmosSimulation` for the given `atmos_model` running on the computational `domain`
starting from `initial_conditions`.

Positional arguments
=====================

- `atmos_model`: Atmospheric model to be simulated defined as an `AtmosModel` object.

- `Δt`: Time step of the simulation in seconds.

Keyword arguments
=================

- `timestepper`: Algorithm for the numerical integration, as defined by an
                 `AtmosTimeStepper` object. These algorithms can be trivially constructed
                 from timesteppers in `ClimaTimeSteppers` and `OrdinaryDiffEq`.

- `callbacks`: Collection of functions that have to be run after every integration step.
               They can be used for output, diagnostics, log messages, and much more. By
               default, no callback is added.

- `stop_time`: Integrate until this simulation time in seconds. If Inf, the simulation will
               have to be stopped using other means (e.g., callbacks), or it will continue
               forever.

- `Δt_save_solution`: Intervals of simulation time in seconds when the solution should be
                      saved.

- `output_dir`: Directory where to save the output. If the directory does not exist, it will
                be created.
"""
function AtmosSimulation(
    atmos_model::AtmosModel,
    Δt::Real;
    timestepper = AtmosTimeStepper(CTS.ARS343()),
    callbacks = [],
    stop_time::Real = Inf,
    Δt_save_solution::Real = Inf,
    output_dir = "output",
)

    # In a nutshell, this constructor prepares the arguments that are passed to ODE.init().

    # TODO: Currently, the cache contains information about the model and about the
    # simulation. However, in the AtmosSimulation paradigm, the two are split, and we define
    # the model before having information about the simulation. So, we have to create a new
    # atmos_model that is identical to the one that was provided, except that it has an
    # updated cache.
    #
    # This is not super nice because it means that
    # AtmosSimulation(atmos_model, ...).atmos_model.cache != atmos_model.cache
    #
    # However, this restriction can be removed with a breaking change in how the cache is
    # designed.

    FT = atmos_model.float_type

    tspan = (start_time, stop_time) = zero(FT), FT(stop_time)

    # TODO: (Wishlist)
    #
    # In the future, it would be nice for the following function to depend on atmos_model,
    # so that user can provide their own custom models and tendencies.
    implicit_func = ODE.ODEFunction(
        implicit_tendency!;
        jac_kwargs(
            timestepper.algorithm,
            atmos_model.initial_state,
            atmos_model.energy_form,
        )...,
        tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= 0),
    )
    if timestepper.is_cts
        func = CTS.ClimaODEFunction(;
            T_lim! = limited_tendency!,
            T_exp! = remaining_tendency!,
            T_imp! = implicit_func,
            lim! = limiters_func!,
            dss!,
        )
    else
        func = ODE.SplitFunction(implicit_func, remaining_tendency!)
    end

    # Output
    isdir(output_dir) ||
        @info "Creating output directory: $(mkpath(output_dir))"

    # Add missing field to the cache
    simulation = (;
        comms_ctx = atmos_model.comms_ctx,
        is_debugging_tc = false,
        output_dir = output_dir,
        restart = false,
        job_id = "A job",
        dt = Δt,
        start_date = DateTime("2023", dateformat"yyyymmdd"),
        t_end = stop_time,
    )

    # .simulation is already defined with one single element, we have to remove
    # it to add the correct entry
    cache_without_simulation = NamedTuple([
        (k, v) for (k, v) in pairs(atmos_model.cache) if k != :simulation
    ])

    atmos_model = AtmosModel(;
        atmos_model.model_config,
        atmos_model.perf_mode,
        atmos_model.moisture_model,
        atmos_model.energy_form,
        atmos_model.precip_model,
        atmos_model.forcing_type,
        atmos_model.subsidence,
        atmos_model.radiation_mode,
        atmos_model.ls_adv,
        atmos_model.edmf_coriolis,
        atmos_model.advection_test,
        atmos_model.edmfx_entr_detr,
        atmos_model.edmfx_sgs_flux,
        atmos_model.edmfx_nh_pressure,
        atmos_model.turbconv_model,
        atmos_model.non_orographic_gravity_wave,
        atmos_model.orographic_gravity_wave,
        atmos_model.hyperdiff,
        atmos_model.vert_diff,
        atmos_model.viscous_sponge,
        atmos_model.rayleigh_sponge,
        atmos_model.sfc_temperature,
        atmos_model.surface_model,
        atmos_model.surface_setup,
        atmos_model.initial_state,
        cache = merge(cache_without_simulation, (; simulation)),
    )

    # Ensure that we save the solution at stop_time
    if Δt_save_solution == Inf
        saveat = stop_time
    elseif stop_time % Δt_save_solution == 0
        saveat = Δt_save_solution
    else
        saveat = [tspan[1]:Δt_save_solution:stop_time..., stop_time]
    end

    problem = ODE.ODEProblem(
        func,
        atmos_model.initial_state,
        tspan,
        atmos_model.cache,
    )

    integrator = ODE.init(
        problem,
        timestepper.algorithm;
        callback = ODE.CallbackSet(callbacks...),
        dt = Δt,
        saveat,
    )

    return AtmosSimulation(integrator)
end

"""
    AtmosSimulation(simulation::AtmosSimulation)

Perform the given `simulation`.

"""

function solve!(simulation::AtmosSimulation)
    ODE.step!(simulation.integrator)
    precompile_callbacks(simulation.integrator)
    GC.gc()
    try
        if is_distributed(simulation.comms_ctx)
            # GC.enable(false) # disabling GC causes a memory leak
            ClimaComms.barrier(simulation.comms_ctx)
            (sol, walltime) = timed_solve!(simulation.integrator)
            ClimaComms.barrier(simulation.comms_ctx)
            GC.enable(true)
            return AtmosSolveResults(sol, :success, walltime)
        else
            (sol, walltime) = timed_solve!(simulation.integrator)
            return AtmosSolveResults(sol, :success, walltime)
        end
    catch ret_code
        @error "ClimaAtmos simulation crashed. Stacktrace for failed simulation" exception =
            (ret_code, catch_backtrace())
        return AtmosSolveResults(nothing, :simulation_crashed, nothing)
    end

end

"""
    AtmosModel

    Construct an `AtmosModel` on the given `domain`, with the given `initial_condition`.

Keyword arguments
=================

- `moisture_model`:

"""
function AtmosModel(
    domain,
    initial_condition;
    moisture_model,
    energy_form,
    sfc_temperature_form,
    surface_model,
    surface_setup,
    params = default_parameter_set(domain.float_type),
    perf_mode = PerfStandard(),
    radiation_mode = nothing,
    subsidence = nothing,
    ls_adv = nothing,
    edmf_coriolis = nothing,
    precip_model = NoPrecipitation(),
    forcing_type = nothing,
    turbconv_model = nothing,
    non_orographic_gravity_wave = nothing,
    orographic_gravity_wave = nothing,
    hyperdiff = nothing,
    vert_diff = nothing,
    viscous_sponge = nothing,
    rayleigh_sponge = nothing,
)
    # This function is mostly a compatibility layer with the current
    # argument-based interface, combined with light-processing of the given
    # arguments.

    float_type = domain.float_type

    # TODO: Remove this compatibility layer
    # We have to work around some stuff in parameters handling
    parsed_args = Dict(
        "entr_coeff" => float_type(NaN),
        "detr_coeff" => float_type(NaN),
        "dt" => "1secs",
        "config" => "column",
        "energy_name" =>
            isa(energy_form, TotalEnergy) ? "rhoe" : "rhotheta",
        "forcing" => nothing,
        "turbconv" => nothing,
        "rad" => nothing,
        "use_reference_state" => true,
        "test_dycore_consistency" => false,
        "energy_upwinding" => "none",
        "tracer_upwinding" => "none",
        "density_upwinding" => "none",
        "edmfx_upwinding" => "none",
        "apply_limiter" => false,
        "bubble" => false,
        "debugging_tc" => false,
        "t_end" => "600mins",
        "job_id" => "YAY",
        "output_dir" => "/tmp/",
        "start_date" => "2023",
        "idealized_insolation" => false,
        "idealized_clouds" => false,
        "split_ode" => true,
    )

    C = typeof(domain.comms_ctx)
    TD = typeof(params)
    PA = typeof(parsed_args)
    atmos_config = AtmosConfig{float_type, TD, PA, C}(
        params,
        parsed_args,
        domain.comms_ctx,
    )

    # NOTE: What we call model_config here is not what AtmosModel calls
    # model_config
    model_config = create_parameter_set(atmos_config)

    initial_condition_generator = initial_condition(model_config)

    # ICs.atmos_state takes a model without a state as an input to create a state,
    # so we'll have create a temporary AtmosModel that doesn't have a state/cache.
    model = AtmosModel(;
        model_config = domain.type,
        perf_mode = perf_mode,
        moisture_model = moisture_model,
        energy_form = energy_form,
        precip_model = precip_model,
        forcing_type = forcing_type,
        subsidence = subsidence,
        radiation_mode = radiation_mode,
        ls_adv = ls_adv,
        edmf_coriolis = edmf_coriolis,
        advection_test = false,
        edmfx_entr_detr = false,
        edmfx_sgs_flux = false,
        edmfx_nh_pressure = false,
        turbconv_model = turbconv_model,
        non_orographic_gravity_wave = non_orographic_gravity_wave,
        orographic_gravity_wave = orographic_gravity_wave,
        hyperdiff = hyperdiff,
        vert_diff = vert_diff,
        viscous_sponge = viscous_sponge,
        rayleigh_sponge = rayleigh_sponge,
        sfc_temperature = sfc_temperature_form,
        surface_model = surface_model,
        surface_setup = nothing,
        initial_state = nothing,
        cache = nothing,
    )

    initial_state = InitialConditions.atmos_state(
        initial_condition_generator,
        model,
        domain.center_space,
        domain.face_space,
    )

    spaces = (; domain.center_space, domain.face_space)
    cache = get_cache(
        initial_state,
        parsed_args,
        model_config,
        spaces,
        model,
        get_numerics(parsed_args),
        (; dt = parsed_args["dt"]),    # The details of the simulation will be introduced by the AtmosSimulation object
        initial_state,
        surface_setup,
    )

    return AtmosModel(;
        model_config = domain.type,
        perf_mode = perf_mode,
        moisture_model = moisture_model,
        energy_form = energy_form,
        precip_model = precip_model,
        forcing_type = forcing_type,
        subsidence = subsidence,
        radiation_mode = radiation_mode,
        ls_adv = ls_adv,
        edmf_coriolis = edmf_coriolis,
        advection_test = false,
        edmfx_entr_detr = false,
        edmfx_sgs_flux = false,
        edmfx_nh_pressure = false,
        turbconv_model = turbconv_model,
        non_orographic_gravity_wave = non_orographic_gravity_wave,
        orographic_gravity_wave = orographic_gravity_wave,
        hyperdiff = hyperdiff,
        vert_diff = vert_diff,
        viscous_sponge = viscous_sponge,
        rayleigh_sponge = rayleigh_sponge,
        sfc_temperature = sfc_temperature_form,
        surface_model = surface_model,
        surface_setup = nothing,
        initial_state = initial_state,
        cache = cache,
    )
end


import CLIMAParameters as CP
function default_parameter_set(;
    float_type = Float64,
    override_toml_files = Set(),
    toml_dict = CP.create_toml_dict(
        float_type;
        override_file = CP.merge_toml_files(override_toml_files),
    ),
)

    return toml_dict
end
