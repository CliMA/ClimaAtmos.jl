using Dates: DateTime, @dateformat_str
import ClimaCore: InputOutput
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaAtmos as CA
import ClimaAtmos:
    DryModel,
    EquilMoistModel,
    NonEquilMoistModel,
    CompressibleFluid,
    AnelasticFluid,
    PotentialTemperature,
    TotalEnergy,
    InternalEnergy,
    Microphysics0Moment,
    HeldSuarezForcing,
    BulkSurfaceScheme,
    MoninObukhovSurface,
    SingleColumnModel,
    SphericalModel,
    BoxModel

import ClimaCore: InputOutput

function get_model_spec(::Type{FT}, parsed_args, namelist) where {FT}
    # should this live in the radiation model?
    idealized_h2o = parsed_args["idealized_h2o"]
    @assert idealized_h2o in (true, false)
    non_orographic_gravity_wave = parsed_args["non_orographic_gravity_wave"]
    @assert non_orographic_gravity_wave in (true, false)

    moisture_model = CA.moisture_model(parsed_args)
    precip_model = CA.precipitation_model(parsed_args)
    radiation_mode = CA.radiation_mode(parsed_args, FT)

    model_spec = (;
        moisture_model,
        model_config = CA.model_config(parsed_args),
        energy_form = CA.energy_form(parsed_args),
        perturb_initstate = parsed_args["perturb_initstate"],
        idealized_h2o,
        radiation_mode,
        subsidence = CA.subsidence_model(parsed_args, radiation_mode, FT),
        ls_adv = CA.large_scale_advection_model(parsed_args, FT),
        edmf_coriolis = CA.edmf_coriolis(parsed_args, FT),
        precip_model,
        forcing_type = CA.forcing_type(parsed_args),
        turbconv_model = CA.turbconv_model(
            FT,
            moisture_model,
            precip_model,
            parsed_args,
            namelist,
        ),
        compressibility_model = CA.compressibility_model(parsed_args),
        surface_scheme = CA.surface_scheme(FT, parsed_args),
        C_E = FT(parsed_args["C_E"]),
        non_orographic_gravity_wave,
    )

    return model_spec
end

function get_numerics(parsed_args)
    # wrap each upwinding mode in a Val for dispatch
    numerics = (;
        energy_upwinding = Val(Symbol(parsed_args["energy_upwinding"])),
        tracer_upwinding = Val(Symbol(parsed_args["tracer_upwinding"])),
        apply_limiter = parsed_args["apply_limiter"],
    )
    @info "numerics" numerics...

    return numerics
end

function get_simulation(::Type{FT}, parsed_args) where {FT}

    job_id = if isnothing(parsed_args["job_id"])
        (s, default_parsed_args) = parse_commandline()
        job_id_from_parsed_args(s, parsed_args)
    else
        parsed_args["job_id"]
    end
    default_output = haskey(ENV, "CI") ? job_id : joinpath("output", job_id)
    output_dir = parse_arg(parsed_args, "output_dir", default_output)
    mkpath(output_dir)

    sim = (;
        is_distributed = haskey(ENV, "CLIMACORE_DISTRIBUTED"),
        is_debugging_tc = parsed_args["debugging_tc"],
        output_dir,
        restart = haskey(ENV, "RESTART_FILE"),
        job_id,
        dt = FT(time_to_seconds(parsed_args["dt"])),
        start_date = DateTime(parsed_args["start_date"], dateformat"yyyymmdd"),
        t_end = FT(time_to_seconds(parsed_args["t_end"])),
    )
    n_steps = floor(Int, sim.t_end / sim.dt)
    @info(
        "Time info:",
        dt = parsed_args["dt"],
        t_end = parsed_args["t_end"],
        floor_n_steps = n_steps,
    )

    return sim
end

function get_spaces(parsed_args, params, comms_ctx)

    FT = eltype(params)
    z_elem = Int(parsed_args["z_elem"])
    z_max = FT(parsed_args["z_max"])
    dz_bottom = FT(parsed_args["dz_bottom"])
    dz_top = FT(parsed_args["dz_top"])
    topography = parsed_args["topography"]

    if topography == "DCMIP200"
        warp_function = topography_dcmip200
    elseif topography == "NoWarp"
        warp_function = nothing
    end
    @assert topography in ("NoWarp", "DCMIP200")
    @info "Topography" topography

    h_elem = parsed_args["h_elem"]
    radius = CAP.planet_radius(params)
    center_space, face_space = if parsed_args["config"] == "sphere"
        nh_poly = parsed_args["nh_poly"]
        quad = Spaces.Quadratures.GLL{nh_poly + 1}()
        horizontal_mesh = cubed_sphere_mesh(; radius, h_elem)
        h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx)
        z_stretch = if parsed_args["z_stretch"]
            Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
        else
            Meshes.Uniform()
        end
        if warp_function == nothing
            make_hybrid_spaces(h_space, z_max, z_elem, z_stretch)
        else
            make_hybrid_spaces(
                h_space,
                z_max,
                z_elem,
                z_stretch,
                surface_warp = warp_function,
            )
        end
    elseif parsed_args["config"] == "column" # single column
        @warn "perturb_initstate flag is ignored for single column configuration"
        FT = eltype(params)
        Δx = FT(1) # Note: This value shouldn't matter, since we only have 1 column.
        quad = Spaces.Quadratures.GL{1}()
        horizontal_mesh = periodic_rectangle_mesh(;
            x_max = Δx,
            y_max = Δx,
            x_elem = 1,
            y_elem = 1,
        )
        h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx)
        z_stretch = if parsed_args["z_stretch"]
            Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
        else
            Meshes.Uniform()
        end
        make_hybrid_spaces(h_space, z_max, z_elem, z_stretch)
    elseif parsed_args["config"] == "box"
        FT = eltype(params)
        nh_poly = parsed_args["nh_poly"]
        quad = Spaces.Quadratures.GLL{nh_poly + 1}()
        x_elem = Int(parsed_args["x_elem"])
        x_max = FT(parsed_args["x_max"])
        y_elem = Int(parsed_args["y_elem"])
        y_max = FT(parsed_args["y_max"])
        horizontal_mesh = periodic_rectangle_mesh(;
            x_max = x_max,
            y_max = y_max,
            x_elem = x_elem,
            y_elem = y_elem,
        )
        h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx)
        z_stretch = if parsed_args["z_stretch"]
            Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
        else
            Meshes.Uniform()
        end
        make_hybrid_spaces(h_space, z_max, z_elem, z_stretch)
    end
    return (;
        center_space,
        face_space,
        horizontal_mesh,
        quad,
        z_max,
        z_elem,
        z_stretch,
    )
end

# get_state(simulation, parsed_args, spaces, params, model_spec)
function get_state(simulation, args...)
    if simulation.restart
        return get_state_restart(comms_ctx)
    else
        return get_state_fresh_start(args...)
    end
end

function get_spaces_restart(Y)
    center_space = axes(Y.c)
    face_space = axes(Y.f)
    hspace = Spaces.horizontal_space(center_space)
    horizontal_mesh = hspace.topology.mesh
    quad = horizontal_mesh.ne + 1
    vertical_mesh = Spaces.vertical_topology(center_space).mesh
    z_max = vertical_mesh.domain.coord_max.z
    z_elem = length(vertical_mesh.faces) - 1
    return (; center_space, face_space, horizontal_mesh, quad, z_max, z_elem)
end

function get_state_restart(comms_ctx)
    @assert haskey(ENV, "RESTART_FILE")
    reader = InputOutput.HDF5Reader(ENV["RESTART_FILE"], comms_ctx)
    Y = InputOutput.read_field(reader, "Y")
    t_start = InputOutput.HDF5.read_attribute(reader.file, "time")
    return (Y, t_start)
end

function get_state_fresh_start(parsed_args, spaces, params, model_spec)
    (; center_space, face_space) = spaces
    FT = eltype(params)
    t_start = FT(0)

    center_initial_condition = if is_baro_wave(parsed_args)
        center_initial_condition_baroclinic_wave
    elseif parsed_args["config"] == "sphere"
        center_initial_condition_3d
    elseif parsed_args["config"] == "column"
        center_initial_condition_column
    elseif parsed_args["config"] == "box"
        center_initial_condition_3d
    end

    Y = init_state(
        center_initial_condition,
        face_initial_condition,
        center_space,
        face_space,
        params,
        model_spec,
    )
    return (Y, t_start)
end

import OrdinaryDiffEq as ODE
import ClimaTimeSteppers as CTS
#=
(; jac_kwargs, alg_kwargs, ode_algorithm) =
    ode_config(Y, parsed_args, model_spec)
=#
function ode_configuration(Y, parsed_args, model_spec)
    test_implicit_solver = false # makes solver extremely slow when set to `true`
    jacobian_flags = jacobi_flags(model_spec.energy_form)
    ode_algorithm = if startswith(parsed_args["ode_algo"], "ODE.")
        @warn "apply_limiter flag is ignored for OrdinaryDiffEq algorithms"
        getproperty(ODE, Symbol(split(parsed_args["ode_algo"], ".")[2]))
    else
        getproperty(CTS, Symbol(parsed_args["ode_algo"]))
    end

    ode_algorithm_type =
        ode_algorithm isa Function ? typeof(ode_algorithm()) : ode_algorithm
    is_imex_CTS_algo = ode_algorithm_type <: ClimaTimeSteppers.IMEXARKAlgorithm
    if ode_algorithm_type <: Union{
        ODE.OrdinaryDiffEqImplicitAlgorithm,
        ODE.OrdinaryDiffEqAdaptiveImplicitAlgorithm,
    } || is_imex_CTS_algo
        use_transform = !(
            is_imex_CTS_algo ||
            ode_algorithm_type in (ODE.Rosenbrock23, ODE.Rosenbrock32)
        )
        W = SchurComplementW(
            Y,
            use_transform,
            jacobian_flags,
            test_implicit_solver,
        )
        jac_kwargs =
            use_transform ? (; jac_prototype = W, Wfact_t = Wfact!) :
            (; jac_prototype = W, Wfact = Wfact!)

        alg_kwargs = (; linsolve = linsolve!)
        if ode_algorithm_type <: Union{
            ODE.OrdinaryDiffEqNewtonAlgorithm,
            ODE.OrdinaryDiffEqNewtonAdaptiveAlgorithm,
        }
            if parsed_args["max_newton_iters"] == 1
                error("OridinaryDiffEq requires at least 2 Newton iterations")
            end
            # κ like a relative tolerance; its default value in ODE is 0.01
            nlsolve = ODE.NLNewton(;
                κ = parsed_args["max_newton_iters"] == 2 ? Inf : 0.01,
                max_iter = parsed_args["max_newton_iters"],
            )
            alg_kwargs = (; alg_kwargs..., nlsolve)
        elseif is_imex_CTS_algo
            newtons_method = NewtonsMethod(;
                linsolve = linsolve!,
                max_iters = parsed_args["max_newton_iters"],
            )
            alg_kwargs = (; newtons_method)
        end
    else
        jac_kwargs = alg_kwargs = ()
    end
    return (; jac_kwargs, alg_kwargs, ode_algorithm)
end

function get_integrator(parsed_args, Y, p, tspan, ode_config, callback)
    (; jac_kwargs, alg_kwargs, ode_algorithm) = ode_config
    (; dt) = p.simulation
    FT = eltype(tspan)
    dt_save_to_sol = time_to_seconds(parsed_args["dt_save_to_sol"])
    show_progress_bar = isinteractive()

    problem = if parsed_args["split_ode"]
        remaining_func =
            startswith(parsed_args["ode_algo"], "ODE.") ?
            remaining_tendency! :
            ForwardEulerODEFunction(remaining_tendency_increment!)
        ODE.SplitODEProblem(
            ODE.ODEFunction(
                implicit_tendency!;
                jac_kwargs...,
                tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= FT(0)),
            ),
            remaining_func,
            Y,
            tspan,
            p,
        )
    else
        ODE.ODEProblem(remaining_tendency!, Y, tspan, p)
    end
    if startswith(parsed_args["ode_algo"], "ODE.")
        ode_algo = ode_algorithm(; alg_kwargs...)
        integrator_kwargs = (;
            adaptive = false,
            progress = show_progress_bar,
            progress_steps = isinteractive() ? 1 : 1000,
        )
    else
        ode_algo = ode_algorithm(alg_kwargs...)
        integrator_kwargs = (;
            kwargshandle = KeywordArgSilent, # allow custom kwargs
            adjustfinal = true,
            # TODO: enable progress bars in ClimaTimeSteppers
        )
    end
    saveat = if dt_save_to_sol == Inf
        tspan[2]
    elseif tspan[2] % dt_save_to_sol == 0
        dt_save_to_sol
    else
        [tspan[1]:dt_save_to_sol:tspan[2]..., tspan[2]]
    end # ensure that tspan[2] is always saved
    integrator =
        ODE.init(problem, ode_algo; saveat, callback, dt, integrator_kwargs...)
    return integrator
end
