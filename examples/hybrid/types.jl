using Dates: DateTime, @dateformat_str

abstract type AbstractMoistureModel end
struct DryModel <: AbstractMoistureModel end
struct EquilMoistModel <: AbstractMoistureModel end
struct NonEquilMoistModel <: AbstractMoistureModel end

abstract type AbstractCompressibilityModel end
struct CompressibleFluid <: AbstractCompressibilityModel end
struct AnelasticFluid <: AbstractCompressibilityModel end

function moisture_model(parsed_args)
    moisture_name = parsed_args["moist"]
    @assert moisture_name in ("dry", "equil", "nonequil")
    return if moisture_name == "dry"
        DryModel()
    elseif moisture_name == "equil"
        EquilMoistModel()
    elseif moisture_name == "nonequil"
        NonEquilMoistModel()
    end
end

abstract type AbstractEnergyFormulation end
struct PotentialTemperature <: AbstractEnergyFormulation end
struct TotalEnergy <: AbstractEnergyFormulation end
struct InternalEnergy <: AbstractEnergyFormulation end

function energy_form(parsed_args)
    energy_name = parse_arg(parsed_args, "energy_name", "rhoe")
    @assert energy_name in ("rhoe", "rhoe_int", "rhotheta")
    return if energy_name == "rhoe"
        TotalEnergy()
    elseif energy_name == "rhoe_int"
        InternalEnergy()
    elseif energy_name == "rhotheta"
        PotentialTemperature()
    end
end


function radiation_model(parsed_args)
    radiation_name = parsed_args["rad"]
    @assert radiation_name in
            (nothing, "clearsky", "gray", "allsky", "allskywithclear")
    return if radiation_name == "clearsky"
        RRTMGPI.ClearSkyRadiation()
    elseif radiation_name == "gray"
        RRTMGPI.GrayRadiation()
    elseif radiation_name == "allsky"
        RRTMGPI.AllSkyRadiation()
    elseif radiation_name == "allskywithclear"
        RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics()
    else
        nothing
    end
end

abstract type AbstractMicrophysicsModel end
struct Microphysics0Moment <: AbstractMicrophysicsModel end

function microphysics_model(parsed_args)
    microphysics_name = parsed_args["microphy"]
    @assert microphysics_name in (nothing, "0M")
    return if microphysics_name == nothing
        nothing
    elseif microphysics_name == "0M"
        Microphysics0Moment()
    end
end

abstract type AbstractForcing end
struct HeldSuarezForcing <: AbstractForcing end

function forcing_type(parsed_args)
    forcing = parsed_args["forcing"]
    @assert forcing in (nothing, "held_suarez")
    return if forcing == nothing
        nothing
    elseif forcing == "held_suarez"
        HeldSuarezForcing()
    end
end

function turbconv_model(FT, parsed_args, namelist)
    turbconv = parsed_args["turbconv"]
    precip_model = nothing
    @assert turbconv in (nothing, "edmf")
    return if turbconv == "edmf"
        TC.EDMFModel(FT, namelist, precip_model)
    else
        nothing
    end
end

Base.broadcastable(x::AbstractMoistureModel) = Ref(x)
Base.broadcastable(x::AbstractEnergyFormulation) = Ref(x)
Base.broadcastable(x::AbstractMicrophysicsModel) = Ref(x)
Base.broadcastable(x::AbstractForcing) = Ref(x)


function get_model_spec(::Type{FT}, parsed_args, namelist) where {FT}

    model_spec = (;
        moisture_model = moisture_model(parsed_args),
        energy_form = energy_form(parsed_args),
        radiation_model = radiation_model(parsed_args),
        microphysics_model = microphysics_model(parsed_args),
        forcing_type = forcing_type(parsed_args),
        turbconv_model = turbconv_model(FT, parsed_args, namelist),
        anelastic_dycore = parsed_args["anelastic_dycore"],
    )

    return model_spec
end

function get_numerics(parsed_args)

    numerics = (;
        upwinding_mode = Symbol(
            parse_arg(parsed_args, "upwinding", "third_order"),
        )
    )
    @assert numerics.upwinding_mode in (:none, :first_order, :third_order)

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
        output_dir,
        restart = haskey(ENV, "RESTART_FILE"),
        job_id,
        dt = FT(time_to_seconds(parsed_args["dt"])),
        start_date = DateTime(parsed_args["start_date"], dateformat"yyyymmdd"),
    )

    return sim
end

function get_spaces(parsed_args, params, comms_ctx)

    FT = eltype(params)
    z_elem = Int(parsed_args["z_elem"])
    z_max = FT(parsed_args["z_max"])
    dz_bottom = FT(parsed_args["dz_bottom"])
    dz_top = FT(parsed_args["dz_top"])

    h_elem = parsed_args["h_elem"]
    radius = CAP.planet_radius(params)
    center_space, face_space = if parsed_args["config"] == "sphere"
        quad = Spaces.Quadratures.GLL{5}()
        horizontal_mesh = cubed_sphere_mesh(; radius, h_elem)
        h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx)
        z_stretch = if parsed_args["z_stretch"]
            Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
        else
            Meshes.Uniform()
        end
        make_hybrid_spaces(h_space, z_max, z_elem, z_stretch)
    elseif parsed_args["config"] == "column" # single column
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
        return get_state_restart(simulation.is_distributed)
    else
        return get_state_fresh_start(args...)
    end
end

function get_state_restart(is_distributed)
    @assert haskey(ENV, "RESTART_FILE")
    restart_file_name = if is_distributed
        split(ENV["RESTART_FILE"], ".jld2")[1] * "_pid$pid.jld2"
    else
        ENV["RESTART_FILE"]
    end
    local Y, t_start
    JLD2.jldopen(restart_file_name) do data
        Y = data["Y"]
        t_start = data["t"]
    end
    return (Y, t_start)
end

function get_state_fresh_start(parsed_args, spaces, params, model_spec)
    (; center_space, face_space) = spaces
    FT = eltype(params)
    t_start = FT(0)

    center_initial_condition = if is_baro_wave(parsed_args)
        center_initial_condition_baroclinic_wave
    elseif parsed_args["config"] == "sphere"
        center_initial_condition_sphere
    elseif parsed_args["config"] == "column"
        center_initial_condition_column
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
#=
(; jac_kwargs, alg_kwargs, ode_algorithm) =
    ode_config(Y, parsed_args, model_spec)
=#
function ode_config(Y, parsed_args, model_spec)
    # TODO: add max_newton_iters, newton_κ, test_implicit_solver to parsed_args?
    max_newton_iters = 2 # only required by ODE algorithms that use Newton's method
    newton_κ = Inf # similar to a reltol for Newton's method (default is 0.01)
    test_implicit_solver = false # makes solver extremely slow when set to `true`
    jacobian_flags = jacobi_flags(model_spec.energy_form)
    ode_algorithm = getproperty(ODE, Symbol(parsed_args["ode_algo"]))

    ode_algorithm_type =
        ode_algorithm isa Function ? typeof(ode_algorithm()) : ode_algorithm
    if ode_algorithm_type <: Union{
        ODE.OrdinaryDiffEqImplicitAlgorithm,
        ODE.OrdinaryDiffEqAdaptiveImplicitAlgorithm,
    }
        use_transform =
            !(ode_algorithm_type in (ODE.Rosenbrock23, ODE.Rosenbrock32))
        W = SchurComplementW(
            Y,
            use_transform,
            jacobian_flags,
            test_implicit_solver,
        )
        Wfact! =
            if :ρe_tot in propertynames(Y.c) &&
               W.flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :no_∂ᶜp∂ᶜK &&
               W.flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
                Wfact_special!
            else
                Wfact_generic!
            end
        jac_kwargs =
            use_transform ? (; jac_prototype = W, Wfact_t = Wfact!) :
            (; jac_prototype = W, Wfact = Wfact!)

        alg_kwargs = (; linsolve = linsolve!)
        if ode_algorithm_type <: Union{
            ODE.OrdinaryDiffEqNewtonAlgorithm,
            ODE.OrdinaryDiffEqNewtonAdaptiveAlgorithm,
        }
            alg_kwargs = (;
                alg_kwargs...,
                nlsolve = ODE.NLNewton(;
                    κ = newton_κ,
                    max_iter = max_newton_iters,
                ),
            )
        end
    else
        jac_kwargs = alg_kwargs = ()
    end
    return (; jac_kwargs, alg_kwargs, ode_algorithm)
end

function get_integrator(parsed_args, Y, p, tspan, jac_kwargs, callback)
    (; dt) = p.simulation
    FT = eltype(tspan)
    dt_save_to_sol = time_to_seconds(parsed_args["dt_save_to_sol"])
    show_progress_bar = isinteractive()
    additional_solver_kwargs = () # e.g., abstol and reltol

    if :ρe_tot in propertynames(Y.c)
        implicit_tendency! = implicit_tendency_special!
    else
        implicit_tendency! = implicit_tendency_generic!
    end

    problem = if parsed_args["split_ode"]
        ODE.SplitODEProblem(
            ODE.ODEFunction(
                implicit_tendency!;
                jac_kwargs...,
                tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= FT(0)),
            ),
            remaining_tendency!,
            Y,
            tspan,
            p,
        )
    else
        ODE.ODEProblem(remaining_tendency!, Y, tspan, p)
    end
    integrator = ODE.init(
        problem,
        ode_algorithm(; alg_kwargs...);
        saveat = dt_save_to_sol == Inf ? [] : dt_save_to_sol,
        callback = callback,
        dt = dt,
        adaptive = false,
        progress = show_progress_bar,
        progress_steps = isinteractive() ? 1 : 1000,
        additional_solver_kwargs...,
    )
    return integrator
end
