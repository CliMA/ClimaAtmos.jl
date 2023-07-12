using Dates: DateTime, @dateformat_str
using NCDatasets
using Dierckx
using DiffEqBase
using ImageFiltering
using Interpolations
import ClimaCore: InputOutput, Meshes, Spaces
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaAtmos as CA
import LinearAlgebra
import ClimaCore.Fields
import OrdinaryDiffEq as ODE
import ClimaTimeSteppers as CTS
import DiffEqCallbacks as DEQ

function get_atmos(config::AtmosConfig, turbconv_params)
    (; parsed_args) = config
    FT = eltype(config)
    moisture_model = get_moisture_model(parsed_args)
    precip_model = get_precipitation_model(parsed_args)
    radiation_mode = get_radiation_mode(parsed_args, FT)
    forcing_type = get_forcing_type(parsed_args)

    diffuse_momentum = !(forcing_type isa HeldSuarezForcing)

    edmfx_adv_test = parsed_args["edmfx_adv_test"]
    @assert edmfx_adv_test in (false, true)

    edmfx_entr_detr = parsed_args["edmfx_entr_detr"]
    @assert edmfx_entr_detr in (false, true)

    edmfx_sgs_flux = parsed_args["edmfx_sgs_flux"]
    @assert edmfx_sgs_flux in (false, true)

    edmfx_nh_pressure = parsed_args["edmfx_nh_pressure"]
    @assert edmfx_nh_pressure in (false, true)

    model_config = get_model_config(parsed_args)
    vert_diff = get_vertical_diffusion_model(diffuse_momentum, parsed_args, FT)
    atmos = AtmosModel(;
        moisture_model,
        model_config,
        perf_mode = get_perf_mode(parsed_args),
        energy_form = get_energy_form(parsed_args, vert_diff),
        radiation_mode,
        subsidence = get_subsidence_model(parsed_args, radiation_mode, FT),
        ls_adv = get_large_scale_advection_model(parsed_args, FT),
        edmf_coriolis = get_edmf_coriolis(parsed_args, FT),
        edmfx_adv_test,
        edmfx_entr_detr,
        edmfx_sgs_flux,
        edmfx_nh_pressure,
        precip_model,
        forcing_type,
        turbconv_model = get_turbconv_model(
            FT,
            moisture_model,
            precip_model,
            parsed_args,
            turbconv_params,
        ),
        non_orographic_gravity_wave = get_non_orographic_gravity_wave_model(
            parsed_args,
            model_config,
            FT,
        ),
        orographic_gravity_wave = get_orographic_gravity_wave_model(
            parsed_args,
            FT,
        ),
        hyperdiff = get_hyperdiffusion_model(parsed_args, FT),
        vert_diff,
        viscous_sponge = get_viscous_sponge_model(parsed_args, FT),
        rayleigh_sponge = get_rayleigh_sponge_model(parsed_args, FT),
    )

    @info "AtmosModel: \n$(summary(atmos))"
    return atmos
end

function get_numerics(parsed_args)
    # wrap each upwinding mode in a Val for dispatch
    numerics = (;
        energy_upwinding = Val(Symbol(parsed_args["energy_upwinding"])),
        tracer_upwinding = Val(Symbol(parsed_args["tracer_upwinding"])),
        density_upwinding = Val(Symbol(parsed_args["density_upwinding"])),
        edmfx_upwinding = Val(Symbol(parsed_args["edmfx_upwinding"])),
        apply_limiter = parsed_args["apply_limiter"],
        bubble = parsed_args["bubble"],
    )
    @info "numerics" numerics...

    return numerics
end

function get_spaces(parsed_args, params, comms_ctx)

    FT = eltype(params)
    z_elem = Int(parsed_args["z_elem"])
    z_max = FT(parsed_args["z_max"])
    dz_bottom = FT(parsed_args["dz_bottom"])
    dz_top = FT(parsed_args["dz_top"])
    topography = parsed_args["topography"]
    bubble = parsed_args["bubble"]

    @assert topography in ("NoWarp", "DCMIP200", "Earth", "Agnesi", "Schar")
    if topography == "DCMIP200"
        warp_function = topography_dcmip200
    elseif topography == "Agnesi"
        warp_function = topography_agnesi
    elseif topography == "Schar"
        warp_function = topography_schar
    elseif topography == "NoWarp"
        warp_function = nothing
    elseif topography == "Earth"
        data_path = joinpath(topo_elev_dataset_path(), "ETOPO1_coarse.nc")
        earth_spline = NCDataset(data_path) do data
            zlevels = data["elevation"][:]
            lon = data["longitude"][:]
            lat = data["latitude"][:]
            # Apply Smoothing
            smooth_degree = Int(parsed_args["smoothing_order"])
            esmth = imfilter(zlevels, Kernel.gaussian(smooth_degree))
            linear_interpolation(
                (lon, lat),
                esmth,
                extrapolation_bc = (Periodic(), Flat()),
            )
        end
        @info "Generated interpolation stencil"
        warp_function = generate_topography_warp(earth_spline)
    end
    @info "Topography" topography


    h_elem = parsed_args["h_elem"]
    radius = CAP.planet_radius(params)
    center_space, face_space = if parsed_args["config"] == "sphere"
        nh_poly = parsed_args["nh_poly"]
        quad = Spaces.Quadratures.GLL{nh_poly + 1}()
        horizontal_mesh = cubed_sphere_mesh(; radius, h_elem)
        h_space =
            make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
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
                z_stretch;
                surface_warp = warp_function,
                topo_smoothing = parsed_args["topo_smoothing"],
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
        if bubble
            @warn "Bubble correction not compatible with single column configuration. It will be switched off."
            bubble = false
        end
        h_space =
            make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
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
        h_space =
            make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
        z_stretch = if parsed_args["z_stretch"]
            Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
        else
            Meshes.Uniform()
        end
        make_hybrid_spaces(
            h_space,
            z_max,
            z_elem,
            z_stretch;
            surface_warp = warp_function,
        )
    elseif parsed_args["config"] == "plane"
        FT = eltype(params)
        nh_poly = parsed_args["nh_poly"]
        quad = Spaces.Quadratures.GLL{nh_poly + 1}()
        x_elem = Int(parsed_args["x_elem"])
        x_max = FT(parsed_args["x_max"])
        horizontal_mesh =
            periodic_line_mesh(; x_max = x_max, x_elem = x_elem)
        h_space =
            make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
        z_stretch = if parsed_args["z_stretch"]
            Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
        else
            Meshes.Uniform()
        end
        make_hybrid_spaces(
            h_space,
            z_max,
            z_elem,
            z_stretch;
            surface_warp = warp_function,
        )
    end
    ncols = Fields.ncolumns(center_space)
    ndofs_total = ncols * z_elem
    hspace = Spaces.horizontal_space(center_space)
    quad_style = Spaces.quadrature_style(hspace)
    Nq = Spaces.Quadratures.degrees_of_freedom(quad_style)

    @info "Resolution stats: " Nq h_elem z_elem ncols ndofs_total
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

# get_state(simulation, parsed_args, spaces, params, atmos)
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

function get_initial_condition(parsed_args)
    if isnothing(parsed_args["turbconv_case"])
        if parsed_args["initial_condition"] in [
            "DryBaroclinicWave",
            "MoistBaroclinicWave",
            "DecayingProfile",
            "MoistBaroclinicWaveWithEDMF",
            "MoistAdiabaticProfileEDMFX",
        ]
            return getproperty(ICs, Symbol(parsed_args["initial_condition"]))(
                parsed_args["perturb_initstate"],
            )
        elseif parsed_args["initial_condition"] in [
            "IsothermalProfile",
            "Bomex",
            "AgnesiHProfile",
            "DryDensityCurrentProfile",
            "RisingThermalBubbleProfile",
            "ScharProfile",
        ]
            return getproperty(ICs, Symbol(parsed_args["initial_condition"]))()
        else
            error(
                "Unknown `initial_condition`: $(parsed_args["initial_condition"])",
            )
        end
    else
        # turbconv_case is also used for surface fluxes for TRMM and ARM cases.
        # I don't want to change that right now, so I'm leaving the
        # EDMF logic as is. This should be obsolete soon.
        return getproperty(ICs, Symbol(parsed_args["turbconv_case"]))()
    end
end

function get_surface_setup(parsed_args)
    return getproperty(SurfaceConditions, Symbol(parsed_args["surface_setup"]))()
end

is_explicit_CTS_algo_type(alg_or_tableau) =
    alg_or_tableau <: CTS.ERKAlgorithmName

is_imex_CTS_algo_type(alg_or_tableau) =
    alg_or_tableau <: CTS.IMEXARKAlgorithmName

is_implicit_type(::typeof(ODE.IMEXEuler)) = true
is_implicit_type(alg_or_tableau) =
    alg_or_tableau <: Union{
        ODE.OrdinaryDiffEqImplicitAlgorithm,
        ODE.OrdinaryDiffEqAdaptiveImplicitAlgorithm,
    } || is_imex_CTS_algo_type(alg_or_tableau)

is_ordinary_diffeq_newton(::typeof(ODE.IMEXEuler)) = true
is_ordinary_diffeq_newton(alg_or_tableau) =
    alg_or_tableau <: Union{
        ODE.OrdinaryDiffEqNewtonAlgorithm,
        ODE.OrdinaryDiffEqNewtonAdaptiveAlgorithm,
    }

is_imex_CTS_algo(::CTS.IMEXAlgorithm) = true
is_imex_CTS_algo(::DiffEqBase.AbstractODEAlgorithm) = false

is_implicit(::ODE.OrdinaryDiffEqImplicitAlgorithm) = true
is_implicit(::ODE.OrdinaryDiffEqAdaptiveImplicitAlgorithm) = true
is_implicit(ode_algo) = is_imex_CTS_algo(ode_algo)

is_rosenbrock(::ODE.Rosenbrock23) = true
is_rosenbrock(::ODE.Rosenbrock32) = true
is_rosenbrock(::DiffEqBase.AbstractODEAlgorithm) = false
use_transform(ode_algo) =
    !(is_imex_CTS_algo(ode_algo) || is_rosenbrock(ode_algo))

additional_integrator_kwargs(::DiffEqBase.AbstractODEAlgorithm) = (;
    adaptive = false,
    progress = isinteractive(),
    progress_steps = isinteractive() ? 1 : 1000,
)
additional_integrator_kwargs(::CTS.DistributedODEAlgorithm) = (;
    kwargshandle = ODE.KeywordArgSilent, # allow custom kwargs
    adjustfinal = true,
    # TODO: enable progress bars in ClimaTimeSteppers
)

is_cts_algo(::DiffEqBase.AbstractODEAlgorithm) = false
is_cts_algo(::CTS.DistributedODEAlgorithm) = true

jacobi_flags(::TotalEnergy) = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK)
jacobi_flags(::PotentialTemperature) = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact)

function jac_kwargs(ode_algo, Y, energy_form)
    if is_implicit(ode_algo)
        W = SchurComplementW(
            Y,
            use_transform(ode_algo),
            jacobi_flags(energy_form),
        )
        if use_transform(ode_algo)
            return (; jac_prototype = W, Wfact_t = Wfact!)
        else
            return (; jac_prototype = W, Wfact = Wfact!)
        end
    else
        return NamedTuple()
    end
end

#=
    ode_configuration(Y, parsed_args, atmos)

Returns the ode algorithm
=#
function ode_configuration(Y, parsed_args, atmos)
    FT = Spaces.undertype(axes(Y.c))
    ode_name = parsed_args["ode_algo"]
    alg_or_tableau = if startswith(ode_name, "ODE.")
        @warn "apply_limiter flag is ignored for OrdinaryDiffEq algorithms"
        getproperty(ODE, Symbol(split(ode_name, ".")[2]))
    else
        getproperty(CTS, Symbol(ode_name))
    end
    @info "Using ODE config: `$alg_or_tableau`"

    if is_explicit_CTS_algo_type(alg_or_tableau)
        return CTS.ExplicitAlgorithm(alg_or_tableau())
    elseif !is_implicit_type(alg_or_tableau)
        return alg_or_tableau()
    elseif is_ordinary_diffeq_newton(alg_or_tableau)
        if parsed_args["max_newton_iters"] == 1
            error("OridinaryDiffEq requires at least 2 Newton iterations")
        end
        # κ like a relative tolerance; its default value in ODE is 0.01
        nlsolve = ODE.NLNewton(;
            κ = parsed_args["max_newton_iters"] == 2 ? Inf : 0.01,
            max_iter = parsed_args["max_newton_iters"],
        )
        return alg_or_tableau(; linsolve = linsolve!, nlsolve)
    elseif is_imex_CTS_algo_type(alg_or_tableau)
        newtons_method = CTS.NewtonsMethod(;
            max_iters = parsed_args["max_newton_iters"],
            krylov_method = if parsed_args["use_krylov_method"]
                CTS.KrylovMethod(;
                    jacobian_free_jvp = CTS.ForwardDiffJVP(;
                        step_adjustment = FT(
                            parsed_args["jvp_step_adjustment"],
                        ),
                    ),
                    forcing_term = if parsed_args["use_dynamic_krylov_rtol"]
                        α = FT(parsed_args["eisenstat_walker_forcing_alpha"])
                        CTS.EisenstatWalkerForcing(; α)
                    else
                        CTS.ConstantForcing(FT(parsed_args["krylov_rtol"]))
                    end,
                )
            else
                nothing
            end,
            convergence_checker = if parsed_args["use_newton_rtol"]
                norm_condition = CTS.MaximumRelativeError(
                    FT(parsed_args["newton_rtol"]),
                )
                CTS.ConvergenceChecker(; norm_condition)
            else
                nothing
            end,
        )
        return CTS.IMEXAlgorithm(alg_or_tableau(), newtons_method)
    else
        return alg_or_tableau(; linsolve = linsolve!)
    end
end

thermo_state_type(::DryModel, ::Type{FT}) where {FT} = TD.PhaseDry{FT}
thermo_state_type(::EquilMoistModel, ::Type{FT}) where {FT} = TD.PhaseEquil{FT}
thermo_state_type(::NonEquilMoistModel, ::Type{FT}) where {FT} =
    TD.PhaseNonEquil{FT}


function get_callbacks(parsed_args, simulation, atmos, params)
    FT = eltype(params)
    (; dt) = simulation

    callbacks = ()
    if startswith(parsed_args["ode_algo"], "ODE.")
        callbacks = (callbacks..., call_every_n_steps(dss_callback!))
    end
    dt_save_to_disk = time_to_seconds(parsed_args["dt_save_to_disk"])
    if !(dt_save_to_disk == Inf)
        callbacks = (
            callbacks...,
            call_every_dt(
                save_to_disk_func,
                dt_save_to_disk;
                skip_first = simulation.restart,
            ),
        )
    end

    dt_save_restart = time_to_seconds(parsed_args["dt_save_restart"])
    if !(dt_save_restart == Inf)
        callbacks =
            (callbacks..., call_every_dt(save_restart_func, dt_save_restart))
    end

    if is_distributed(simulation.comms_ctx)
        callbacks = (
            callbacks...,
            call_every_n_steps(
                gc_func,
                parse(Int, get(ENV, "CLIMAATMOS_GC_NSTEPS", "1000")),
                skip_first = true,
            ),
        )
    end

    if parsed_args["check_conservation"]
        callbacks = (
            callbacks...,
            call_every_n_steps(
                flux_accumulation!;
                skip_first = true,
                call_at_end = true,
            ),
        )
    end

    if atmos.radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        # TODO: better if-else criteria?
        dt_rad = if parsed_args["config"] == "column"
            dt
        else
            FT(time_to_seconds(parsed_args["dt_rad"]))
        end
        callbacks =(
            callbacks..., 
            call_every_dt(rrtmgp_model_callback!, dt_rad), 
        )
    end

    if atmos.turbconv_model isa TC.EDMFModel
        callbacks = (
            callbacks...,
            call_every_n_steps(turb_conv_affect_filter!; skip_first = true),
        )
    end

    if parsed_args["surface_setup"] == "MoninObukhovInitTemperature"
        callbacks = (
            callbacks...,
            call_every_n_steps(update_surface_temp!; skip_first = true),
        )
    end

    return ODE.CallbackSet(callbacks...)
end


function get_cache(
    Y,
    parsed_args,
    params,
    spaces,
    atmos,
    numerics,
    simulation,
    initial_condition,
    surface_setup,
)
    _default_cache = default_cache(
        Y,
        parsed_args,
        params,
        atmos,
        spaces,
        numerics,
        simulation,
        surface_setup,
    )
    merge(
        _default_cache,
        additional_cache(
            Y,
            _default_cache,
            parsed_args,
            params,
            atmos,
            simulation.dt,
            initial_condition,
        ),
    )
end

function get_simulation(config::AtmosConfig, comms_ctx)
    (; parsed_args) = config
    FT = eltype(config)

    job_id = if isnothing(parsed_args["job_id"])
        s = argparse_settings()
        job_id_from_parsed_args(s, parsed_args)
    else
        parsed_args["job_id"]
    end
    default_output = haskey(ENV, "CI") ? job_id : joinpath("output", job_id)
    out_dir = parsed_args["output_dir"]
    output_dir = isnothing(out_dir) ? default_output : out_dir
    mkpath(output_dir)

    sim = (;
        comms_ctx,
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

function args_integrator(parsed_args, Y, p, tspan, ode_algo, callback)
    (; atmos, simulation) = p
    (; dt) = simulation
    dt_save_to_sol = time_to_seconds(parsed_args["dt_save_to_sol"])

    s = @timed_str begin
        func = if parsed_args["split_ode"]
            implicit_func = ODE.ODEFunction(
                implicit_tendency!;
                jac_kwargs(ode_algo, Y, atmos.energy_form)...,
                tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= 0),
            )
            if is_cts_algo(ode_algo)
                CTS.ClimaODEFunction(;
                    T_lim! = limited_tendency!,
                    T_exp! = remaining_tendency!,
                    T_imp! = implicit_func,
                    # Can we just pass implicit_tendency! and jac_prototype etc.?
                    lim! = limiters_func!,
                    dss!,
                )
            else
                ODE.SplitFunction(implicit_func, remaining_tendency!)
            end
        else
            remaining_tendency! # should be total_tendency!
        end
    end
    @info "Define ode function: $s"
    problem = ODE.ODEProblem(func, Y, tspan, p)
    saveat = if dt_save_to_sol == Inf
        tspan[2]
    elseif tspan[2] % dt_save_to_sol == 0
        dt_save_to_sol
    else
        [tspan[1]:dt_save_to_sol:tspan[2]..., tspan[2]]
    end # ensure that tspan[2] is always saved
    @info "dt_save_to_sol: $dt_save_to_sol, length(saveat): $(length(saveat))"
    args = (problem, ode_algo)
    kwargs = (; saveat, callback, dt, additional_integrator_kwargs(ode_algo)...)
    return (args, kwargs)
end

import ClimaComms, Logging, NVTX
function get_comms_context(parsed_args)
    device = if parsed_args["device"] == "CUDADevice"
        ClimaComms.CUDADevice()
    elseif parsed_args["device"] == "CPUMultiThreaded" || Threads.nthreads() > 1
        ClimaComms.CPUMultiThreaded()
    else
        ClimaComms.CPUSingleThreaded()
    end
    comms_ctx = ClimaComms.context(device)
    ClimaComms.init(comms_ctx)
    if ClimaComms.iamroot(comms_ctx)
        Logging.global_logger(Logging.ConsoleLogger(stderr, Logging.Info))
    else
        Logging.global_logger(Logging.NullLogger())
    end
    @info "Running on $(nameof(typeof(device)))."
    if comms_ctx isa ClimaComms.SingletonCommsContext
        @info "Setting up single-process ClimaAtmos run"
    else
        @info "Setting up distributed ClimaAtmos run" nprocs =
            ClimaComms.nprocs(comms_ctx)
    end
    if NVTX.isactive()
        # makes output on buildkite a bit nicer
        if ClimaComms.iamroot(comms_ctx)
            atexit() do
                println("--- Saving profiler information")
            end
        end
    end

    return comms_ctx
end

function get_integrator(config::AtmosConfig)
    params = create_parameter_set(config)

    atmos = get_atmos(config, params.turbconv_params)
    numerics = get_numerics(config.parsed_args)
    simulation = get_simulation(config, config.comms_ctx)
    initial_condition = get_initial_condition(config.parsed_args)
    surface_setup = get_surface_setup(config.parsed_args)

    s = @timed_str begin
        if simulation.restart
            (Y, t_start) = get_state_restart(config.comms_ctx)
            spaces = get_spaces_restart(Y)
        else
            spaces = get_spaces(config.parsed_args, params, config.comms_ctx)
            Y = ICs.atmos_state(
                initial_condition(params),
                atmos,
                spaces.center_space,
                spaces.face_space,
            )
            t_start = Spaces.undertype(axes(Y.c))(0)
        end
    end
    @info "Allocating Y: $s"

    s = @timed_str begin
        p = get_cache(
            Y,
            config.parsed_args,
            params,
            spaces,
            atmos,
            numerics,
            simulation,
            initial_condition,
            surface_setup,
        )
    end
    @info "Allocating cache (p): $s"

    if config.parsed_args["discrete_hydrostatic_balance"]
        set_discrete_hydrostatic_balanced_state!(Y, p)
    end

    s = @timed_str begin
        ode_algo = ode_configuration(Y, config.parsed_args, atmos)
    end
    @info "ode_configuration: $s"

    s = @timed_str begin
        callback = get_callbacks(config.parsed_args, simulation, atmos, params)
    end
    @info "get_callbacks: $s"
    @info "n_steps_per_cycle_per_cb: $(n_steps_per_cycle_per_cb(callback, simulation.dt))"
    @info "n_steps_per_cycle: $(n_steps_per_cycle(callback, simulation.dt))"
    tspan = (t_start, simulation.t_end)
    s = @timed_str begin
        integrator_args, integrator_kwargs = args_integrator(
            config.parsed_args,
            Y,
            p,
            tspan,
            ode_algo,
            callback,
        )
    end

    s = @timed_str begin
        integrator = ODE.init(integrator_args...; integrator_kwargs...)
    end
    @info "init integrator: $s"
    return integrator
end
