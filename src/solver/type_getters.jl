using Adapt
using Dates: DateTime, @dateformat_str
using Dierckx
using Interpolations
import NCDatasets
import ClimaCore: InputOutput, Meshes, Spaces, Quadratures
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaAtmos as CA
import LinearAlgebra
import ClimaCore.Fields
import ClimaTimeSteppers as CTS
import DiffEqCallbacks as DECB

function get_atmos(config::AtmosConfig, params)
    (; turbconv_params) = params
    (; parsed_args) = config
    FT = eltype(config)
    moisture_model = get_moisture_model(parsed_args)
    precip_model = get_precipitation_model(parsed_args)
    cloud_model = get_cloud_model(parsed_args)
    radiation_mode = get_radiation_mode(parsed_args, FT)
    forcing_type = get_forcing_type(parsed_args)

    diffuse_momentum = !(forcing_type isa HeldSuarezForcing)

    advection_test = parsed_args["advection_test"]
    @assert advection_test in (false, true)

    gs_tendency = parsed_args["gs_tendency"]
    @assert gs_tendency in (false, true)

    edmfx_entr_model = get_entrainment_model(parsed_args)
    edmfx_detr_model = get_detrainment_model(parsed_args)

    edmfx_sgs_mass_flux = parsed_args["edmfx_sgs_mass_flux"]
    @assert edmfx_sgs_mass_flux in (false, true)

    edmfx_sgs_diffusive_flux = parsed_args["edmfx_sgs_diffusive_flux"]
    @assert edmfx_sgs_diffusive_flux in (false, true)

    edmfx_nh_pressure = parsed_args["edmfx_nh_pressure"]
    @assert edmfx_nh_pressure in (false, true)

    edmfx_velocity_relaxation = parsed_args["edmfx_velocity_relaxation"]
    @assert edmfx_velocity_relaxation in (false, true)

    implicit_diffusion = parsed_args["implicit_diffusion"]
    @assert implicit_diffusion in (true, false)

    model_config = get_model_config(parsed_args)
    vert_diff =
        get_vertical_diffusion_model(diffuse_momentum, parsed_args, params, FT)
    atmos = AtmosModel(;
        moisture_model,
        model_config,
        perf_mode = get_perf_mode(parsed_args),
        radiation_mode,
        subsidence = get_subsidence_model(parsed_args, radiation_mode, FT),
        ls_adv = get_large_scale_advection_model(parsed_args, FT),
        edmf_coriolis = get_edmf_coriolis(parsed_args, FT),
        advection_test,
        gs_tendency,
        edmfx_entr_model,
        edmfx_detr_model,
        edmfx_sgs_mass_flux,
        edmfx_sgs_diffusive_flux,
        edmfx_nh_pressure,
        edmfx_velocity_relaxation,
        precip_model,
        cloud_model,
        forcing_type,
        turbconv_model = get_turbconv_model(FT, parsed_args, turbconv_params),
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
        diff_mode = implicit_diffusion ? Implicit() : Explicit(),
        viscous_sponge = get_viscous_sponge_model(parsed_args, params, FT),
        rayleigh_sponge = get_rayleigh_sponge_model(parsed_args, params, FT),
        sfc_temperature = get_sfc_temperature_form(parsed_args),
        surface_model = get_surface_model(parsed_args),
        surface_albedo = get_surface_albedo_model(parsed_args, params, FT),
        numerics = get_numerics(parsed_args),
    )
    @assert !@any_reltype(atmos, (UnionAll, DataType))

    @info "AtmosModel: \n$(summary(atmos))"
    return atmos
end

function get_numerics(parsed_args)
    test_dycore =
        parsed_args["test_dycore_consistency"] ? TestDycoreConsistency() :
        nothing

    energy_upwinding = Val(Symbol(parsed_args["energy_upwinding"]))
    tracer_upwinding = Val(Symbol(parsed_args["tracer_upwinding"]))
    edmfx_upwinding = Val(Symbol(parsed_args["edmfx_upwinding"]))
    edmfx_sgsflux_upwinding =
        Val(Symbol(parsed_args["edmfx_sgsflux_upwinding"]))

    limiter = parsed_args["apply_limiter"] ? CA.QuasiMonotoneLimiter() : nothing

    # wrap each upwinding mode in a Val for dispatch
    numerics = AtmosNumerics(;
        energy_upwinding,
        tracer_upwinding,
        edmfx_upwinding,
        edmfx_sgsflux_upwinding,
        limiter,
        test_dycore_consistency = test_dycore,
        use_reference_state = parsed_args["use_reference_state"],
    )
    @info "numerics $(summary(numerics))"

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
    deep = parsed_args["deep_atmosphere"]

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
        array_type = ClimaComms.array_type(comms_ctx.device)
        earth_spline = NCDatasets.NCDataset(data_path) do data
            zlevels = Array(data["elevation"])
            lon = Array(data["longitude"])
            lat = Array(data["latitude"])
            # Apply Smoothing
            smooth_degree = Int(parsed_args["smoothing_order"])
            esmth = CA.gaussian_smooth(zlevels, smooth_degree)
            Adapt.adapt(
                array_type,
                linear_interpolation(
                    (lon, lat),
                    esmth,
                    extrapolation_bc = (Periodic(), Flat()),
                ),
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
        quad = Quadratures.GLL{nh_poly + 1}()
        horizontal_mesh = cubed_sphere_mesh(; radius, h_elem)
        h_space =
            make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
        z_stretch = if parsed_args["z_stretch"]
            Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
        else
            Meshes.Uniform()
        end
        if warp_function == nothing
            make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; deep)
        else
            make_hybrid_spaces(
                h_space,
                z_max,
                z_elem,
                z_stretch;
                parsed_args = parsed_args,
                surface_warp = warp_function,
                deep,
            )
        end
    elseif parsed_args["config"] == "column" # single column
        @warn "perturb_initstate flag is ignored for single column configuration"
        FT = eltype(params)
        Δx = FT(1) # Note: This value shouldn't matter, since we only have 1 column.
        quad = Quadratures.GL{1}()
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
        make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args)
    elseif parsed_args["config"] == "box"
        FT = eltype(params)
        nh_poly = parsed_args["nh_poly"]
        quad = Quadratures.GLL{nh_poly + 1}()
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
            parsed_args,
            surface_warp = warp_function,
            deep,
        )
    elseif parsed_args["config"] == "plane"
        FT = eltype(params)
        nh_poly = parsed_args["nh_poly"]
        quad = Quadratures.GLL{nh_poly + 1}()
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
            parsed_args,
            surface_warp = warp_function,
            deep,
        )
    end
    ncols = Fields.ncolumns(center_space)
    ndofs_total = ncols * z_elem
    hspace = Spaces.horizontal_space(center_space)
    quad_style = Spaces.quadrature_style(hspace)
    Nq = Quadratures.degrees_of_freedom(quad_style)

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

function get_spaces_restart(Y)
    center_space = axes(Y.c)
    face_space = axes(Y.f)
    return (; center_space, face_space)
end

function get_state_restart(config::AtmosConfig)
    (; parsed_args, comms_ctx) = config
    restart_file = parsed_args["restart_file"]
    @assert !isnothing(restart_file)
    reader = InputOutput.HDF5Reader(restart_file, comms_ctx)
    Y = InputOutput.read_field(reader, "Y")
    t_start = InputOutput.HDF5.read_attribute(reader.file, "time")
    return (Y, t_start)
end

function get_initial_condition(parsed_args)
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
        "Nieuwstadt",
        "GABLS",
        "GATE_III",
        "Soares",
        "Bomex",
        "LifeCycleTan2018",
        "ARM_SGP",
        "DYCOMS_RF01",
        "DYCOMS_RF02",
        "Rico",
        "TRMM_LBA",
        "SimplePlume",
    ]
        return getproperty(ICs, Symbol(parsed_args["initial_condition"]))(
            parsed_args["prognostic_tke"],
        )
    elseif parsed_args["initial_condition"] in [
        "IsothermalProfile",
        "AgnesiHProfile",
        "DryDensityCurrentProfile",
        "RisingThermalBubbleProfile",
        "ScharProfile",
        "PrecipitatingColumn",
    ]
        return getproperty(ICs, Symbol(parsed_args["initial_condition"]))()
    else
        error(
            "Unknown `initial_condition`: $(parsed_args["initial_condition"])",
        )
    end
end

function get_surface_setup(parsed_args)
    return getproperty(SurfaceConditions, Symbol(parsed_args["surface_setup"]))()
end

is_explicit_CTS_algo_type(alg_or_tableau) =
    alg_or_tableau <: CTS.ERKAlgorithmName

is_imex_CTS_algo_type(alg_or_tableau) =
    alg_or_tableau <: CTS.IMEXARKAlgorithmName

is_implicit_type(alg_or_tableau) = is_imex_CTS_algo_type(alg_or_tableau)

is_imex_CTS_algo(::CTS.IMEXAlgorithm) = true
is_imex_CTS_algo(::SciMLBase.AbstractODEAlgorithm) = false

is_implicit(ode_algo) = is_imex_CTS_algo(ode_algo)

is_rosenbrock(::SciMLBase.AbstractODEAlgorithm) = false
use_transform(ode_algo) =
    !(is_imex_CTS_algo(ode_algo) || is_rosenbrock(ode_algo))

additional_integrator_kwargs(::SciMLBase.AbstractODEAlgorithm) = (;
    adaptive = false,
    progress = isinteractive(),
    progress_steps = isinteractive() ? 1 : 1000,
)
import DiffEqBase
additional_integrator_kwargs(::CTS.DistributedODEAlgorithm) = (;
    kwargshandle = DiffEqBase.KeywordArgSilent, # allow custom kwargs
    adjustfinal = true,
    # TODO: enable progress bars in ClimaTimeSteppers
)

is_cts_algo(::SciMLBase.AbstractODEAlgorithm) = false
is_cts_algo(::CTS.DistributedODEAlgorithm) = true

function jac_kwargs(ode_algo, Y, atmos, parsed_args)
    if is_implicit(ode_algo)
        A = ImplicitEquationJacobian(
            Y,
            atmos;
            approximate_solve_iters = parsed_args["approximate_linear_solve_iters"],
            transform_flag = use_transform(ode_algo),
        )
        if use_transform(ode_algo)
            return (; jac_prototype = A, Wfact_t = Wfact!)
        else
            return (; jac_prototype = A, Wfact = Wfact!)
        end
    else
        return NamedTuple()
    end
end

#=
    ode_configuration(Y, parsed_args)

Returns the ode algorithm
=#
function ode_configuration(::Type{FT}, parsed_args) where {FT}
    ode_name = parsed_args["ode_algo"]
    alg_or_tableau = getproperty(CTS, Symbol(ode_name))
    @info "Using ODE config: `$alg_or_tableau`"

    if is_explicit_CTS_algo_type(alg_or_tableau)
        return CTS.ExplicitAlgorithm(alg_or_tableau())
    elseif !is_implicit_type(alg_or_tableau)
        return alg_or_tableau()
    elseif is_imex_CTS_algo_type(alg_or_tableau)
        newtons_method = CTS.NewtonsMethod(;
            max_iters = parsed_args["max_newton_iters_ode"],
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

function get_sim_info(config::AtmosConfig)
    (; parsed_args) = config
    FT = eltype(config)

    job_id = if isnothing(parsed_args["job_id"])
        job_id_from_config(parsed_args)
    else
        parsed_args["job_id"]
    end
    default_output = haskey(ENV, "CI") ? job_id : joinpath("output", job_id)
    out_dir = parsed_args["output_dir"]
    output_dir = isnothing(out_dir) ? default_output : out_dir
    mkpath(output_dir)

    sim = (;
        output_dir,
        restart = !isnothing(parsed_args["restart_file"]),
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
    (; atmos, dt) = p
    dt_save_to_sol = time_to_seconds(parsed_args["dt_save_to_sol"])

    s = @timed_str begin
        func = if parsed_args["split_ode"]
            implicit_func = SciMLBase.ODEFunction(
                implicit_tendency!;
                jac_kwargs(ode_algo, Y, atmos, parsed_args)...,
                tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= 0),
            )
            if is_cts_algo(ode_algo)
                CTS.ClimaODEFunction(;
                    T_exp_T_lim! = remaining_tendency!,
                    T_imp! = implicit_func,
                    # Can we just pass implicit_tendency! and jac_prototype etc.?
                    lim! = limiters_func!,
                    dss!,
                    post_explicit! = set_precomputed_quantities!,
                    post_implicit! = set_precomputed_quantities!,
                )
            else
                SciMLBase.SplitFunction(implicit_func, remaining_tendency!)
            end
        else
            remaining_tendency! # should be total_tendency!
        end
    end
    @info "Define ode function: $s"
    problem = SciMLBase.ODEProblem(func, Y, tspan, p)
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
    device = if parsed_args["device"] == "auto"
        ClimaComms.device()
    elseif parsed_args["device"] == "CUDADevice"
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

function get_simulation(config::AtmosConfig)
    params = create_parameter_set(config)
    atmos = get_atmos(config, params)

    sim_info = get_sim_info(config)
    job_id = sim_info.job_id
    output_dir = sim_info.output_dir

    CP.log_parameter_information(
        config.toml_dict,
        joinpath(output_dir, "$(job_id)_parameters.toml"),
        strict = true,
    )
    YAML.write_file(joinpath(output_dir, "$job_id.yml"), config.parsed_args)

    if sim_info.restart
        s = @timed_str begin
            (Y, t_start) = get_state_restart(config)
            spaces = get_spaces_restart(Y)
        end
        @info "Allocating Y: $s"
    else
        spaces = get_spaces(config.parsed_args, params, config.comms_ctx)
    end

    initial_condition = get_initial_condition(config.parsed_args)
    surface_setup = get_surface_setup(config.parsed_args)

    if !sim_info.restart
        s = @timed_str begin
            Y = ICs.atmos_state(
                initial_condition(params),
                atmos,
                spaces.center_space,
                spaces.face_space,
            )
            t_start = Spaces.undertype(axes(Y.c))(0)
        end
        @info "Allocating Y: $s"
    end

    tracers = get_tracers(config.parsed_args)

    s = @timed_str begin
        p = build_cache(
            Y,
            atmos,
            params,
            surface_setup,
            sim_info,
            tracers.aerosol_names,
        )
    end
    @info "Allocating cache (p): $s"

    if config.parsed_args["discrete_hydrostatic_balance"]
        set_discrete_hydrostatic_balanced_state!(Y, p)
    end

    FT = Spaces.undertype(axes(Y.c))
    s = @timed_str begin
        ode_algo = ode_configuration(FT, config.parsed_args)
    end
    @info "ode_configuration: $s"

    s = @timed_str begin
        callback = get_callbacks(config, sim_info, atmos, params, Y, p, t_start)
    end
    @info "get_callbacks: $s"

    # Initialize diagnostics
    s = @timed_str begin
        diagnostics, writers =
            get_diagnostics(config.parsed_args, atmos, Spaces.axes(Y.c))
    end
    @info "initializing diagnostics: $s"

    length(diagnostics) > 0 && @info "Computing diagnostics:"

    # First, we convert all the ScheduledDiagnosticTime into ScheduledDiagnosticIteration,
    # ensuring that there is consistency in the timestep and the periods and translating
    # those periods that depended on the timestep
    diagnostics_iterations =
        [CAD.ScheduledDiagnosticIterations(d, sim_info.dt) for d in diagnostics]

    # For diagnostics that perform reductions, the storage is used for the values computed
    # at each call. Reductions also save the accumulated value in diagnostic_accumulators.
    diagnostic_storage = Dict()
    diagnostic_accumulators = Dict()
    diagnostic_counters = Dict()

    s = @timed_str begin
        diagnostics_functions = CAD.get_callbacks_from_diagnostics(
            diagnostics_iterations,
            diagnostic_storage,
            diagnostic_accumulators,
            diagnostic_counters,
            output_dir,
        )
    end
    @info "Prepared diagnostic callbacks: $s"

    # It would be nice to just pass the callbacks to the integrator. However, this leads to
    # a significant increase in compile time for reasons that are not known. For this
    # reason, we only add one callback to the integrator, and this function takes care of
    # executing the other callbacks. This single function is orchestrate_diagnostics

    orchestrate_diagnostics(integrator) =
        CAD.orchestrate_diagnostics(integrator, diagnostics_functions)

    diagnostic_callbacks =
        call_every_n_steps(orchestrate_diagnostics, skip_first = true)

    # The generic constructor for SciMLBase.CallbackSet has to split callbacks into discrete
    # and continuous. This is not hard, but can introduce significant latency. However, all
    # the callbacks in ClimaAtmos are discrete_callbacks, so we directly pass this
    # information to the constructor
    continuous_callbacks = tuple()
    discrete_callbacks = (callback..., diagnostic_callbacks)

    s = @timed_str begin
        all_callbacks =
            SciMLBase.CallbackSet(continuous_callbacks, discrete_callbacks)
    end
    @info "Prepared SciMLBase.CallbackSet callbacks: $s"
    steps_cycle_non_diag = n_steps_per_cycle_per_cb(all_callbacks, sim_info.dt)
    steps_cycle_diag =
        n_steps_per_cycle_per_cb_diagnostic(diagnostics_functions)
    steps_cycle = lcm([steps_cycle_non_diag..., steps_cycle_diag...])
    @info "n_steps_per_cycle_per_cb (non diagnostics): $steps_cycle_non_diag"
    @info "n_steps_per_cycle_per_cb_diagnostic: $steps_cycle_diag"
    @info "n_steps_per_cycle (non diagnostics): $steps_cycle"

    tspan = (t_start, sim_info.t_end)
    s = @timed_str begin
        integrator_args, integrator_kwargs = args_integrator(
            config.parsed_args,
            Y,
            p,
            tspan,
            ode_algo,
            all_callbacks,
        )
    end

    s = @timed_str begin
        integrator = SciMLBase.init(integrator_args...; integrator_kwargs...)
    end
    @info "init integrator: $s"
    reset_graceful_exit(output_dir)

    s = @timed_str init_diagnostics!(
        diagnostics_iterations,
        diagnostic_storage,
        diagnostic_accumulators,
        diagnostic_counters,
        output_dir,
        Y,
        p,
        t_start;
        warn_allocations = config.parsed_args["warn_allocations_diagnostics"],
    )
    @info "Init diagnostics: $s"

    return AtmosSimulation(
        job_id,
        output_dir,
        sim_info.start_date,
        sim_info.t_end,
        writers,
        integrator,
    )
end

# Compatibility with old get_integrator
function get_integrator(config::AtmosConfig)
    Base.depwarn(
        "get_integrator is deprecated, use get_simulation instead",
        :get_integrator,
    )
    return get_simulation(config).integrator
end
