using Adapt
using Dates: DateTime, @dateformat_str
using Interpolations
import NCDatasets
import ClimaUtilities.OutputPathGenerator
import ClimaCore: InputOutput, Meshes, Spaces, Quadratures
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaAtmos as CA
import ClimaCore.Fields
import ClimaTimeSteppers as CTS
import ClimaDiagnostics

function get_atmos(config::AtmosConfig, params)
    (; turbconv_params) = params
    (; parsed_args) = config
    FT = eltype(config)
    check_case_consistency(parsed_args)
    moisture_model = get_moisture_model(parsed_args)
    precip_model = get_precipitation_model(parsed_args)
    cloud_model = get_cloud_model(parsed_args)
    ozone = get_ozone(parsed_args)
    radiation_mode = get_radiation_mode(parsed_args, FT)
    forcing_type = get_forcing_type(parsed_args)
    call_cloud_diagnostics_per_stage =
        get_call_cloud_diagnostics_per_stage(parsed_args)

    if isnothing(ozone) && radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        @warn "prescribe_ozone is set to nothing with an RRTMGP model. Resetting to IdealizedOzone. This behavior will stop being supported in some future release"
        ozone = IdealizedOzone()
    end

    diffuse_momentum = !(forcing_type isa HeldSuarezForcing)

    advection_test = parsed_args["advection_test"]
    @assert advection_test in (false, true)

    implicit_diffusion = parsed_args["implicit_diffusion"]
    @assert implicit_diffusion in (true, false)

    implicit_sgs_advection = parsed_args["implicit_sgs_advection"]
    @assert implicit_sgs_advection in (true, false)

    edmfx_model = EDMFXModel(;
        entr_model = get_entrainment_model(parsed_args),
        detr_model = get_detrainment_model(parsed_args),
        sgs_mass_flux = Val(parsed_args["edmfx_sgs_mass_flux"]),
        sgs_diffusive_flux = Val(parsed_args["edmfx_sgs_diffusive_flux"]),
        nh_pressure = Val(parsed_args["edmfx_nh_pressure"]),
        filter = Val(parsed_args["edmfx_filter"]),
    )

    model_config = get_model_config(parsed_args)
    vert_diff =
        get_vertical_diffusion_model(diffuse_momentum, parsed_args, params, FT)

    atmos = AtmosModel(;
        moisture_model,
        model_config,
        ozone,
        radiation_mode,
        subsidence = get_subsidence_model(parsed_args, radiation_mode, FT),
        ls_adv = get_large_scale_advection_model(parsed_args, FT),
        external_forcing = get_external_forcing_model(parsed_args),
        edmf_coriolis = get_edmf_coriolis(parsed_args, FT),
        advection_test,
        tendency_model = get_tendency_model(parsed_args),
        edmfx_model,
        precip_model,
        cloud_model,
        forcing_type,
        call_cloud_diagnostics_per_stage,
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
        sgs_adv_mode = implicit_sgs_advection ? Implicit() : Explicit(),
        viscous_sponge = get_viscous_sponge_model(parsed_args, params, FT),
        rayleigh_sponge = get_rayleigh_sponge_model(parsed_args, params, FT),
        sfc_temperature = get_sfc_temperature_form(parsed_args),
        insolation = get_insolation_form(parsed_args),
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
    )
    @info "numerics $(summary(numerics))"

    return numerics
end

function get_spaces(parsed_args, params, comms_ctx)

    FT = eltype(params)
    z_elem = Int(parsed_args["z_elem"])
    z_max = FT(parsed_args["z_max"])
    dz_bottom = FT(parsed_args["dz_bottom"])
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
            Meshes.HyperbolicTangentStretching(dz_bottom)
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
            Meshes.HyperbolicTangentStretching(dz_bottom)
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
            Meshes.HyperbolicTangentStretching(dz_bottom)
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
            Meshes.HyperbolicTangentStretching(dz_bottom)
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

function get_state_restart(config::AtmosConfig, restart_file, atmos_model_hash)
    (; parsed_args, comms_ctx) = config

    @assert !isnothing(restart_file)
    reader = InputOutput.HDF5Reader(restart_file, comms_ctx)
    Y = InputOutput.read_field(reader, "Y")
    # TODO: Do not use InputOutput.HDF5 directly
    t_start = InputOutput.HDF5.read_attribute(reader.file, "time")
    if "atmos_model_hash" in keys(InputOutput.HDF5.attrs(reader.file))
        atmos_model_hash_in_restart =
            InputOutput.HDF5.read_attribute(reader.file, "atmos_model_hash")
        if atmos_model_hash_in_restart != atmos_model_hash
            error(
                "Restart file $(restart_file) was constructed with a different AtmosModel",
            )
        end
    end
    return (Y, t_start)
end

function get_initial_condition(parsed_args)
    if parsed_args["initial_condition"] in [
        "DryBaroclinicWave",
        "MoistBaroclinicWave",
        "MoistBaroclinicWaveWithEDMF",
    ]
        return getproperty(ICs, Symbol(parsed_args["initial_condition"]))(
            parsed_args["perturb_initstate"],
            parsed_args["deep_atmosphere"],
        )
    elseif parsed_args["initial_condition"] in
           ["DecayingProfile", "MoistAdiabaticProfileEDMFX"]
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
    elseif parsed_args["initial_condition"] == "ISDAC"
        ICs.ISDAC(
            parsed_args["prognostic_tke"],
            parsed_args["perturb_initstate"],
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
    elseif parsed_args["initial_condition"] == "GCM"
        @assert parsed_args["prognostic_tke"] == true
        return ICs.GCMDriven(
            parsed_args["external_forcing_file"],
            parsed_args["cfsite_number"],
        )
    else
        error(
            "Unknown `initial_condition`: $(parsed_args["initial_condition"])",
        )
    end
end

function get_surface_setup(parsed_args)
    parsed_args["surface_setup"] == "GCM" && return SurfaceConditions.GCMDriven(
        parsed_args["external_forcing_file"],
        parsed_args["cfsite_number"],
    )

    return getproperty(SurfaceConditions, Symbol(parsed_args["surface_setup"]))()
end

get_jacobian(ode_algo, Y, atmos, parsed_args) =
    if ode_algo isa Union{CTS.IMEXAlgorithm, CTS.RosenbrockAlgorithm}
        use_exact_jacobian = parsed_args["use_exact_jacobian"]
        always_update_exact_jacobian =
            parsed_args["n_steps_update_exact_jacobian"] == 0
        diagnostics_names =
            getindex.(get(parsed_args, "diagnostics", []), ("short_name",))
        preserve_unfactorized_jacobian =
            parsed_args["output_default_diagnostics"] ||
            "ejac1" in diagnostics_names ||
            any(contains("ejac1"), Iterators.flatten(diagnostics_names))
        exact_jacobian_alg = ExactJacobian(;
            always_update_exact_jacobian,
            preserve_unfactorized_jacobian,
        )
        approximate_solve_iters = parsed_args["approximate_linear_solve_iters"]
        approx_jacobian_alg = ApproxJacobian(; approximate_solve_iters)
        jacobian_algorithm = if parsed_args["debug_approximate_jacobian"]
            DebugJacobian(
                exact_jacobian_alg,
                approx_jacobian_alg;
                use_exact_jacobian,
            )
        else
            use_exact_jacobian ? exact_jacobian_alg : approx_jacobian_alg
        end
        @info "Jacobian algorithm: $(dump_string(jacobian_algorithm))"
        ImplicitEquationJacobian(jacobian_algorithm, Y, atmos)
    else
        nothing
    end

function ode_configuration(::Type{FT}, parsed_args) where {FT}
    ode_name = parsed_args["ode_algo"]
    ode_algo_name = getproperty(CTS, Symbol(ode_name))
    @info "Using ODE config: `$ode_algo_name`"
    return if ode_algo_name <: CTS.RosenbrockAlgorithmName
        CTS.RosenbrockAlgorithm(CTS.tableau(ode_algo_name()))
    elseif ode_algo_name <: CTS.ERKAlgorithmName
        CTS.ExplicitAlgorithm(ode_algo_name())
    else
        @assert ode_algo_name <: CTS.IMEXARKAlgorithmName
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
        CTS.IMEXAlgorithm(ode_algo_name(), newtons_method)
    end
end

thermo_state_type(::DryModel, ::Type{FT}) where {FT} = TD.PhaseDry{FT}
thermo_state_type(::EquilMoistModel, ::Type{FT}) where {FT} = TD.PhaseEquil{FT}
thermo_state_type(::NonEquilMoistModel, ::Type{FT}) where {FT} =
    TD.PhaseNonEquil{FT}

auto_detect_restart_file(::OutputPathGenerator.OutputPathGeneratorStyle, _) =
    error("auto_detect_restart_file works only with ActiveLink")

"""
    auto_detect_restart_file(::ActiveLinkStyle, base_output_dir)

Return the most recent restart file in the directory structure in `base_output_dir`, if any.

`auto_detect_restart_file` scans the content of `base_output_dir` matching the expected
names for output folders generated by `ActiveLinkStyle` and for restart files
(`dayDDDD.SSSSS.hdf5`). If no folder or no restart file is found, return `nothing`: this
means that the simulation cannot be automatically restarted. If a folder is found, look
inside it and return the latest restart file (latest measured by the time in the file name).
"""
function auto_detect_restart_file(
    output_dir_style::OutputPathGenerator.ActiveLinkStyle,
    base_output_dir,
)
    # if base_output_dir does not exist, we return restart_file = nothing because there is
    # no restart file to be detected
    isdir(base_output_dir) || return nothing

    # output_dir will be something like ABC/DEF/output_1234
    name_rx = r"output_(\d\d\d\d)"
    restart_file_rx = r"day\d+\.\w+\.hdf5"
    restart_file = nothing

    existing_outputs =
        filter(x -> !isnothing(match(name_rx, x)), readdir(base_output_dir))

    isempty(existing_outputs) && return nothing

    latest_output = first(sort(existing_outputs, rev = true))
    previous_folder = joinpath(base_output_dir, latest_output)
    possible_restart_files =
        filter(f -> occursin(restart_file_rx, f), readdir(previous_folder))
    if isempty(possible_restart_files)
        @warn "Detected folder $(previous_folder), but no restart file was found"
        return nothing
    end

    restart_file_name = last(CA.sort_files_by_time(possible_restart_files))
    restart_file = joinpath(previous_folder, restart_file_name)
    @assert isfile(restart_file) "Restart file does not exist"

    return restart_file
end

function get_sim_info(config::AtmosConfig)
    (; parsed_args) = config
    FT = eltype(config)

    (; job_id) = config
    default_output = haskey(ENV, "CI") ? job_id : joinpath("output", job_id)
    out_dir = parsed_args["output_dir"]
    base_output_dir = isnothing(out_dir) ? default_output : out_dir

    allowed_dir_styles = Dict(
        "activelink" => OutputPathGenerator.ActiveLinkStyle(),
        "removepreexisting" => OutputPathGenerator.RemovePreexistingStyle(),
    )

    requested_style = parsed_args["output_dir_style"]

    haskey(allowed_dir_styles, lowercase(requested_style)) ||
        error("output_dir_style $(requested_style) not available")

    output_dir_style = allowed_dir_styles[lowercase(requested_style)]

    # We look for a restart before creating a new output dir because we want to
    # look for previous folders
    restart_file =
        parsed_args["detect_restart_file"] ?
        auto_detect_restart_file(output_dir_style, base_output_dir) :
        parsed_args["restart_file"]

    output_dir = OutputPathGenerator.generate_output_path(
        base_output_dir;
        context = config.comms_ctx,
        style = output_dir_style,
    )

    isnothing(restart_file) ||
        @info "Restarting simulation from file $restart_file"

    sim = (;
        output_dir,
        restart = !isnothing(restart_file),
        restart_file,
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

function args_integrator(parsed_args, Y, p, tspan, ode_algo, jacobian, callback)
    (; dt) = p
    dt_save_to_sol = time_to_seconds(parsed_args["dt_save_to_sol"])

    s = @timed_str begin
        T_imp! = SciMLBase.ODEFunction(
            implicit_tendency!;
            jac_prototype = jacobian,
            Wfact = update_jacobian!,
        )
        ode_func = CTS.ClimaODEFunction(;
            T_exp_T_lim! = remaining_tendency!,
            T_imp!,
            lim! = limiters_func!,
            dss!,
            post_explicit! = set_precomputed_quantities!,
            post_implicit! = set_precomputed_quantities!,
        )
    end
    @info "Define ode function: $s"
    problem = SciMLBase.ODEProblem(ode_func, Y, tspan, p)
    saveat = if dt_save_to_sol == Inf
        tspan[2]
    elseif tspan[2] % dt_save_to_sol == 0
        dt_save_to_sol
    else
        [tspan[1]:dt_save_to_sol:tspan[2]..., tspan[2]]
    end # ensure that tspan[2] is always saved
    @info "dt_save_to_sol: $dt_save_to_sol, length(saveat): $(length(saveat))"
    args = (problem, ode_algo)
    kwargs = (;
        saveat,
        callback,
        dt,
        kwargshandle = DiffEqBase.KeywordArgSilent, # allow custom kwargs
        adjustfinal = true,
    )
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
    @info "Simulation info" job_id output_dir

    CP.log_parameter_information(
        config.toml_dict,
        joinpath(output_dir, "$(job_id)_parameters.toml"),
        strict = true,
    )
    YAML.write_file(joinpath(output_dir, "$job_id.yml"), config.parsed_args)

    if sim_info.restart
        s = @timed_str begin
            (Y, t_start) = get_state_restart(
                config,
                sim_info.restart_file,
                hash(atmos),
            )
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

    FT = Spaces.undertype(axes(Y.c))

    tracers = get_tracers(config.parsed_args)

    s = @timed_str begin
        ode_algo = ode_configuration(FT, config.parsed_args)
    end
    @info "ode_configuration: $s"

    s = @timed_str begin
        jacobian = get_jacobian(ode_algo, Y, atmos, config.parsed_args)
    end
    @info "Allocating jacobian: $s"

    s = @timed_str begin
        p = build_cache(
            Y,
            atmos,
            params,
            surface_setup,
            sim_info,
            jacobian,
            tracers.aerosol_names,
        )
    end
    @info "Allocating cache (p): $s"

    if config.parsed_args["discrete_hydrostatic_balance"]
        set_discrete_hydrostatic_balanced_state!(Y, p)
    end

    s = @timed_str begin
        callback = get_callbacks(config, sim_info, atmos, params, Y, p, t_start)
    end
    @info "get_callbacks: $s"

    # Initialize diagnostics
    if config.parsed_args["enable_diagnostics"]
        s = @timed_str begin
            scheduled_diagnostics, writers = get_diagnostics(
                config.parsed_args,
                atmos,
                Y,
                p,
                sim_info.dt,
                t_start,
            )
        end
        @info "initializing diagnostics: $s"
    else
        writers = nothing
    end

    continuous_callbacks = tuple()
    discrete_callbacks = callback

    s = @timed_str begin
        all_callbacks =
            SciMLBase.CallbackSet(continuous_callbacks, discrete_callbacks)
    end
    @info "Prepared SciMLBase.CallbackSet callbacks: $s"
    steps_cycle_non_diag = n_steps_per_cycle_per_cb(all_callbacks, sim_info.dt)
    steps_cycle = lcm(steps_cycle_non_diag)
    @info "n_steps_per_cycle_per_cb (non diagnostics): $steps_cycle_non_diag"
    @info "n_steps_per_cycle (non diagnostics): $steps_cycle"

    tspan = (t_start, sim_info.t_end)
    s = @timed_str begin
        integrator_args, integrator_kwargs = args_integrator(
            config.parsed_args,
            Y,
            p,
            tspan,
            ode_algo,
            jacobian,
            all_callbacks,
        )
    end

    s = @timed_str begin
        integrator = SciMLBase.init(integrator_args...; integrator_kwargs...)
    end
    @info "init integrator: $s"

    short_names = map(diag -> diag.variable.short_name, scheduled_diagnostics)
    if "ejac1" in short_names || "ajac1" in short_names
        # TODO: Only do this if the Jacobian needs to be saved at t = 0.
        update_jacobian_init!(integrator)
    end

    if config.parsed_args["enable_diagnostics"]
        s = @timed_str begin
            integrator = ClimaDiagnostics.IntegratorWithDiagnostics(
                integrator,
                scheduled_diagnostics,
            )
        end
        @info "Added diagnostics: $s"
    end

    reset_graceful_exit(output_dir)

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
