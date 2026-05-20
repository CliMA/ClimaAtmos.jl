using Dates: DateTime, @dateformat_str
import Interpolations
import NCDatasets
import ClimaCore
import ClimaUtilities.OutputPathGenerator
import ClimaCore: InputOutput, Meshes, Spaces, Quadratures
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaAtmos as CA
import ClimaCore: Fields, Grids
import ClimaTimeSteppers as CTS
import Logging

import ClimaUtilities.TimeManager: ITime

import ClimaDiagnostics

"""
    convert_time_args(dt, t_start, t_end, start_date)

Convert dt, t_start, and t_end to ITime.
"""
function convert_time_args(dt, t_start, t_end, start_date)
    to_seconds(t) = t isa AbstractString ? time_to_seconds(t) : Float64(t)
    dt = ITime(to_seconds(dt))
    t_start = ITime(to_seconds(t_start), epoch = start_date)
    t_end = ITime(to_seconds(t_end), epoch = start_date)
    # ITime(0) is added for backward compatibility (since t_start used to always be 0)
    (dt, t_start, t_end, _) = promote(dt, t_start, t_end, ITime(0))
    return (dt, t_start, t_end)
end

"""
    ClimaAtmosParameters(config::AtmosConfig)

Translate the YAML config into a typed `ClimaAtmosParameters`. Pre-computes
the microphysics model and gravity-wave toggles from `parsed_args` so the
underlying constructor only loads the parameter sets that will actually be
used.
"""
function ClimaAtmosParameters(config::AtmosConfig)
    pa = config.parsed_args
    return ClimaAtmosParameters(
        config.toml_dict;
        microphysics_model = get_microphysics_model(pa),
        has_non_orographic_gw = get(pa, "non_orographic_gravity_wave", false) != false,
        has_orographic_gw =
        !isnothing(get(pa, "orographic_gravity_wave", nothing)),
    )
end

function get_atmos(config::AtmosConfig, params; setup_type = nothing)
    pa = config.parsed_args
    FT = eltype(config)
    check_case_consistency(pa)

    radiation = AtmosRadiation(config, FT; setup_type)

    # disable_momentum_vertical_diffusion ⇐ HeldSuarezForcing radiation; threads
    # into VerticalDiffusion construction.
    disable_momentum_vertical_diffusion =
        radiation.radiation_mode isa HeldSuarezForcing

    implicit_diffusion = pa["implicit_diffusion"]
    @assert implicit_diffusion in (true, false)

    vertical_diffusion = get_vertical_diffusion_model(
        disable_momentum_vertical_diffusion, pa, params, FT,
    )

    prescribed_flow = if !isnothing(setup_type)
        Setups.prescribed_flow_model(setup_type, FT)
    else
        nothing
    end
    if isnothing(prescribed_flow) && pa["prescribed_flow"] == "ShipwayHill2012"
        prescribed_flow = ShipwayHill2012VelocityProfile{FT}()
    end

    atmos = AtmosModel(;
        water = AtmosWater(config, params, FT),
        scm_setup = SCMSetup(config, FT; setup_type),
        prescribed_flow,
        radiation,
        turbconv = AtmosTurbconv(config, params, FT),
        gravity_wave = AtmosGravityWave(config, params, FT),
        sponge = AtmosSponge(config, params),
        surface = AtmosSurface(config, params, FT; setup_type),
        numerics = AtmosNumerics(config, FT),
        vertical_diffusion,
        disable_surface_flux_tendency = pa["disable_surface_flux_tendency"],
    )
    # TODO: Should this go in the AtmosModel constructor?
    @assert !@any_reltype(atmos, (UnionAll, DataType))

    @info "AtmosModel: \n$(summary(atmos))"
    return atmos
end

function get_numerics(parsed_args, FT)
    test_dycore_consistency =
        parsed_args["test_dycore_consistency"] ? TestDycoreConsistency() :
        nothing
    reproducible_restart =
        parsed_args["reproducible_restart"] ? ReproducibleRestart() :
        nothing

    energy_q_tot_upwinding = Symbol(parsed_args["energy_q_tot_upwinding"])
    tracer_upwinding = Symbol(parsed_args["tracer_upwinding"])

    # Compat
    if !(pkgversion(ClimaCore) ≥ v"0.14.22") &&
       energy_q_tot_upwinding == :vanleer_limiter
        energy_q_tot_upwinding = :none
        @warn "energy_q_tot_upwinding=vanleer_limiter is not supported for ClimaCore $(pkgversion(ClimaCore)), please upgrade. Setting energy_q_tot_upwinding to :none"
    end
    if !(pkgversion(ClimaCore) ≥ v"0.14.22") &&
       tracer_upwinding == :vanleer_limiter
        tracer_upwinding = :none
        @warn "tracer_upwinding=vanleer_limiter is not supported for ClimaCore $(pkgversion(ClimaCore)), please upgrade. Setting tracer_upwinding to :none"
    end

    edmfx_mse_q_tot_upwinding = Symbol(parsed_args["edmfx_mse_q_tot_upwinding"])
    edmfx_sgsflux_upwinding = Symbol(parsed_args["edmfx_sgsflux_upwinding"])
    edmfx_tracer_upwinding = Symbol(parsed_args["edmfx_tracer_upwinding"])

    limiter =
        parsed_args["apply_sem_quasimonotone_limiter"] ? CA.QuasiMonotoneLimiter() : nothing

    diff_mode = parsed_args["implicit_diffusion"] ? Implicit() : Explicit()

    hyperdiff = get_hyperdiffusion_model(parsed_args, FT)

    numerics = AtmosNumerics(;
        energy_q_tot_upwinding,
        tracer_upwinding,
        edmfx_mse_q_tot_upwinding,
        edmfx_sgsflux_upwinding,
        edmfx_tracer_upwinding,
        limiter,
        test_dycore_consistency,
        reproducible_restart,
        diff_mode,
        hyperdiff,
    )
    @info "numerics $(summary(numerics))"

    return numerics
end

"""
    get_spaces(grid)

Create center and face spaces from a ClimaCore grid.
"""
function get_spaces(grid)
    if grid isa Grids.ExtrudedFiniteDifferenceGrid
        center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
        face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)
    elseif grid isa Grids.FiniteDifferenceGrid
        center_space = Spaces.CenterFiniteDifferenceSpace(grid)
        face_space = Spaces.FaceFiniteDifferenceSpace(grid)
    else
        error(
            """Unsupported grid type: $(typeof(grid)). Expected \
            ExtrudedFiniteDifferenceGrid or FiniteDifferenceGrid""",
        )
    end
    return (; center_space, face_space)
end

function get_state_restart(config::AtmosConfig, restart_file, atmos_model_hash)
    return get_state_restart(
        restart_file,
        parse_date(config.parsed_args["start_date"]),
        atmos_model_hash,
        config.comms_ctx,
    )
end

function get_state_restart(
    restart_file,
    start_date,
    atmos_model_hash,
    comms_ctx,
)
    @assert !isnothing(restart_file)
    reader = InputOutput.HDF5Reader(restart_file, comms_ctx)
    Y = InputOutput.read_field(reader, "Y")
    # TODO: Do not use InputOutput.HDF5 directly
    t_start = InputOutput.HDF5.read_attribute(reader.file, "time")
    t_start = ITime(t_start; epoch = start_date)
    if "atmos_model_hash" in keys(InputOutput.HDF5.attrs(reader.file))
        atmos_model_hash_in_restart =
            InputOutput.HDF5.read_attribute(reader.file, "atmos_model_hash")
        if atmos_model_hash_in_restart != atmos_model_hash
            @warn "Restart file $(restart_file) was constructed with a different AtmosModel"
        end
    end
    return (Y, t_start)
end

"""
    handle_restart(restart_file, t_start_original, start_date, model, context)

Handle restart file loading with validation and logging.

Validates that t_start is zero when restarting, loads state from restart file,
logs restart information, and returns the state, t_start, and spaces.

Returns:
- `Y`: State loaded from restart file
- `t_start`: Time from restart file (ITime)
- `spaces`: Named tuple with center_space and face_space extracted from Y
"""
function handle_restart(
    restart_file,
    t_start_original,
    start_date,
    model,
    context,
)
    # Validate t_start before restart (matches get_simulation behavior)
    t_start_seconds =
        t_start_original isa AbstractString ?
        time_to_seconds(t_start_original) : Float64(t_start_original)
    if t_start_seconds != 0
        @warn "Non zero `t_start` passed with a restarting simulation. The provided `t_start` will be ignored."
    end

    (Y, t_start) = get_state_restart(
        restart_file, start_date, hash(model), context,
    )

    @info "Restarting simulation from file" restart_file restart_time = string(t_start)

    spaces = (; center_space = axes(Y.c), face_space = axes(Y.f))

    return Y, t_start, spaces
end


function get_setup_type(parsed_args, thermo_params)
    ic_name = parsed_args["initial_condition"]
    if ic_name == "Bomex"
        return Setups.Bomex(; prognostic_tke = parsed_args["prognostic_tke"], thermo_params)
    elseif ic_name == "Rico"
        return Setups.Rico(; prognostic_tke = parsed_args["prognostic_tke"], thermo_params)
    elseif ic_name == "GCM"
        return Setups.GCMDriven(
            parsed_args["external_forcing_file"],
            parsed_args["cfsite_number"],
        )
    elseif ic_name == "ReanalysisTimeVarying"
        FT = eltype(thermo_params)
        external_forcing_file =
            get_external_daily_forcing_file_path(parsed_args)
        if !isfile(external_forcing_file) ||
           !check_daily_forcing_times(external_forcing_file, parsed_args)
            @info "External forcing file $(external_forcing_file) does not exist or does not cover the expected time range. Generating it now."
            generate_multiday_era5_external_forcing_file(
                parsed_args,
                external_forcing_file,
                FT,
                input_data_dir = joinpath(
                    @clima_artifact("era5_hourly_atmos_raw"),
                    "daily",
                ),
            )
        end
        return Setups.InterpolatedColumnProfile(
            external_forcing_file,
            parsed_args["start_date"],
        )
    elseif ic_name == "WeatherModel"
        return Setups.WeatherModel(
            parsed_args["start_date"],
            parsed_args["era5_initial_condition_dir"],
        )
    elseif ic_name == "AMIPFromERA5"
        return Setups.AMIPFromERA5(parsed_args["start_date"])
    elseif ic_name == "DecayingProfile"
        return Setups.DecayingProfile(;
            perturb = parsed_args["perturb_initstate"],
            thermo_params,
        )
    elseif ic_name in
           ("DryBaroclinicWave", "MoistBaroclinicWave", "MoistBaroclinicWaveWithEDMF")
        return getproperty(Setups, Symbol(ic_name))(;
            perturb = parsed_args["perturb_initstate"],
            deep_atmosphere = parsed_args["deep_atmosphere"],
        )
    elseif ic_name in ("Soares", "GATE_III", "DYCOMS_RF01", "DYCOMS_RF02", "TRMM_LBA")
        return getproperty(Setups, Symbol(ic_name))(;
            prognostic_tke = parsed_args["prognostic_tke"],
            thermo_params,
        )
    elseif ic_name == "GABLS"
        return Setups.GABLS(;
            prognostic_tke = parsed_args["prognostic_tke"],
            thermo_params,
        )
    elseif ic_name == "ISDAC"
        return Setups.ISDAC(;
            prognostic_tke = parsed_args["prognostic_tke"],
            perturb = parsed_args["perturb_initstate"],
            thermo_params,
        )
    elseif ic_name in ("IsothermalProfile", "ConstantBuoyancyFrequencyProfile",
        "DryDensityCurrentProfile", "RisingThermalBubbleProfile")
        return getproperty(Setups, Symbol(ic_name))()
    elseif ic_name == "MoistAdiabaticProfileEDMFX"
        return Setups.MoistAdiabaticProfileEDMFX(;
            perturb = parsed_args["perturb_initstate"],
        )
    elseif ic_name == "SimplePlume"
        return Setups.SimplePlume(;
            prognostic_tke = parsed_args["prognostic_tke"],
        )
    elseif ic_name in ("RCEMIPIIProfile_295", "RCEMIPIIProfile_300", "RCEMIPIIProfile_305")
        return getproperty(Setups, Symbol(ic_name))()
    elseif ic_name == "PrecipitatingColumn"
        return Setups.PrecipitatingColumn(; thermo_params)
    elseif ic_name == "ShipwayHill2012"
        return Setups.ShipwayHill2012(; thermo_params)
    elseif isfile(ic_name)
        return Setups.MoistFromFile(ic_name)
    end
    error("Unknown initial_condition: $ic_name")
end

function get_topography(FT, parsed_args)
    topo_str = parsed_args["topography"]
    topo_types = Dict("NoWarp" => NoTopography(),
        "Cosine2D" => CosineTopography{2, FT}(),
        "Cosine3D" => CosineTopography{3, FT}(),
        "Agnesi" => AgnesiTopography{FT}(),
        "Schar" => ScharTopography{FT}(),
        "Earth" => EarthTopography(),
        "Hughes2023" => Hughes2023Topography(),
        "DCMIP200" => DCMIP200Topography(),
    )

    @assert topo_str in keys(topo_types)
    return topo_types[topo_str]
end

function get_steady_state_velocity(params, Y, topo, initial_condition, mesh_warp_type)
    initial_condition == "ConstantBuoyancyFrequencyProfile" &&
        mesh_warp_type == "Linear" ||
        error("The steady-state velocity can currently be computed only for a \
               ConstantBuoyancyFrequencyProfile with Linear mesh warping")
    top_level = Spaces.nlevels(axes(Y.c)) + Fields.half
    z_top = Fields.level(Fields.coordinate_field(Y.f).z, top_level)

    @info "Approximating steady-state velocity"
    s = @timed_str begin
        ᶜu = steady_state_velocity.(topo, params, Fields.coordinate_field(Y.c), z_top)
        ᶠu =
            steady_state_velocity.(topo, params, Fields.coordinate_field(Y.f), z_top)
    end
    return (; ᶜu, ᶠu)
end

function get_surface_setup(parsed_args; setup_type = nothing)
    if !isnothing(setup_type)
        return function (params)
            result = Setups.surface_condition(setup_type, params)
            if !isnothing(result)
                return result
            end
            return _config_surface_setup(parsed_args)(params)
        end
    end
    return _config_surface_setup(parsed_args)
end

function _config_surface_setup(parsed_args)
    return getproperty(SurfaceConditions, Symbol(parsed_args["surface_setup"]))()
end

# Translate YAML config keys into a user-facing JacobianAlgorithm stub.
function jacobian_from_parsed_args(parsed_args)
    approximate_solve_iters = parsed_args["approximate_linear_solve_iters"]
    if parsed_args["use_dense_jacobian"]
        return AutoDenseJacobian()
    elseif parsed_args["use_auto_jacobian"]
        return AutoSparseJacobian(;
            approximate_solve_iters,
            padding_bands_per_block = parsed_args["auto_jacobian_padding_bands"],
        )
    else
        return ManualSparseJacobian(; approximate_solve_iters)
    end
end

function get_jacobian(
    ode_algo, Y, atmos, jacobian::JacobianAlgorithm, debug_jacobian,
)
    ode_algo isa Union{CTS.IMEXAlgorithm, CTS.RosenbrockAlgorithm} ||
        return nothing
    @info "Jacobian algorithm: $(summary_string(jacobian))"
    jac = Jacobian(jacobian, Y, atmos; verbose = debug_jacobian)
    if hasproperty(jac.cache, :derivative_flags)
        flags_str = join(
            ("$k = $(typeof(v).name.name)" for (k, v) in pairs(jac.cache.derivative_flags)),
            ", ",
        )
        @info "Jacobian derivative flags: $flags_str"
    end
    return jac
end

function ode_configuration(::Type{FT}, args) where {FT}
    return ode_configuration(
        FT,
        args["ode_algo"],
        args["update_jacobian_every"],
        args["max_newton_iters_ode"],
        args["use_krylov_method"],
        args["use_dynamic_krylov_rtol"],
        args["eisenstat_walker_forcing_alpha"],
        args["krylov_rtol"],
        args["use_newton_rtol"],
        args["newton_rtol"],
        args["jvp_step_adjustment"],
    )
end


function ode_configuration(::Type{FT}, ode_name, update_jacobian_every,
    max_newton_iters_ode, use_krylov_method, use_dynamic_krylov_rtol,
    eisenstat_walker_forcing_alpha, krylov_rtol, use_newton_rtol, newton_rtol,
    jvp_step_adjustment,
) where {FT}
    ode_algo_name = getproperty(CTS, Symbol(ode_name))
    @info "Using ODE config: `$ode_algo_name`"
    return if ode_algo_name <: CTS.RosenbrockAlgorithmName
        if update_jacobian_every != "solve"
            @warn "Rosenbrock algorithms in ClimaTimeSteppers currently only \
                   support `update_jacobian_every` = \"solve\""
        end
        CTS.RosenbrockAlgorithm(CTS.tableau(ode_algo_name()))
    elseif ode_algo_name <: CTS.ERKAlgorithmName
        CTS.ExplicitAlgorithm(ode_algo_name())
    else
        @assert ode_algo_name <: CTS.IMEXARKAlgorithmName
        newtons_method = CTS.NewtonsMethod(;
            max_iters = max_newton_iters_ode,
            update_j = if update_jacobian_every == "dt"
                CTS.UpdateEvery(CTS.NewTimeStep)
            elseif update_jacobian_every == "stage"
                CTS.UpdateEvery(CTS.NewNewtonSolve)
            elseif update_jacobian_every == "solve"
                CTS.UpdateEvery(CTS.NewNewtonIteration)
            else
                error("Unknown value of `update_jacobian_every`: \
                        $(update_jacobian_every)")
            end,
            krylov_method = if use_krylov_method
                CTS.KrylovMethod(;
                    jacobian_free_jvp = CTS.ForwardDiffJVP(;
                        step_adjustment = FT(
                            jvp_step_adjustment,
                        ),
                    ),
                    forcing_term = if use_dynamic_krylov_rtol
                        α = FT(eisenstat_walker_forcing_alpha)
                        CTS.EisenstatWalkerForcing(; α)
                    else
                        CTS.ConstantForcing(FT(krylov_rtol))
                    end,
                )
            else
                nothing
            end,
            convergence_checker = if use_newton_rtol
                norm_condition = CTS.MaximumRelativeError(
                    FT(newton_rtol),
                )
                CTS.ConvergenceChecker(; norm_condition)
            else
                nothing
            end,
        )
        CTS.IMEXAlgorithm(ode_algo_name(), newtons_method)
    end
end

auto_detect_restart_file(::OutputPathGenerator.OutputPathGeneratorStyle, _) =
    error("auto_detect_restart_file works only with ActiveLink")

"""
    auto_detect_restart_file(::ActiveLinkStyle, base_output_dir)

Return the most recent restart file in the directory structure in `base_output_dir`, if any.

`auto_detect_restart_file` scans the content of `base_output_dir` matching the expected
names for output folders generated by `ActiveLinkStyle` (e.g., `output_0000`, `output_0001`).
It iterates through these folders sorted by number in descending order and returns the latest
restart file (latest measured by the time in the file name) from the first folder that contains
restart files matching the pattern `dayDDDD.SSSSS.hdf5`. This ensures that empty or incomplete
higher-numbered folders are skipped in favor of folders that actually contain restart files.
If no folder with restart files is found, return `nothing`: this means that the simulation
cannot be automatically restarted.
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

    # Sort folders by number (descending) and find the first with restart files
    for output_folder in sort(existing_outputs, rev = true)
        folder_path = joinpath(base_output_dir, output_folder)
        possible_restart_files =
            filter(f -> occursin(restart_file_rx, f), readdir(folder_path))
        if !isempty(possible_restart_files)
            previous_folder = folder_path
            restart_file_name = last(CA.sort_files_by_time(possible_restart_files))
            restart_file = joinpath(previous_folder, restart_file_name)
            @assert isfile(restart_file) "Restart file does not exist"
            return restart_file
        end
    end

    # No folder with restart files found
    @warn "No restart files found in any output folder in $(base_output_dir)"
    return nothing
end


import ClimaUtilities.OutputPathGenerator

"""
    setup_output_dir(job_id, output_dir, output_dir_style, detect_restart_file, restart_file, comms_ctx)

Unified function for setting up output directories and detecting restart files.
Used by both AtmosSimulation constructor and get_simulation.

Returns a named tuple with:
- `output_dir`: The final output directory path
- `restart_file`: The restart file path (if any)
"""
function setup_output_dir(
    job_id,
    output_dir,
    output_dir_style,
    detect_restart_file,
    restart_file,
    comms_ctx,
)
    # Set up base output directory
    default_output = haskey(ENV, "CI") ? job_id : joinpath("output", job_id)
    base_output_dir = isnothing(output_dir) ? default_output : output_dir

    # Validate and get output directory style
    allowed_dir_styles = Dict(
        "activelink" => OutputPathGenerator.ActiveLinkStyle(),
        "removepreexisting" => OutputPathGenerator.RemovePreexistingStyle(),
    )

    haskey(allowed_dir_styles, lowercase(output_dir_style)) ||
        error("output_dir_style $(output_dir_style) not available")

    output_dir_style_obj = allowed_dir_styles[lowercase(output_dir_style)]

    final_restart_file = if detect_restart_file && isnothing(restart_file)
        auto_detect_restart_file(output_dir_style_obj, base_output_dir)
    else
        restart_file
    end

    # Generate the actual output directory
    output_dir = OutputPathGenerator.generate_output_path(
        base_output_dir;
        context = comms_ctx,
        style = output_dir_style_obj,
    )

    return output_dir, final_restart_file
end

"""
    fully_explicit_tendency!

Experimental timestepping mode where all implicit tendencies are treated explicitly.
"""
function fully_explicit_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    (; temp_Yₜ_imp) = p.scratch
    implicit_tendency!(temp_Yₜ_imp, Y, p, t)
    remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ .+= temp_Yₜ_imp
end

function args_integrator(Y, p, tspan, ode_algo, callback,
    jacobian, debug_jacobian, prescribed_flow, dt_integrator,
)
    (; atmos) = p
    s = @timed_str begin
        if isnothing(prescribed_flow)

            # This is the default case
            T_exp_T_lim! = remaining_tendency!
            T_imp! = CTS.ODEFunction(
                implicit_tendency!;
                jac_prototype = get_jacobian(
                    ode_algo, Y, atmos, jacobian, debug_jacobian,
                ),
                Wfact = update_jacobian!,
            )
            cache_imp! = set_implicit_precomputed_quantities!
        else
            # `prescribed_flow` is an experimental case where the flow is prescribed,
            # so implicit tendencies are treated explicitly to avoid treatment of sound waves
            T_exp_T_lim! = fully_explicit_tendency!
            T_imp! = nothing
            cache_imp! = nothing
        end
        tendency_function = CTS.ClimaODEFunction(;
            T_exp_T_lim!, T_imp!,
            cache! = set_precomputed_quantities!, cache_imp!,
            lim! = limiters_func!, dss! = constrain_state!,  # TODO: Rename ClimaODEFunction kwarg to `constrain_state!`
            initialize_imp! = initialize_implicit_stage_problem!,
        )
    end
    problem = CTS.ODEProblem(tendency_function, Y, tspan, p)
    # Promote to ensure t_begin, t_end, and dt_integrator all have the same type
    # (dt_integrator is ITime, p.dt is FT)
    t_begin, t_end, dt = promote(tspan[1], tspan[2], dt_integrator)
    # Save solution to integrator.sol at the beginning and end
    saveat = [t_begin, t_end]
    args = (problem, ode_algo)
    kwargs = (; saveat, callback, dt)
    return (args, kwargs)
end

import ClimaComms, Logging, NVTX
function get_comms_context(parsed_args)
    device =
        if !haskey(parsed_args, "device") || parsed_args["device"] === "auto"
            ClimaComms.device()
        elseif parsed_args["device"] == "CUDADevice"
            ClimaComms.CUDADevice()
        elseif parsed_args["device"] == "CPUMultiThreaded" ||
               Threads.nthreads() > 1
            ClimaComms.CPUMultiThreaded()
        else
            ClimaComms.CPUSingleThreaded()
        end
    comms_ctx = ClimaComms.context(device)
    ClimaComms.init(comms_ctx)

    if NVTX.isactive() && get(ENV, "BUILDKITE", "") == "true"
        # makes output on buildkite a bit nicer
        if ClimaComms.iamroot(comms_ctx)
            atexit() do
                println("--- Saving profiler information")
            end
        end
    end

    return comms_ctx
end

function get_mesh_warp_type(FT, parsed_args)
    warp_type_str = parsed_args["mesh_warp_type"]
    if warp_type_str == "SLEVE"
        return SLEVEWarp{FT}(
            eta = parsed_args["sleve_eta"],
            s = parsed_args["sleve_s"],
        )
    elseif warp_type_str == "Linear"
        return LinearWarp()
    else
        error(
            "Unknown mesh warp type string: $warp_type_str. Supported types are 'SLEVE' and 'Linear'",
        )
    end
end

get_grid(config::AtmosConfig, params) =
    get_grid(config.parsed_args, params, config.comms_ctx)

function get_grid(parsed_args, params, context)
    FT = eltype(params)
    config = parsed_args["config"]

    # Common vertical discretization parameters
    kwargs = (
        z_elem = parsed_args["z_elem"],
        z_max = parsed_args["z_max"],
        z_stretch = parsed_args["z_stretch"],
        dz_bottom = parsed_args["dz_bottom"],
    )

    # Add topography parameters for non-column grids
    if config != "column"
        kwargs = (
            kwargs...,
            topography = get_topography(FT, parsed_args),
            topography_damping_factor = parsed_args["topography_damping_factor"],
            mesh_warp_type = get_mesh_warp_type(FT, parsed_args),
            topo_smoothing = parsed_args["topo_smoothing"],
        )
    end

    # Grid-specific construction
    if config == "sphere"
        SphereGrid(
            FT;
            context,
            radius = CAP.planet_radius(params),
            h_elem = parsed_args["h_elem"],
            nh_poly = parsed_args["nh_poly"],
            bubble = parsed_args["bubble"],
            deep_atmosphere = parsed_args["deep_atmosphere"],
            kwargs...,
        )
    elseif config == "column"
        # Use a minimal BoxGrid for column simulations (x_elem=1, y_elem=1, nh_poly=1)
        # This provides a 2x2 horizontal grid that behaves like a single column
        BoxGrid(
            FT;
            context,
            x_elem = 1,
            x_max = FT(1e5),  # 100 km
            y_elem = 1,
            y_max = FT(1e5),
            nh_poly = 1,
            bubble = false,
            periodic_x = true,
            periodic_y = true,
            z_elem = parsed_args["z_elem"],
            z_max = parsed_args["z_max"],
            z_stretch = parsed_args["z_stretch"],
            dz_bottom = parsed_args["dz_bottom"],
            kwargs...,
        )
    elseif config == "box"
        BoxGrid(
            FT;
            context,
            x_elem = parsed_args["x_elem"],
            x_max = parsed_args["x_max"],
            y_elem = parsed_args["y_elem"],
            y_max = parsed_args["y_max"],
            nh_poly = parsed_args["nh_poly"],
            bubble = parsed_args["bubble"],
            periodic_x = true,
            periodic_y = true,
            kwargs...,
        )
    elseif config == "plane"
        PlaneGrid(
            FT;
            context,
            x_elem = parsed_args["x_elem"],
            x_max = parsed_args["x_max"],
            nh_poly = parsed_args["nh_poly"],
            periodic_x = true,
            kwargs...,
        )
    end
end

"""
    steady_state_velocity_from_config(config::AtmosConfig, params)

Return a callable `(Y, params) -> velocity` when `check_steady_state` is set,
else `nothing`. `AtmosSimulation{FT}` invokes the callable after building `Y`.
"""
function steady_state_velocity_from_config(config::AtmosConfig, params)
    config.parsed_args["check_steady_state"] || return nothing
    parsed_args = config.parsed_args
    FT = eltype(params)
    topo = get_topography(FT, Dict("topography" => parsed_args["topography"]))
    initial_condition = parsed_args["initial_condition"]
    mesh_warp_type = parsed_args["mesh_warp_type"]
    return steady_state_velocity(Y, params) =
        get_steady_state_velocity(params, Y, topo, initial_condition, mesh_warp_type)
end

"""
    vertical_water_borrowing_species_from_config(config::AtmosConfig)

Returns the parsed VWB-species tuple, or `nothing` if not configured.
Mirrors the legacy YAML driver's parsing logic.
"""
function vertical_water_borrowing_species_from_config(config::AtmosConfig)
    pa = config.parsed_args
    method = pa["tracer_nonnegativity_method"]
    is_vwb =
        !isnothing(method) && (
            method == "vertical_water_borrowing" ||
            startswith(method, "vertical_water_borrowing_")
        )
    is_vwb || return nothing
    species = get(pa, "vertical_water_borrowing_species", nothing)
    isnothing(species) && return nothing
    if species isa Vector
        return tuple(Symbol.(species)...)
    elseif species isa String
        return (Symbol(species),)
    else
        error(
            "vertical_water_borrowing_species must be a string or list of strings, got $(typeof(species))",
        )
    end
end

"""
    callback_kwargs_from_config(config::AtmosConfig)

Bundle YAML callback knobs into the NamedTuple expected by
`AtmosSimulation{FT}`'s `callback_kwargs` slot.
"""
function callback_kwargs_from_config(config::AtmosConfig)
    pa = config.parsed_args
    return (;
        dt_rad = pa["dt_rad"],
        dt_nogw = pa["dt_nogw"],
        dt_ogw = pa["dt_ogw"],
        log_progress = pa["log_progress"],
        check_nan_every = pa["check_nan_every"],
        check_conservation = pa["check_conservation"],
    )
end

"""
    diagnostics_config_from_config(config::AtmosConfig)

Translate the YAML diagnostic toggles into a `DiagnosticsConfig`. Collapses
`enable_diagnostics` (master switch) and `output_default_diagnostics` (add
built-ins) into `DiagnosticsConfig.default`. The user-specified diagnostic
list passes through to `DiagnosticsConfig.additional`.
"""
function diagnostics_config_from_config(config::AtmosConfig)
    pa = config.parsed_args
    enabled = pa["enable_diagnostics"]
    return DiagnosticsConfig(;
        default = enabled && pa["output_default_diagnostics"],
        additional = enabled ? get(pa, "diagnostics", ()) : (),
        interpolation_num_points = pa["netcdf_interpolation_num_points"],
        output_at_levels = pa["netcdf_output_at_levels"],
    )
end

"""
    log_yaml_and_toml_manifests(config::AtmosConfig, output_dir, job_id)

Side-effect: write the run's TOML parameter manifest and a YAML snapshot of
the merged config into `output_dir`. YAML-driver-only — programmatic users
don't get these manifests.
"""
function log_yaml_and_toml_manifests(config::AtmosConfig, output_dir, job_id)
    output_toml_file = joinpath(output_dir, "$(job_id)_parameters.toml")
    CP.log_parameter_information(
        config.toml_dict,
        output_toml_file;
        strict = config.parsed_args["strict_params"],
    )
    output_args = copy(config.parsed_args)
    output_args["toml"] = [abspath(output_toml_file)]
    YAML.write_file(joinpath(output_dir, "$(job_id).yml"), output_args)
    return nothing
end

"""
    get_simulation(config::AtmosConfig)

Build an `AtmosSimulation` from a YAML-driven `AtmosConfig`. Translates the
parsed YAML into the kwargs that `AtmosSimulation{FT}(; ...)` accepts and
forwards. After the simulation is built, writes the YAML-driver-only TOML
parameter manifest and YAML config snapshot into the resolved `output_dir`.
"""
function get_simulation(config::AtmosConfig)
    pa = config.parsed_args
    FT = eltype(config)
    job_id = config.job_id
    params = ClimaAtmosParameters(config)
    setup = get_setup_type(pa, CAP.thermodynamics_params(params))
    model = get_atmos(config, params; setup_type = setup)
    grid = get_grid(pa, params, config.comms_ctx)
    tracers = get_tracers(pa)

    log_context(config.comms_ctx)

    sim = AtmosSimulation{FT}(;
        model,
        params,
        context = config.comms_ctx,
        grid,
        setup,
        surface_setup = get_surface_setup(pa; setup_type = setup),
        steady_state_velocity = steady_state_velocity_from_config(config, params),
        dt = pa["dt"],
        start_date = parse_date(pa["start_date"]),
        t_start = pa["t_start"],
        t_end = pa["t_end"],
        ode_config = ode_configuration(FT, pa),
        jacobian = jacobian_from_parsed_args(pa),
        debug_jacobian = pa["debug_jacobian"],
        aerosol_names = tracers.aerosol_names,
        time_varying_trace_gases = tracers.time_varying_trace_gas_names,
        vertical_water_borrowing_species =
        vertical_water_borrowing_species_from_config(config),
        job_id,
        output_dir = pa["output_dir"],
        output_dir_style = pa["output_dir_style"],
        restart_file = pa["restart_file"],
        detect_restart_file = pa["detect_restart_file"],
        callback_kwargs = callback_kwargs_from_config(config),
        diagnostics = diagnostics_config_from_config(config),
        checkpoint_frequency = pa["dt_save_state_to_disk"],
        log_to_file = pa["log_to_file"],
    )

    @info "Simulation info" job_id = sim.job_id output_dir = sim.output_dir

    log_yaml_and_toml_manifests(config, sim.output_dir, sim.job_id)

    return sim
end

"""
    extract_diagnostic_periods(diagnostics)

Extract accumulation periods from diagnostics that have reduction functions.
Returns a Set of Period objects.
"""
function extract_diagnostic_periods(diagnostics)
    periods_reductions = Set()
    for diag in diagnostics
        isa_reduction = !isnothing(diag.reduction_time_func)
        isa_reduction || continue

        if diag.output_schedule_func isa CAD.EveryDtSchedule
            period = Dates.Second(diag.output_schedule_func.dt)
        elseif diag.output_schedule_func isa CAD.EveryCalendarDtSchedule
            period = diag.output_schedule_func.dt
        else
            continue
        end

        push!(periods_reductions, period)
    end
    return periods_reductions
end
