using Dates: DateTime, @dateformat_str
import Interpolations
import NCDatasets
import ClimaCore
import ClimaUtilities.OutputPathGenerator
import ClimaCore: InputOutput, Meshes, Spaces, Quadratures
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaAtmos as CA
import ClimaCore.Fields
import ClimaTimeSteppers as CTS
import Logging

import ClimaUtilities.TimeManager: ITime

import ClimaDiagnostics

function get_scale_blending_method(parsed_args)
    method_name = parsed_args["edmfx_scale_blending"]
    if method_name == "SmoothMinimum"
        return SmoothMinimumBlending()
    elseif method_name == "HardMinimum"
        return HardMinimumBlending()
    else
        error("Unknown edmfx_scale_blending method: $method_name")
    end
end

function get_numerics(parsed_args, FT)
    test_dycore =
        parsed_args["test_dycore_consistency"] ? TestDycoreConsistency() :
        nothing

    energy_upwinding = Val(Symbol(parsed_args["energy_upwinding"]))
    tracer_upwinding = Val(Symbol(parsed_args["tracer_upwinding"]))

    # Compat
    if !(pkgversion(ClimaCore) ≥ v"0.14.22") &&
       energy_upwinding == Val(:vanleer_limiter)
        energy_upwinding = Val(:none)
        @warn "energy_upwinding=vanleer_limiter is not supported for ClimaCore $(pkgversion(ClimaCore)), please upgrade. Setting energy_upwinding to :none"
    end
    if !(pkgversion(ClimaCore) ≥ v"0.14.22") &&
       tracer_upwinding == Val(:vanleer_limiter)
        tracer_upwinding = Val(:none)
        @warn "tracer_upwinding=vanleer_limiter is not supported for ClimaCore $(pkgversion(ClimaCore)), please upgrade. Setting tracer_upwinding to :none"
    end

    edmfx_upwinding = Val(Symbol(parsed_args["edmfx_upwinding"]))
    edmfx_sgsflux_upwinding =
        Val(Symbol(parsed_args["edmfx_sgsflux_upwinding"]))

    limiter = parsed_args["apply_limiter"] ? CA.QuasiMonotoneLimiter() : nothing

    # wrap each upwinding mode in a Val for dispatch
    diff_mode = parsed_args["implicit_diffusion"] ? Implicit() : Explicit()

    hyperdiff = get_hyperdiffusion_model(parsed_args, FT)

    numerics = AtmosNumerics(;
        energy_upwinding,
        tracer_upwinding,
        edmfx_upwinding,
        edmfx_sgsflux_upwinding,
        limiter,
        test_dycore_consistency = test_dycore,
        diff_mode,
        hyperdiff,
    )
    @info "numerics $(summary(numerics))"

    return numerics
end

function get_spaces(domain::PlaneDomain, params, comms_ctx)
    FT = eltype(params)
    quad = Quadratures.GLL{domain.nh_poly + 1}()
    horizontal_mesh = periodic_line_mesh(;
        x_max = domain.x_max,
        x_elem = domain.x_elem,
        periodic = domain.periodic_x,
    )
    h_space =
        make_horizontal_space(horizontal_mesh, quad, comms_ctx, domain.bubble)
    z_stretch = if domain.z_stretch
        Meshes.HyperbolicTangentStretching(domain.dz_bottom)
    else
        Meshes.Uniform()
    end
    center_space, face_space = make_hybrid_spaces(
        h_space,
        domain.z_max,
        domain.z_elem,
        z_stretch;
        deep = domain.deep_atmosphere,
    )
    return (; center_space, face_space)
end

function get_spaces(domain::BoxDomain, params, comms_ctx)
    FT = eltype(params)
    quad = Quadratures.GLL{domain.nh_poly + 1}()
    horizontal_mesh = periodic_rectangle_mesh(;
        x_max = domain.x_max,
        y_max = domain.y_max,
        x_elem = domain.x_elem,
        y_elem = domain.y_elem,
        periodic = (domain.periodic_x, domain.periodic_y),
    )
    h_space =
        make_horizontal_space(horizontal_mesh, quad, comms_ctx, domain.bubble)
    z_stretch = if domain.z_stretch
        Meshes.HyperbolicTangentStretching(domain.dz_bottom)
    else
        Meshes.Uniform()
    end
    center_space, face_space = make_hybrid_spaces(
        h_space,
        domain.z_max,
        domain.z_elem,
        z_stretch;
        deep = domain.deep_atmosphere,
    )
    return (; center_space, face_space)
end

function get_spaces(domain::ColumnDomain, params, comms_ctx)
    FT = eltype(params)
    @warn "perturb_initstate flag is ignored for single column configuration"
    Δx = FT(1) # Note: This value shouldn't matter, since we only have 1 column.
    quad = Quadratures.GL{1}()
    horizontal_mesh = periodic_rectangle_mesh(;
        x_max = Δx,
        y_max = Δx,
        x_elem = 1,
        y_elem = 1,
        periodic = (true, true),
    )
    bubble = false # bubble correction not compatible with single column configuration
    h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
    z_stretch = if domain.z_stretch
        Meshes.HyperbolicTangentStretching(domain.dz_bottom)
    else
        Meshes.Uniform()
    end
    center_space, face_space =
        make_hybrid_spaces(h_space, domain.z_max, domain.z_elem, z_stretch)
    return (; center_space, face_space)
end

function get_spaces(domain::SphereDomain, params, comms_ctx)
    FT = eltype(params)
    quad = Quadratures.GLL{domain.nh_poly + 1}()
    horizontal_mesh = cubed_sphere_mesh(; radius = domain.radius, h_elem = domain.h_elem)
    h_space = make_horizontal_space(horizontal_mesh, quad, comms_ctx, domain.bubble)
    z_stretch = if domain.z_stretch
        Meshes.HyperbolicTangentStretching(domain.dz_bottom)
    else
        Meshes.Uniform()
    end
    center_space, face_space = make_hybrid_spaces(
        h_space,
        domain.z_max,
        domain.z_elem,
        z_stretch;
        deep = domain.deep_atmosphere,
    )
    return (; center_space, face_space)
end

#=
This is the old get_spaces function. We will replace it with the new dispatch-based
functions. I'm keeping it here for now as a reference.
=#
# function get_spaces(parsed_args, params, comms_ctx)

#     FT = eltype(params)
#     z_elem = Int(parsed_args["z_elem"])
#     z_max = FT(parsed_args["z_max"])
#     dz_bottom = FT(parsed_args["dz_bottom"])
#     bubble = parsed_args["bubble"]
#     deep = parsed_args["deep_atmosphere"]

#     h_elem = parsed_args["h_elem"]
#     radius = CAP.planet_radius(params)
#     center_space, face_space = if parsed_args["config"] == "sphere"
#         nh_poly = parsed_args["nh_poly"]
#         quad = Quadratures.GLL{nh_poly + 1}()
#         horizontal_mesh = cubed_sphere_mesh(; radius, h_elem)
#         h_space =
#             make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
#         z_stretch = if parsed_args["z_stretch"]
#             Meshes.HyperbolicTangentStretching(dz_bottom)
#         else
#             Meshes.Uniform()
#         end
#         make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; deep, parsed_args)
#     elseif parsed_args["config"] == "column" # single column
#         @warn "perturb_initstate flag is ignored for single column configuration"
#         FT = eltype(params)
#         Δx = FT(1) # Note: This value shouldn't matter, since we only have 1 column.
#         quad = Quadratures.GL{1}()
#         horizontal_mesh = periodic_rectangle_mesh(;
#             x_max = Δx,
#             y_max = Δx,
#             x_elem = 1,
#             y_elem = 1,
#         )
#         if bubble
#             @warn "Bubble correction not compatible with single column configuration. It will be switched off."
#             bubble = false
#         end
#         h_space =
#             make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
#         z_stretch = if parsed_args["z_stretch"]
#             Meshes.HyperbolicTangentStretching(dz_bottom)
#         else
#             Meshes.Uniform()
#         end
#         make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args)
#     elseif parsed_args["config"] == "box"
#         FT = eltype(params)
#         nh_poly = parsed_args["nh_poly"]
#         quad = Quadratures.GLL{nh_poly + 1}()
#         x_elem = Int(parsed_args["x_elem"])
#         x_max = FT(parsed_args["x_max"])
#         y_elem = Int(parsed_args["y_elem"])
#         y_max = FT(parsed_args["y_max"])
#         horizontal_mesh = periodic_rectangle_mesh(;
#             x_max = x_max,
#             y_max = y_max,
#             x_elem = x_elem,
#             y_elem = y_elem,
#         )
#         h_space =
#             make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
#         z_stretch = if parsed_args["z_stretch"]
#             Meshes.HyperbolicTangentStretching(dz_bottom)
#         else
#             Meshes.Uniform()
#         end
#         make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args, deep)
#     elseif parsed_args["config"] == "plane"
#         FT = eltype(params)
#         nh_poly = parsed_args["nh_poly"]
#         quad = Quadratures.GLL{nh_poly + 1}()
#         x_elem = Int(parsed_args["x_elem"])
#         x_max = FT(parsed_args["x_max"])
#         horizontal_mesh =
#             periodic_line_mesh(; x_max = x_max, x_elem = x_elem)
#         h_space =
#             make_horizontal_space(horizontal_mesh, quad, comms_ctx, bubble)
#         z_stretch = if parsed_args["z_stretch"]
#             Meshes.HyperbolicTangentStretching(dz_bottom)
#         else
#             Meshes.Uniform()
#         end
#         make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args, deep)
#     end
#     ncols = Fields.ncolumns(center_space)
#     ndofs_total = ncols * z_elem
#     hspace = Spaces.horizontal_space(center_space)
#     quad_style = Spaces.quadrature_style(hspace)
#     Nq = Quadratures.degrees_of_freedom(quad_style)

#     @info "Resolution stats: " Nq h_elem z_elem ncols ndofs_total
#     return (;
#         center_space,
#         face_space,
#         horizontal_mesh,
#         quad,
#         z_max,
#         z_elem,
#         z_stretch,
#     )
# end

function get_spaces_restart(Y)
    center_space = axes(Y.c)
    face_space = axes(Y.f)
    return (; center_space, face_space)
end

function get_state_restart(config::AtmosConfig, restart_file, atmos_model_hash)
    (; parsed_args, comms_ctx) = config
    sim_info = get_sim_info(config)

    @assert !isnothing(restart_file)
    reader = InputOutput.HDF5Reader(restart_file, comms_ctx)
    Y = InputOutput.read_field(reader, "Y")
    # TODO: Do not use InputOutput.HDF5 directly
    t_start = InputOutput.HDF5.read_attribute(reader.file, "time")
    t_start =
        parsed_args["use_itime"] ? ITime(t_start; epoch = sim_info.start_date) :
        t_start
    if "atmos_model_hash" in keys(InputOutput.HDF5.attrs(reader.file))
        atmos_model_hash_in_restart =
            InputOutput.HDF5.read_attribute(reader.file, "atmos_model_hash")
        if atmos_model_hash_in_restart != atmos_model_hash
            @warn "Restart file $(restart_file) was constructed with a different AtmosModel"
        end
    end
    return (Y, t_start)
end

function get_initial_condition(parsed_args, atmos)
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
        "ConstantBuoyancyFrequencyProfile",
        "IsothermalProfile",
        "DryDensityCurrentProfile",
        "RisingThermalBubbleProfile",
        "PrecipitatingColumn",
    ]
        return getproperty(ICs, Symbol(parsed_args["initial_condition"]))()
    elseif isfile(parsed_args["initial_condition"])
        return ICs.MoistFromFile(parsed_args["initial_condition"])
    elseif parsed_args["initial_condition"] == "GCM"
        @assert parsed_args["prognostic_tke"] == true
        return ICs.GCMDriven(
            parsed_args["external_forcing_file"],
            parsed_args["cfsite_number"],
        )
    elseif parsed_args["initial_condition"] == "ReanalysisTimeVarying"
        return ICs.external_tv_initial_condition(
            atmos.external_forcing.external_forcing_file,
            parsed_args["start_date"],
        )
    elseif parsed_args["initial_condition"] == "WeatherModel"
        return ICs.WeatherModel(parsed_args["start_date"])
    else
        error(
            "Unknown `initial_condition`: $(parsed_args["initial_condition"])",
        )
    end
end

function get_steady_state_velocity(params, Y, parsed_args)
    parsed_args["check_steady_state"] || return nothing
    parsed_args["initial_condition"] == "ConstantBuoyancyFrequencyProfile" &&
        parsed_args["mesh_warp_type"] == "Linear" ||
        error("The steady-state velocity can currently be computed only for a \
               ConstantBuoyancyFrequencyProfile with Linear mesh warping")
    topography = parsed_args["topography"]
    steady_state_velocity = if topography == "NoWarp"
        steady_state_velocity_no_warp
    elseif topography == "Cosine2D"
        steady_state_velocity_cosine_2d
    elseif topography == "Cosine3D"
        steady_state_velocity_cosine_3d
    elseif topography == "Agnesi"
        steady_state_velocity_agnesi
    elseif topography == "Schar"
        steady_state_velocity_schar
    else
        error("The steady-state velocity for $topography topography cannot \
               be computed analytically")
    end
    top_level = Spaces.nlevels(axes(Y.c)) + Fields.half
    z_top = Fields.level(Fields.coordinate_field(Y.f).z, top_level)

    # TODO: This can be very expensive! It should be moved to a separate CI job.
    @info "Approximating steady-state velocity"
    s = @timed_str begin
        ᶜu = steady_state_velocity.(params, Fields.coordinate_field(Y.c), z_top)
        ᶠu =
            steady_state_velocity.(params, Fields.coordinate_field(Y.f), z_top)
    end
    @info "Steady-state velocity approximation completed: $s"
    return (; ᶜu, ᶠu)
end

function get_surface_setup(parsed_args)
    parsed_args["surface_setup"] == "GCM" && return SurfaceConditions.GCMDriven(
        parsed_args["external_forcing_file"],
        parsed_args["cfsite_number"],
    )

    return getproperty(SurfaceConditions, Symbol(parsed_args["surface_setup"]))()
end

function get_jacobian(ode_algo, Y, atmos, parsed_args)
    ode_algo isa Union{CTS.IMEXAlgorithm, CTS.RosenbrockAlgorithm} ||
        return nothing
    jacobian_algorithm = if parsed_args["use_dense_jacobian"]
        AutoDenseJacobian()
    else
        manual_jacobian_algorithm = ManualSparseJacobian(
            DerivativeFlag(has_topography(axes(Y.c))),
            DerivativeFlag(atmos.diff_mode),
            DerivativeFlag(atmos.sgs_adv_mode),
            DerivativeFlag(atmos.sgs_entr_detr_mode),
            DerivativeFlag(atmos.sgs_mf_mode),
            DerivativeFlag(atmos.sgs_nh_pressure_mode),
            parsed_args["approximate_linear_solve_iters"],
        )
        parsed_args["use_auto_jacobian"] ?
        AutoSparseJacobian(
            manual_jacobian_algorithm,
            parsed_args["auto_jacobian_padding_bands"],
        ) : manual_jacobian_algorithm
    end
    @info "Jacobian algorithm: $(summary_string(jacobian_algorithm))"
    verbose = parsed_args["debug_jacobian"]
    return Jacobian(jacobian_algorithm, Y, atmos; verbose)
end

#=
    ode_configuration(Y, parsed_args)

Returns the ode algorithm
=#
function ode_configuration(::Type{FT}, parsed_args) where {FT}
    ode_name = parsed_args["ode_algo"]
    ode_algo_name = getproperty(CTS, Symbol(ode_name))
    @info "Using ODE config: `$ode_algo_name`"
    return if ode_algo_name <: CTS.RosenbrockAlgorithmName
        if parsed_args["update_jacobian_every"] != "solve"
            @warn "Rosenbrock algorithms in ClimaTimeSteppers currently only \
                   support `update_jacobian_every` = \"solve\""
        end
        CTS.RosenbrockAlgorithm(CTS.tableau(ode_algo_name()))
    elseif ode_algo_name <: CTS.ERKAlgorithmName
        CTS.ExplicitAlgorithm(ode_algo_name())
    else
        @assert ode_algo_name <: CTS.IMEXARKAlgorithmName
        newtons_method = CTS.NewtonsMethod(;
            max_iters = parsed_args["max_newton_iters_ode"],
            update_j = if parsed_args["update_jacobian_every"] == "dt"
                CTS.UpdateEvery(CTS.NewTimeStep)
            elseif parsed_args["update_jacobian_every"] == "stage"
                CTS.UpdateEvery(CTS.NewNewtonSolve)
            elseif parsed_args["update_jacobian_every"] == "solve"
                CTS.UpdateEvery(CTS.NewNewtonIteration)
            else
                error("Unknown value of `update_jacobian_every`: \
                       $(parsed_args["update_jacobian_every"])")
            end,
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
    (; comms_ctx, parsed_args) = config
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
        context = comms_ctx,
        style = output_dir_style,
    )
    if parsed_args["log_to_file"]
        @info "Logging to $output_dir/output.log"
        logger = ClimaComms.FileLogger(comms_ctx, output_dir)
        Logging.global_logger(logger)
    end
    @info "Running on $(nameof(typeof(ClimaComms.device(comms_ctx))))"
    if comms_ctx isa ClimaComms.SingletonCommsContext
        @info "Setting up single-process ClimaAtmos run"
    else
        @info "Setting up distributed ClimaAtmos run" nprocs =
            ClimaComms.nprocs(comms_ctx)
    end

    isnothing(restart_file) ||
        @info "Restarting simulation from file $restart_file"
    epoch = parse_date(parsed_args["start_date"])
    t_start_int = time_to_seconds(parsed_args["t_start"])
    if !isnothing(restart_file) && t_start_int != 0
        @warn "Non zero `t_start` passed with a restarting simulation. The provided `t_start` will be ignored."
    end
    if parsed_args["use_itime"]
        dt = ITime(time_to_seconds(parsed_args["dt"]))
        t_start = ITime(time_to_seconds(parsed_args["t_start"]), epoch = epoch)
        t_end = ITime(time_to_seconds(parsed_args["t_end"]), epoch = epoch)
        # ITime(0) is added for backward compatibility (since t_start used to always be 0)
        (dt, t_start, t_end, _) = promote(dt, t_start, t_end, ITime(0))
    else
        dt = FT(time_to_seconds(parsed_args["dt"]))
        t_start = FT(time_to_seconds(parsed_args["t_start"]))
        t_end = FT(time_to_seconds(parsed_args["t_end"]))
    end
    sim = (;
        output_dir,
        restart = !isnothing(restart_file),
        restart_file,
        job_id,
        dt = dt,
        start_date = epoch,
        t_start = t_start,
        t_end = t_end,
    )
    n_steps = floor(Int, (sim.t_end - sim.t_start) / sim.dt)
    @info(
        "Time info:",
        dt = parsed_args["dt"],
        t_start = parsed_args["t_start"],
        t_end = parsed_args["t_end"],
        floor_n_steps = n_steps,
    )

    return sim
end

function args_integrator(parsed_args, Y, p, tspan, ode_algo, callback)
    (; atmos, dt) = p
    s = @timed_str begin
        T_imp! = SciMLBase.ODEFunction(
            implicit_tendency!;
            jac_prototype = get_jacobian(ode_algo, Y, atmos, parsed_args),
            Wfact = update_jacobian!,
        )
        tendency_function = CTS.ClimaODEFunction(;
            T_exp_T_lim! = remaining_tendency!,
            T_imp!,
            lim! = limiters_func!,
            dss!,
            cache! = set_precomputed_quantities!,
            cache_imp! = set_implicit_precomputed_quantities!,
        )
    end
    @info "Define ode function: $s"
    problem = SciMLBase.ODEProblem(tendency_function, Y, tspan, p)
    t_begin, t_end, _ = promote(tspan[1], tspan[2], p.dt)
    # Save solution to integrator.sol at the beginning and end
    saveat = [t_begin, t_end]
    args = (problem, ode_algo)
    allow_custom_kwargs = (; kwargshandle = CTS.DiffEqBase.KeywordArgSilent)
    kwargs =
        (; saveat, callback, dt, adjustfinal = true, allow_custom_kwargs...)
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

function get_simulation(config::AtmosConfig)
    comms_ctx = get_comms_context(config.parsed_args)
    params = ClimaAtmosParameters(config)
    FT = eltype(params)

    # Get the domain
    if config.parsed_args["config"] == "sphere"
        domain = SphereDomain(
            FT;
            radius = CAP.planet_radius(params),
            h_elem = config.parsed_args["h_elem"],
            nh_poly = config.parsed_args["nh_poly"],
            z_elem = config.parsed_args["z_elem"],
            z_max = config.parsed_args["z_max"],
            z_stretch = config.parsed_args["z_stretch"],
            dz_bottom = config.parsed_args["dz_bottom"],
            bubble = config.parsed_args["bubble"],
            deep_atmosphere = config.parsed_args["deep_atmosphere"],
        )
    elseif config.parsed_args["config"] == "column"
        domain = ColumnDomain(
            FT;
            z_elem = config.parsed_args["z_elem"],
            z_max = config.parsed_args["z_max"],
            z_stretch = config.parsed_args["z_stretch"],
            dz_bottom = config.parsed_args["dz_bottom"],
        )
    elseif config.parsed_args["config"] == "box"
        domain = BoxDomain(
            FT;
            x_elem = config.parsed_args["x_elem"],
            x_max = config.parsed_args["x_max"],
            y_elem = config.parsed_args["y_elem"],
            y_max = config.parsed_args["y_max"],
            z_elem = config.parsed_args["z_elem"],
            z_max = config.parsed_args["z_max"],
            nh_poly = config.parsed_args["nh_poly"],
            z_stretch = config.parsed_args["z_stretch"],
            dz_bottom = config.parsed_args["dz_bottom"],
            bubble = config.parsed_args["bubble"],
            deep_atmosphere = config.parsed_args["deep_atmosphere"],
            periodic_x = true,
            periodic_y = true,
        )
    elseif config.parsed_args["config"] == "plane"
        domain = PlaneDomain(
            FT;
            x_elem = config.parsed_args["x_elem"],
            x_max = config.parsed_args["x_max"],
            z_elem = config.parsed_args["z_elem"],
            z_max = config.parsed_args["z_max"],
            nh_poly = config.parsed_args["nh_poly"],
            z_stretch = config.parsed_args["z_stretch"],
            dz_bottom = config.parsed_args["dz_bottom"],
            bubble = config.parsed_args["bubble"],
            deep_atmosphere = config.parsed_args["deep_atmosphere"],
            periodic_x = true,
        )
    end

    # Get the AtmosModel
    atmos = get_atmos(config, params)
    
    # Get other simulation parameters
    initial_condition = get_initial_condition(config.parsed_args, atmos)
    surface_setup = get_surface_setup(config.parsed_args)
    sim_info = get_sim_info(config)
    ode_algo_type = getproperty(CTS, Symbol(config.parsed_args["ode_algo"]))
    callbacks = get_callbacks(config, sim_info, atmos, params, nothing, nothing)
    diagnostics = if config.parsed_args["enable_diagnostics"]
        get_diagnostics_from_config(config.parsed_args, atmos)
    else
        ()
    end
    
    # Create the simulation
    simulation = AtmosSimulation(;
        model = atmos,
        domain,
        initial_condition,
        params,
        comms_ctx;
        dt = sim_info.dt,
        t_span = (sim_info.t_start, sim_info.t_end),
        ode_algo_type = ode_algo_type,
        surface_setup = surface_setup,
        job_id = sim_info.job_id,
        output_dir = sim_info.output_dir,
        restart_file = sim_info.restart_file,
        start_date = sim_info.start_date,
        tracers = get_tracers(config.parsed_args),
        callbacks = callbacks,
        diagnostics = diagnostics,
    )
    return simulation
end

function AtmosSimulation(;
    model::AtmosModel,
    domain::AbstractDomain,
    initial_condition::AbstractInitialCondition,
    params,
    comms_ctx;
    dt,
    t_span,
    ode_algo_type::Type{<:CTS.AbstractAlgorithm},
    surface_setup = nothing,
    job_id = "atmos_sim",
    output_dir = "output",
    restart_file = nothing,
    start_date = DateTime(0),
    tracers = nothing,
    callbacks = (),
    diagnostics = (),
)
    FT = eltype(params)
    t_start, t_end = t_span

    sim_info = (;
        job_id,
        output_dir,
        restart_file,
        start_date,
        dt,
        t_start,
        t_end,
    )

    if !isnothing(restart_file)
        (Y, t_start) =
            get_state_restart(restart_file, hash(atmos), comms_ctx)
        spaces = get_spaces_restart(Y)
        sim_info = merge(sim_info, (; t_start))
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

    p = build_cache(
        Y,
        model,
        params,
        surface_setup,
        sim_info,
        isnothing(tracers) ? String[] : tracers.aerosol_names,
        nothing, # steady_state_velocity
    )

    ode_algo = ode_configuration(ode_algo_type, Y, model)
    callback_set = SciMLBase.CallbackSet(callbacks...)

    integrator_args, integrator_kwargs = args_integrator(
        Y,
        p,
        (t_start, t_end),
        ode_algo,
        callback_set,
    )
    integrator = SciMLBase.init(integrator_args...; integrator_kwargs...)

    # Initialize diagnostics
    if !isempty(diagnostics)
        scheduled_diagnostics, writers, _ = get_diagnostics(
            diagnostics,
            model,
            Y,
            p,
            sim_info,
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


# Compatibility with old get_integrator
function get_integrator(config::AtmosConfig)
    Base.depwarn(
        "get_integrator is deprecated, use get_simulation instead",
        :get_integrator,
    )
    return get_simulation(config).integrator
end
