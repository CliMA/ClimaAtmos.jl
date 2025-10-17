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

function get_atmos(config::AtmosConfig, params)
    (; turbconv_params) = params
    (; parsed_args) = config
    FT = eltype(config)
    check_case_consistency(parsed_args)
    moisture_model = get_moisture_model(parsed_args)
    microphysics_model = get_microphysics_model(parsed_args)
    cloud_model = get_cloud_model(parsed_args)

    if moisture_model isa DryModel
        @warn "Running simulations without any moisture present."
        @assert microphysics_model isa NoPrecipitation
    end
    if moisture_model isa EquilMoistModel
        @warn "Running simulations with equilibrium thermodynamics assumptions."
        @assert microphysics_model isa
                Union{NoPrecipitation, Microphysics0Moment}
    end
    if moisture_model isa NonEquilMoistModel
        @assert microphysics_model isa Union{
            NoPrecipitation, Microphysics1Moment,
            Microphysics2Moment, Microphysics2MomentP3,
        }
    end
    if microphysics_model isa NoPrecipitation
        @warn "Running simulations without any precipitation formation."
    end

    implicit_noneq_cloud_formation =
        parsed_args["implicit_noneq_cloud_formation"]
    @assert implicit_noneq_cloud_formation in (true, false)

    ozone = get_ozone(parsed_args)
    radiation_mode = get_radiation_mode(parsed_args, FT)
    forcing_type = get_forcing_type(parsed_args)
    call_cloud_diagnostics_per_stage =
        get_call_cloud_diagnostics_per_stage(parsed_args)

    if isnothing(ozone) && radiation_mode isa RRTMGPI.AbstractRRTMGPMode
        @warn "prescribe_ozone is set to nothing with an RRTMGP model. Resetting to IdealizedOzone. This behavior will stop being supported in some future release"
        ozone = IdealizedOzone()
    end
    co2 = get_co2(parsed_args)
    with_rrtmgp = radiation_mode isa RRTMGPI.AbstractRRTMGPMode
    if with_rrtmgp && isnothing(co2)
        @warn (
            "co2_model set to nothing with an RRTMGP model. Resetting to FixedCO2"
        )
        co2 = FixedCO2()
    end
    (!isnothing(co2) && !with_rrtmgp) &&
        @warn ("$(co2) does nothing if RRTMGP is not used")

    # HeldSuarezForcing can be set via radiation_mode or legacy forcing option for now
    final_radiation_mode =
        forcing_type isa HeldSuarezForcing ? forcing_type : radiation_mode
    # Note: when disable_momentum_vertical_diffusion is true, the surface flux tendency
    # for momentum is not applied.
    disable_momentum_vertical_diffusion =
        final_radiation_mode isa HeldSuarezForcing

    advection_test = parsed_args["advection_test"]
    @assert advection_test in (false, true)

    implicit_diffusion = parsed_args["implicit_diffusion"]
    @assert implicit_diffusion in (true, false)

    implicit_sgs_advection = parsed_args["implicit_sgs_advection"]
    @assert implicit_sgs_advection in (true, false)

    implicit_sgs_entr_detr = parsed_args["implicit_sgs_entr_detr"]
    @assert implicit_sgs_entr_detr in (true, false)

    implicit_sgs_nh_pressure = parsed_args["implicit_sgs_nh_pressure"]
    @assert implicit_sgs_nh_pressure in (true, false)

    implicit_sgs_vertdiff = parsed_args["implicit_sgs_vertdiff"]
    @assert implicit_sgs_vertdiff in (true, false)

    implicit_sgs_mass_flux = parsed_args["implicit_sgs_mass_flux"]
    @assert implicit_sgs_mass_flux in (true, false)

    edmfx_model = EDMFXModel(;
        entr_model = get_entrainment_model(parsed_args),
        detr_model = get_detrainment_model(parsed_args),
        sgs_mass_flux = Val(parsed_args["edmfx_sgs_mass_flux"]),
        sgs_diffusive_flux = Val(parsed_args["edmfx_sgs_diffusive_flux"]),
        nh_pressure = Val(parsed_args["edmfx_nh_pressure"]),
        vertical_diffusion = Val(parsed_args["edmfx_vertical_diffusion"]),
        filter = Val(parsed_args["edmfx_filter"]),
        scale_blending_method = get_scale_blending_method(parsed_args),
    )

    vertical_diffusion = get_vertical_diffusion_model(
        disable_momentum_vertical_diffusion,
        parsed_args,
        params,
        FT,
    )

    atmos = AtmosModel(;
        # AtmosWater - Moisture, Precipitation & Clouds
        moisture_model,
        microphysics_model,
        cloud_model,
        noneq_cloud_formation_mode = implicit_noneq_cloud_formation ?
                                     Implicit() : Explicit(),
        call_cloud_diagnostics_per_stage,

        # SCMSetup - Single-Column Model components
        subsidence = get_subsidence_model(parsed_args, radiation_mode, FT),
        external_forcing = get_external_forcing_model(parsed_args, FT),
        ls_adv = get_large_scale_advection_model(parsed_args, FT),
        advection_test,
        scm_coriolis = get_scm_coriolis(parsed_args, FT),

        # AtmosRadiation
        radiation_mode = final_radiation_mode,
        ozone,
        co2,
        insolation = get_insolation_form(parsed_args),

        # AtmosTurbconv - Turbulence & Convection
        edmfx_model,
        turbconv_model = get_turbconv_model(FT, parsed_args, turbconv_params),
        sgs_adv_mode = implicit_sgs_advection ? Implicit() : Explicit(),
        sgs_entr_detr_mode = implicit_sgs_entr_detr ? Implicit() : Explicit(),
        sgs_nh_pressure_mode = implicit_sgs_nh_pressure ? Implicit() :
                               Explicit(),
        sgs_vertdiff_mode = implicit_sgs_vertdiff ? Implicit() : Explicit(),
        sgs_mf_mode = implicit_sgs_mass_flux ? Implicit() : Explicit(),
        smagorinsky_lilly = get_smagorinsky_lilly_model(parsed_args),
        amd_les = get_amd_les_model(parsed_args, FT),

        # AtmosGravityWave
        non_orographic_gravity_wave = get_non_orographic_gravity_wave_model(
            parsed_args,
            FT,
        ),
        orographic_gravity_wave = get_orographic_gravity_wave_model(
            parsed_args,
            FT,
        ),

        # AtmosSponge
        viscous_sponge = get_viscous_sponge_model(parsed_args, params, FT),
        rayleigh_sponge = get_rayleigh_sponge_model(parsed_args, params, FT),

        # AtmosSurface
        sfc_temperature = get_sfc_temperature_form(parsed_args),
        surface_model = get_surface_model(parsed_args),
        surface_albedo = get_surface_albedo_model(parsed_args, params, FT),

        # Top-level options (not grouped)
        vertical_diffusion,
        numerics = get_numerics(parsed_args, FT),
        disable_surface_flux_tendency = parsed_args["disable_surface_flux_tendency"],
    )
    # TODO: Should this go in the AtmosModel constructor?
    @assert !@any_reltype(atmos, (UnionAll, DataType))

    @info "AtmosModel: \n$(summary(atmos))"
    return atmos
end

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

    energy_q_tot_upwinding = Val(Symbol(parsed_args["energy_q_tot_upwinding"]))
    tracer_upwinding = Val(Symbol(parsed_args["tracer_upwinding"]))

    # Compat
    if !(pkgversion(ClimaCore) ≥ v"0.14.22") &&
       energy_q_tot_upwinding == Val(:vanleer_limiter)
        energy_q_tot_upwinding = Val(:none)
        @warn "energy_q_tot_upwinding=vanleer_limiter is not supported for ClimaCore $(pkgversion(ClimaCore)), please upgrade. Setting energy_q_tot_upwinding to :none"
    end
    if !(pkgversion(ClimaCore) ≥ v"0.14.22") &&
       tracer_upwinding == Val(:vanleer_limiter)
        tracer_upwinding = Val(:none)
        @warn "tracer_upwinding=vanleer_limiter is not supported for ClimaCore $(pkgversion(ClimaCore)), please upgrade. Setting tracer_upwinding to :none"
    end

    edmfx_mse_q_tot_upwinding = Val(Symbol(parsed_args["edmfx_mse_q_tot_upwinding"]))
    edmfx_sgsflux_upwinding =
        Val(Symbol(parsed_args["edmfx_sgsflux_upwinding"]))
    edmfx_tracer_upwinding =
        Val(Symbol(parsed_args["edmfx_tracer_upwinding"]))

    limiter = parsed_args["apply_limiter"] ? CA.QuasiMonotoneLimiter() : nothing

    # wrap each upwinding mode in a Val for dispatch
    diff_mode = parsed_args["implicit_diffusion"] ? Implicit() : Explicit()

    hyperdiff = get_hyperdiffusion_model(parsed_args, FT)

    numerics = AtmosNumerics(;
        energy_q_tot_upwinding,
        tracer_upwinding,
        edmfx_mse_q_tot_upwinding,
        edmfx_sgsflux_upwinding,
        edmfx_tracer_upwinding,
        limiter,
        test_dycore_consistency = test_dycore,
        diff_mode,
        hyperdiff,
    )
    @info "numerics $(summary(numerics))"

    return numerics
end

function get_spaces(parsed_args, params, comms_ctx)

    FT = eltype(params)
    z_elem = Int(parsed_args["z_elem"])
    z_max = FT(parsed_args["z_max"])
    dz_bottom = FT(parsed_args["dz_bottom"])
    bubble = parsed_args["bubble"]
    deep = parsed_args["deep_atmosphere"]

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
        make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; deep, parsed_args)
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
        make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args, deep)
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
        make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args, deep)
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
        return ICs.WeatherModel(
            parsed_args["start_date"],
            parsed_args["era5_initial_condition_dir"],
        )
    else
        error(
            "Unknown `initial_condition`: $(parsed_args["initial_condition"])",
        )
    end
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

function get_steady_state_velocity(params, Y, parsed_args)
    parsed_args["check_steady_state"] || return nothing
    parsed_args["initial_condition"] == "ConstantBuoyancyFrequencyProfile" &&
        parsed_args["mesh_warp_type"] == "Linear" ||
        error("The steady-state velocity can currently be computed only for a \
               ConstantBuoyancyFrequencyProfile with Linear mesh warping")
    topo = get_topography(eltype(params), parsed_args)
    top_level = Spaces.nlevels(axes(Y.c)) + Fields.half
    z_top = Fields.level(Fields.coordinate_field(Y.f).z, top_level)

    @info "Approximating steady-state velocity"
    s = @timed_str begin
        ᶜu = steady_state_velocity.(topo, params, Fields.coordinate_field(Y.c), z_top)
        ᶠu =
            steady_state_velocity.(topo, params, Fields.coordinate_field(Y.f), z_top)
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
            DerivativeFlag(atmos.sgs_vertdiff_mode),
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
    sim_info = get_sim_info(config)
    params = ClimaAtmosParameters(config)
    atmos = get_atmos(config, params)
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
            # Fix the t_start in sim_info with the one from the restart
            sim_info = merge(sim_info, (; t_start))
        end
        @info "Allocating Y: $s"
    else
        spaces = get_spaces(config.parsed_args, params, config.comms_ctx)
    end
    initial_condition = get_initial_condition(config.parsed_args, atmos)
    surface_setup = get_surface_setup(config.parsed_args)
    if !sim_info.restart
        s = @timed_str begin
            Y = ICs.atmos_state(
                initial_condition(params),
                atmos,
                spaces.center_space,
                spaces.face_space,
            )
        end
        @info "Allocating Y: $s"

        # In instances where we wish to interpolate existing datasets, e.g.
        # NetCDF files containing spatially varying thermodynamic properties,
        # this call to `overwrite_initial_conditions` overwrites the variables
        # in `Y` (specific to `initial_condition`) with those computed using the
        # `SpaceVaryingInputs` tool.
        CA.InitialConditions.overwrite_initial_conditions!(
            initial_condition,
            Y,
            params.thermodynamics_params,
        )
    end

    tracers = get_tracers(config.parsed_args)

    steady_state_velocity =
        get_steady_state_velocity(params, Y, config.parsed_args)

    s = @timed_str begin
        p = build_cache(
            Y,
            atmos,
            params,
            surface_setup,
            sim_info,
            tracers.aerosol_names,
            steady_state_velocity,
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
        callback = get_callbacks(config, sim_info, atmos, params, Y, p)
    end
    @info "get_callbacks: $s"

    # Initialize diagnostics
    if config.parsed_args["enable_diagnostics"]
        s = @timed_str begin
            scheduled_diagnostics, writers, periods_reductions =
                get_diagnostics(
                    config.parsed_args,
                    atmos,
                    Y,
                    p,
                    sim_info,
                    output_dir,
                )
        end
        @info "initializing diagnostics: $s"

        # Check for consistency between diagnostics and checkpoints
        checkpoint_frequency = checkpoint_frequency_from_parsed_args(
            config.parsed_args["dt_save_state_to_disk"],
        )

        if checkpoint_frequency != Inf
            if any(
                x -> !CA.isdivisible(checkpoint_frequency, x),
                periods_reductions,
            )
                accum_str =
                    join(CA.promote_period.(collect(periods_reductions)), ", ")
                checkpt_str = CA.promote_period(checkpoint_frequency)
                @warn "The checkpointing frequency (dt_save_state_to_disk = $checkpt_str) should be an integer multiple of all diagnostics accumulation periods ($accum_str) so simulations can be safely restarted from any checkpoint"
            end
        end
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

    tspan = (sim_info.t_start, sim_info.t_end)
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
