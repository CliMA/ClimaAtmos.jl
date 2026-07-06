import ClimaComms
import ClimaCore: Fields, Grids, Spaces
import Logging, NVTX

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
        microphysics_1m_options = get_microphysics_1m_options(pa, config.toml_dict),
        has_non_orographic_gw = get(pa, "non_orographic_gravity_wave", false) != false,
        has_orographic_gw =
        !isnothing(get(pa, "orographic_gravity_wave", nothing)),
    )
end

function get_atmos(config::AtmosConfig, params; setup_type = nothing)
    pa = config.parsed_args
    FT = eltype(config)
    check_case_consistency(pa)

    disable_momentum_vertical_diffusion = pa["rad"] == "held_suarez"

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
        radiation = AtmosRadiation(config, FT; setup_type),
        turbconv = AtmosTurbconv(config, params, FT),
        gravity_wave = AtmosGravityWave(config, params, FT),
        sponge = AtmosSponge(config, params),
        surface = AtmosSurface(config, params, FT; setup_type),
        numerics = AtmosNumerics(config, FT),
        chemistry = AtmosChem(config),
        vertical_diffusion,
        disable_surface_flux_tendency = pa["disable_surface_flux_tendency"],
    )
    # TODO: Should this go in the AtmosModel constructor?
    @assert !@any_reltype(atmos, (UnionAll, DataType))

    @info "AtmosModel: \n$(summary(atmos))"
    if !isnothing(params.microphysics_1m_params)
        microphysics_model = atmos.water.microphysics_model
        options = params.microphysics_1m_params.options
        @info "Microphysics settings: $(sprint(summary_microphysics, microphysics_model, options))"
    end
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
        parsed_args["apply_sem_quasimonotone_limiter"] ? QuasiMonotoneLimiter() : nothing

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

function get_state_restart(config::AtmosConfig, restart_file, atmos_model_hash)
    return get_state_restart(
        restart_file,
        parse_date(config.parsed_args["start_date"]),
        atmos_model_hash,
        config.comms_ctx,
    )
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
    elseif ic_name == "ARMVARANAL"
        return Setups.ARMVARANAL(
            parsed_args["external_forcing_file"];
            thermo_params,
            start_date = parsed_args["start_date"],
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
    elseif ic_name in
           ("Soares", "GATE_III", "DYCOMS_RF01", "DYCOMS_RF02", "TRMM_LBA", "Larcform1")
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

    @timed_log true "Approximating steady-state velocity" begin
        ᶜu = steady_state_velocity.(topo, params, Fields.coordinate_field(Y.c), z_top)
        ᶠu =
            steady_state_velocity.(topo, params, Fields.coordinate_field(Y.f), z_top)
    end
    return (; ᶜu, ᶠu)
end

# Translate YAML config keys into a user-facing JacobianAlgorithm stub.
function jacobian_from_parsed_args(parsed_args)
    approximate_solve_iters = parsed_args["approximate_linear_solve_iters"]
    if parsed_args["use_dense_jacobian"]
        return AutoDenseJacobian()
    elseif parsed_args["use_auto_jacobian"]
        scaling_str = parsed_args["auto_jacobian_scaling"]
        scaling_str in (nothing, "static") || error(
            "Invalid auto_jacobian_scaling: $scaling_str (`~` or `static`)",
        )
        deprecated_manual_bands = parsed_args["auto_jacobian_manual_bands"]
        padding_mode = if isnothing(deprecated_manual_bands)
            Symbol(parsed_args["auto_jacobian_padding_mode"])
        else
            @warn "auto_jacobian_manual_bands is deprecated; use \
                   auto_jacobian_padding_mode = \"manual_rules\" (for `true`) \
                   or \"constant\" (for `false`) instead"
            deprecated_manual_bands ? :manual_rules : :constant
        end
        return AutoSparseJacobian(;
            approximate_solve_iters,
            padding_bands_per_block = parsed_args["auto_jacobian_padding_bands"],
            seed_scaling = isnothing(scaling_str) ? nothing : Symbol(scaling_str),
            padding_mode,
            runtime_remeasure = parsed_args["auto_jacobian_runtime_remeasure"],
            remeasure_switch_step = parsed_args["auto_jacobian_remeasure_switch_step"],
            cross_field_threshold = parsed_args["auto_jacobian_measured_cross_field_threshold"],
            support_rtol = parsed_args["auto_jacobian_measured_support_rtol"],
        )
    else
        return ManualSparseJacobian(; approximate_solve_iters)
    end
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
        ColumnGrid(FT; context, kwargs...)
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

    log_context(config.comms_ctx)

    sim = AtmosSimulation{FT}(;
        model,
        params,
        context = config.comms_ctx,
        grid,
        setup,
        steady_state_velocity = steady_state_velocity_from_config(config, params),
        dt = pa["dt"],
        start_date = parse_date(pa["start_date"]),
        t_start = pa["t_start"],
        t_end = pa["t_end"],
        ode_config = ode_configuration(FT, pa),
        jacobian = jacobian_from_parsed_args(pa),
        debug_jacobian = pa["debug_jacobian"],
        aerosol_names = Tuple(pa["prescribed_aerosols"]),
        time_varying_trace_gases = Tuple(pa["time_varying_trace_gases"]),
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
        verbose = true,  # Config-based runs are always verbose
    )

    @info "Simulation info" job_id = sim.job_id output_dir = sim.output_dir

    log_yaml_and_toml_manifests(config, sim.output_dir, sim.job_id)

    return sim
end
