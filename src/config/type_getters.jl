import ClimaComms
import ClimaCore: Fields, Grids, Spaces
import Logging, NVTX

"""
    ClimaAtmosParameters(config::AtmosConfig)

Translate a configuration into a typed `ClimaAtmosParameters`.

The microphysics model, the 1-moment process options, gravity-wave, and prognostic-aerosol toggles are
resolved from the config first, so the underlying constructor only loads the parameter
sets that the run will actually use.
"""
function ClimaAtmosParameters(config::AtmosConfig)
    pa = config.parsed_args
    return ClimaAtmosParameters(
        config.toml_dict;
        microphysics_model = get_microphysics_model(pa),
        has_non_orographic_gw = get(pa, "non_orographic_gravity_wave", false) != false,
        has_orographic_gw =
        !isnothing(get(pa, "orographic_gravity_wave", nothing)),
        has_beres_source = get(pa, "nogw_beres_source", false) != false,
        has_prognostic_aerosols = !isempty(get(pa, "prognostic_aerosols", ())),
    )
end

"""
    get_atmos(config::AtmosConfig, params; setup_type = nothing)

Build the `AtmosModel` described by a configuration.

Validates the configuration with `check_case_consistency`, then assembles the model
groups (`AtmosWater`, `SCMSetup`, `AtmosRadiation`, `AtmosTurbconv`,
`AtmosGravityWave`, `AtmosSponge`, `AtmosSurface`, `AtmosNumerics`, `AtmosChem`,
`COSPModel`) plus the vertical diffusion model and the prescribed flow, which comes
either from the setup or from `prescribed_flow = "ShipwayHill2012"`. Momentum vertical
diffusion is disabled for Held-Suarez runs (`rad = "held_suarez"`). `setup_type` is the
setup object built by `get_setup_type`; the setup supplies model pieces (forcings,
surface conditions, insolation, prescribed flow) that have no config key.
"""
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
        cosp = COSPModel(config),
        aerosols = AtmosAerosols(config, params),
        vertical_diffusion,
        disable_surface_flux_tendency = pa["disable_surface_flux_tendency"],
    )
    # TODO: Should this go in the AtmosModel constructor?
    @assert !@any_reltype(atmos, (UnionAll, DataType))

    @info "AtmosModel: \n$(summary(atmos))"
    microphysics_model = atmos.water.microphysics_model
    if microphysics_model isa NonEquilibriumMicrophysics1M
        @info "Microphysics settings: $(sprint(summary_microphysics, microphysics_model))"
    end
    return atmos
end

"""
    get_numerics(parsed_args, FT)

Build the `AtmosNumerics` group from the numerics config keys.

Reads the upwinding schemes (`energy_q_tot_upwinding`, `tracer_upwinding`,
`edmfx_mse_q_tot_upwinding`, `edmfx_sgsflux_upwinding`, `edmfx_tracer_upwinding`, each
converted to a `Symbol`), the `apply_sem_quasimonotone_limiter`,
`test_dycore_consistency`, and `reproducible_restart` switches, `implicit_diffusion`
(which selects `Implicit` or `Explicit` diffusion), and the hyperdiffusion model from
`get_hyperdiffusion_model`. `vanleer_limiter` upwinding falls back to `:none` with a
warning on ClimaCore versions older than 0.14.22.
"""
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

"""
    get_state_restart(config::AtmosConfig, restart_file, atmos_model_hash)

Read the state `Y` and start time from `restart_file`, using the start date and
communications context of `config`. Thin wrapper around the typed
`get_state_restart` method in `simulation/restart.jl`.
"""
function get_state_restart(config::AtmosConfig, restart_file, atmos_model_hash)
    return get_state_restart(
        restart_file,
        parse_date(config.parsed_args["start_date"]),
        atmos_model_hash,
        config.comms_ctx,
    )
end

"""
    get_setup_type(parsed_args, thermo_params)

Build the setup object named by the `initial_condition` config key.

The setup determines the initial state and, for single-column cases, the forcings,
surface conditions, and insolation that have no config key of their own. Accepted
values map to same-named types in `Setups`:

  - Idealized profiles: `"DecayingProfile"`, `"IsothermalProfile"`,
    `"ConstantBuoyancyFrequencyProfile"`, `"DryDensityCurrentProfile"`,
    `"RisingThermalBubbleProfile"`, `"MoistAdiabaticProfileEDMFX"`, `"SimplePlume"`,
    `"PrecipitatingColumn"`, `"ShipwayHill2012"`.
  - Baroclinic waves: `"DryBaroclinicWave"`, `"MoistBaroclinicWave"`,
    `"MoistBaroclinicWaveWithEDMF"`, which read `perturb_initstate` and `deep_atmosphere`.
  - LES/SCM cases: `"Bomex"`, `"Rico"`, `"Soares"`, `"GATE_III"`, `"DYCOMS_RF01"`,
    `"DYCOMS_RF02"`, `"TRMM_LBA"`, `"Larcform1"`, `"GABLS"`, `"ISDAC"`, which read
    `prognostic_tke` (and, for ISDAC, `perturb_initstate`).
  - RCEMIP II: `"RCEMIPIIProfile_295"`, `"RCEMIPIIProfile_300"`, `"RCEMIPIIProfile_305"`.
  - File and reanalysis-driven: `"GCM"` (`external_forcing_file` plus `cfsite_number`),
    `"ARMVARANAL"` (an ARM VARANAL file, converted to the ClimaColumn schema),
    `"ForcingFromFile"` (a ClimaColumn file), `"ReanalysisTimeVarying"` (an ERA5 file for
    the site, generated when missing or stale), `"WeatherModel"` (reads
    `era5_initial_condition_dir` and `era5_ic_full_pressure`), and `"AMIPFromERA5"`.

A value that names an existing file is read as a `Setups.MoistFromFile` initial state.
Anything else raises an error.
"""
function get_setup_type(parsed_args, thermo_params)
    ic_name = parsed_args["initial_condition"]
    if ic_name == "Bomex"
        return Setups.Bomex(; prognostic_tke = parsed_args["prognostic_tke"], thermo_params)
    elseif ic_name == "Rico"
        return Setups.Rico(; prognostic_tke = parsed_args["prognostic_tke"], thermo_params)
    elseif ic_name == "GCM"
        # Read the cfsite group into steady in-memory profiles, then drive it
        # through the generic ForcingFromFile path. Defaults give an interactive
        # Monin-Obukhov surface with the file's `ts` and the constant insolation
        # carried in the data (matching the former GCMDrivenInsolation).
        data = ColumnDatasets.GCMColumnData.read_cfsite(
            parsed_args["external_forcing_file"],
            parsed_args["cfsite_number"];
            thermo_params,
        )
        return Setups.ForcingFromFile(data, parsed_args["start_date"])
    elseif ic_name == "ARMVARANAL"
        varanal_file = parsed_args["external_forcing_file"]
        isnothing(varanal_file) && error(
            "initial_condition `ARMVARANAL` requires `external_forcing_file` \
             to point at an ARM VARANAL file",
        )
        start_date = parsed_args["start_date"]
        FT = eltype(thermo_params)
        # Convert the pressure-level VARANAL file to the ClimaColumn schema, then
        # drive it through the generic ForcingFromFile path with the VARANAL
        # forcing composition (no vertical fluctuation, subsidence from `wa`).
        varanal_dir =
            get(ENV, "BUILDKITE", "") == "true" ? mktempdir() :
            dirname(varanal_file)
        canonical = ColumnDatasets.VaranalFiles.to_climacolumn(
            varanal_file;
            thermo_params,
            dir = varanal_dir,
        )
        data = ColumnDatasets.ColumnDataset(canonical)
        (; latitude, longitude) = ColumnDatasets.site_location(data)
        flux_scheme = if issubset((:hfls, :hfss), data.surface_vars)
            SurfaceConditions.MoninObukhov(;
                z0 = FT(0.05),
                ustar = FT(0.28),
                fluxes = SurfaceConditions.FileHeatFluxes(data, start_date),
            )
        else
            SurfaceConditions.MoninObukhov(; z0 = FT(0.05), ustar = FT(0.28))
        end
        return Setups.ForcingFromFile(
            data,
            start_date;
            forcing = (
                HorizontalAdvection(),
                Nudging(:ta, :hus),
                Nudging(:ua, :va),
                Subsidence(),
            ),
            flux_scheme,
            insolation = TimeVaryingInsolation(;
                start_date = parse_date(start_date),
                latitude,
                longitude,
            ),
        )
    elseif ic_name == "ReanalysisTimeVarying"
        FT = eltype(thermo_params)
        return Setups.ForcingFromFile(
            era5_dataset(parsed_args, FT),
            parsed_args["start_date"],
        )
    elseif ic_name == "ForcingFromFile"
        external_forcing_file = parsed_args["external_forcing_file"]
        isnothing(external_forcing_file) && error(
            "initial_condition `ForcingFromFile` requires `external_forcing_file` \
             to point at a column forcing file",
        )
        return Setups.ForcingFromFile(
            ColumnDatasets.ColumnDataset(external_forcing_file),
            parsed_args["start_date"],
        )
    elseif ic_name == "WeatherModel"
        return Setups.WeatherModel(
            parsed_args["start_date"],
            parsed_args["era5_initial_condition_dir"];
            use_full_pressure = parsed_args["era5_ic_full_pressure"],
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

"""
    get_topography(FT, parsed_args)

Build the surface elevation profile named by the `topography` config key: `"NoWarp"`
(flat), `"Cosine2D"`, `"Cosine3D"`, `"Agnesi"`, `"Schar"`, `"Earth"`, `"Hughes2023"`,
or `"DCMIP200"`. Any other value trips an assertion.
"""
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

"""
    get_steady_state_velocity(params, Y, topo, initial_condition, mesh_warp_type)

Compute the analytic steady-state velocity over topography on the center and face
spaces of `Y`, returned as `(; ᶜu, ᶠu)` [m/s], for comparison against the simulated
flow.

Only defined for a `ConstantBuoyancyFrequencyProfile` initial condition with `"Linear"`
mesh warping; any other combination raises an error. Called through
`steady_state_velocity_from_config`.
"""
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

"""
    jacobian_from_parsed_args(parsed_args)

Build the `JacobianAlgorithm` selected by the config keys `use_dense_jacobian`
(`AutoDenseJacobian`) and `use_auto_jacobian` (`AutoSparseJacobian`, with
`padding_bands_per_block` from `auto_jacobian_padding_bands`), falling back to
`ManualSparseJacobian`. The sparse algorithms take `approximate_solve_iters` from
`approximate_linear_solve_iters`.
"""
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

"""
    ode_configuration(::Type{FT}, args) where {FT}

Build the ODE algorithm from the config keys in `args`, forwarding `ode_algo`,
`update_jacobian_every`, `max_newton_iters_ode`, the Krylov settings
(`use_krylov_method`, `use_dynamic_krylov_rtol`, `eisenstat_walker_forcing_alpha`,
`krylov_rtol`, `jvp_step_adjustment`), and the Newton tolerance settings
(`use_newton_rtol`, `newton_rtol`) to the typed `ode_configuration` method in
`simulation/integrator.jl`.
"""
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

"""
    get_comms_context(parsed_args)

Create and initialize the `ClimaComms` context for the device named by the `device`
config key.

  - `"auto"` (or no `device` key): the device `ClimaComms` detects.
  - `"CUDADevice"`: a CUDA GPU.
  - `"CPUMultiThreaded"`: a multithreaded CPU. Note that any other value also gives a
    multithreaded CPU when Julia is started with more than one thread.
  - anything else: a single-threaded CPU.
"""
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

"""
    get_mesh_warp_type(FT, parsed_args)

Build the interior mesh warping selected by the `mesh_warp_type` config key: `"SLEVE"`
gives a `SLEVEWarp` with decay parameters `sleve_eta` and `sleve_s`, and `"Linear"`
gives a `LinearWarp`. Any other value raises an error.
"""
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

"""
    get_grid(config::AtmosConfig, params)
    get_grid(parsed_args, params, context)

Build the computational grid selected by the `config` key: `"sphere"` gives a
`SphereGrid`, `"column"` a `ColumnGrid`, `"box"` a `BoxGrid`, and `"plane"` a
`PlaneGrid`.

All grids read the vertical discretization keys `z_elem`, `z_max`, `z_stretch`, and
`dz_bottom`. Every grid except the column also reads the topography keys `topography`,
`topography_damping_factor`, `mesh_warp_type`, and `topo_smoothing`. The sphere reads
`h_elem`, `nh_poly`, `bubble`, and `deep_atmosphere`, with the planet radius taken from
`params`; the box and plane read `x_elem`/`x_max` (and, for the box, `y_elem`/`y_max`)
and are periodic in the horizontal.
"""
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

Return a callable `(Y, params) -> velocity` when the `check_steady_state` config key is
set, and `nothing` otherwise. `AtmosSimulation{FT}` invokes the callable once `Y` has
been built; it forwards to `get_steady_state_velocity`.
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

Return the tuple of species `Symbol`s that vertical water borrowing may draw from, or
`nothing` when `tracer_nonnegativity_method` is not a vertical-water-borrowing variant
or `vertical_water_borrowing_species` is unset.

The config value may be a single string or a list of strings; anything else raises an
error.
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

Bundle the callback config keys (`dt_subcol`, `dt_rad`, `dt_nogw`, `dt_ogw`,
`log_progress`, `check_nan_every`, `check_conservation`) into the `NamedTuple` expected
by the `callback_kwargs` keyword of `AtmosSimulation{FT}`.
"""
function callback_kwargs_from_config(config::AtmosConfig)
    pa = config.parsed_args
    return (;
        dt_subcol = pa["dt_subcol"],
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

Translate the diagnostic config keys into a [`DiagnosticsConfig`](@ref).

`enable_diagnostics` (master switch) and `output_default_diagnostics` (add the built-in
diagnostics) are collapsed into the `default` field, and the user-specified
`diagnostics` list passes through to `additional`; both are empty when diagnostics are
disabled. The NetCDF output shape comes from `netcdf_interpolation_num_points` and
`netcdf_output_at_levels`.
"""
function diagnostics_config_from_config(config::AtmosConfig)
    pa = config.parsed_args
    enabled = pa["enable_diagnostics"]
    return DiagnosticsConfig(;
        default = enabled && pa["output_default_diagnostics"],
        additional = enabled ? get(pa, "diagnostics", ()) : (),
        interpolation_num_points = pa["netcdf_interpolation_num_points"],
        output_at_levels = pa["netcdf_output_at_levels"],
        debug_tendency = enabled && get(pa, "debug_tendency_diagnostics", false),
    )
end

"""
    log_yaml_and_toml_manifests(config::AtmosConfig, output_dir, job_id)

Write the run's TOML parameter manifest (`<job_id>_parameters.toml`) and a YAML
snapshot of the merged configuration (`<job_id>.yml`) into `output_dir`.

Returns `nothing`. Config-driven runs only: simulations built directly from
`AtmosSimulation{FT}` do not get these manifests. `strict_params` controls whether
unused parameters are an error.
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

Build an [`AtmosSimulation`](@ref) from a configuration.

Resolves the parameters, setup, model, and grid from `config` and forwards them, along
with the time, output, restart, numerics, callback, and diagnostics keys, to the
`AtmosSimulation{FT}` keyword constructor. Config-driven runs are always verbose, and
their parameter manifest and config snapshot are written into the resolved output
directory by `log_yaml_and_toml_manifests`.

# Examples

```julia
import ClimaAtmos as CA
config = CA.AtmosConfig("config/model_configs/baroclinic_wave.yml")
simulation = CA.get_simulation(config)
CA.solve_atmos!(simulation)
```
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
        update_cache_every = pa["update_cache_every"],
        update_constrain_state_every = pa["update_constrain_state_every"],
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
