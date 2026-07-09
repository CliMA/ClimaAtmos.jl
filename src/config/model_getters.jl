using Flux
import JLD2
import CloudMicrophysics as CM

function get_microphysics_model(parsed_args, params = nothing)
    model_name = parsed_args["microphysics_model"]
    if model_name == "dry"
        DryModel()
    elseif model_name == "0M"
        EquilibriumMicrophysics0M()
    elseif model_name == "1M"
        n_substeps = parsed_args["microphysics_n_substeps"]
        n_substeps_quad = parsed_args["microphysics_n_substeps_quadrature"]
        NonEquilibriumMicrophysics1M(; n_substeps, n_substeps_quad)
    elseif model_name == "2M"
        NonEquilibriumMicrophysics2M()
    elseif model_name == "2MP3"
        NonEquilibriumMicrophysics2MP3()
    else
        error(
            """Unknown microphysics_model `$model_name`. Expected: "dry", "0M", "1M", "2M", or "2MP3".""",
        )
    end
end

"""
    get_microphysics_1m_options(parsed_args, toml_dict)

Parse the YAML config keys for 1-moment microphysics process options and
return a `NamedTuple` of keyword arguments for `CMP.Microphysics1MParams`.

Each YAML key maps to one field of `get_microphysics_1m_options`, selecting the
process option type that controls dispatch inside `bulk_microphysics_tendencies`.
Option types that carry parameters are constructed from `toml_dict`.
Setting a YAML value to `~` (null) disables the process (`nothing`).
"""
function get_microphysics_1m_options(parsed_args, toml_dict)
    CMP = CM.Parameters

    cloud_liquid_formation = parse_option(
        parsed_args["cloud_liquid_formation"],
        Dict(
            "CloudLiquidFormation" =>
                CMP.CloudLiquidFormation(toml_dict),
        ),
        "cloud_liquid_formation",
    )
    cloud_ice_formation = parse_option(
        parsed_args["cloud_ice_formation"],
        Dict(
            "ConstantTimescale" =>
                CMP.ConstantTimescale(toml_dict),
            "TemperatureDependent" =>
                CMP.TemperatureDependent(toml_dict),
        ),
        "cloud_ice_formation",
    )
    cloud_ice_melt = parse_option(
        parsed_args["cloud_ice_melt"],
        Dict("CloudIceMelt" => CMP.CloudIceMelt()),
        "cloud_ice_melt",
    )
    rain_autoconversion = parse_option(
        parsed_args["rain_autoconversion"],
        Dict(
            "Kessler1M" => CMP.Kessler1M(toml_dict),
            "PrescribedNd" => CMP.PrescribedNd(toml_dict),
        ),
        "rain_autoconversion",
    )
    snow_autoconversion = parse_option(
        parsed_args["snow_autoconversion"],
        Dict(
            "NoSupersaturation" =>
                CMP.NoSupersaturation(toml_dict),
            "WithSupersaturation" =>
                CMP.WithSupersaturation(toml_dict),
        ),
        "snow_autoconversion",
    )
    rain_condensation_evaporation = parse_option(
        parsed_args["rain_condensation_evaporation"],
        Dict("RainEvaporation" => CMP.RainEvaporation()),
        "rain_condensation_evaporation",
    )
    snow_deposition_sublimation = parse_option(
        parsed_args["snow_deposition_sublimation"],
        Dict(
            "SublimationOnly" => CMP.SublimationOnly(),
            "DepositionAndSublimation" =>
                CMP.DepositionAndSublimation(),
        ),
        "snow_deposition_sublimation",
    )
    snow_melt = parse_option(
        parsed_args["snow_melt"],
        Dict("SnowMelt" => CMP.SnowMelt()),
        "snow_melt",
    )
    cloud_liquid_rain_accretion = parse_option(
        parsed_args["cloud_liquid_rain_accretion"],
        Dict(
            "CloudLiquidRainAccretion" =>
                CMP.CloudLiquidRainAccretion(toml_dict),
        ),
        "cloud_liquid_rain_accretion",
    )
    cloud_liquid_snow_accretion = parse_option(
        parsed_args["cloud_liquid_snow_accretion"],
        Dict(
            "CloudLiquidSnowAccretion" =>
                CMP.CloudLiquidSnowAccretion(toml_dict),
        ),
        "cloud_liquid_snow_accretion",
    )
    cloud_ice_rain_accretion = parse_option(
        parsed_args["cloud_ice_rain_accretion"],
        Dict(
            "CloudIceRainAccretion" =>
                CMP.CloudIceRainAccretion(toml_dict),
        ),
        "cloud_ice_rain_accretion",
    )
    cloud_ice_snow_accretion = parse_option(
        parsed_args["cloud_ice_snow_accretion"],
        Dict(
            "CloudIceSnowAccretion" =>
                CMP.CloudIceSnowAccretion(toml_dict),
        ),
        "cloud_ice_snow_accretion",
    )
    rain_snow_accretion = parse_option(
        parsed_args["rain_snow_accretion"],
        Dict(
            "RainSnowAccretion" =>
                CMP.RainSnowAccretion(toml_dict),
        ),
        "rain_snow_accretion",
    )

    return (;
        cloud_liquid_formation,
        cloud_ice_formation,
        cloud_ice_melt,
        rain_autoconversion,
        snow_autoconversion,
        rain_condensation_evaporation,
        snow_deposition_sublimation,
        snow_melt,
        cloud_liquid_rain_accretion,
        cloud_liquid_snow_accretion,
        cloud_ice_rain_accretion,
        cloud_ice_snow_accretion,
        rain_snow_accretion,
    )
end

"""
    parse_option(value, options_map, key_name)

Look up `value` in `options_map` (a `Dict{String, T}`), returning the
corresponding option type. Returns `nothing` when `value` is `nothing`
(YAML `~` / null). Throws an informative error if the value is invalid.
"""
function parse_option(value, options_map, key_name)
    isnothing(value) && return nothing
    haskey(options_map, value) && return options_map[value]
    valid = join(sort(collect(keys(options_map))), ", ")
    error("Invalid `$key_name`: \"$value\". Valid options: $valid")
end

function get_sgs_quadrature(parsed_args, params = nothing)
    use_sgs_quadrature = get(parsed_args, "use_sgs_quadrature", false)
    use_sgs_quadrature || return nothing
    FT = parsed_args["FLOAT_TYPE"] == "Float64" ? Float64 : Float32
    distribution = get_sgs_distribution(parsed_args)
    quadrature_order = get(parsed_args, "quadrature_order", 2)
    T_min = isnothing(params) ? FT(150) : FT(CAP.T_min_sgs(params))
    q_max = isnothing(params) ? FT(0.1) : FT(CAP.q_max_sgs(params))
    return SGSQuadrature(FT; quadrature_order, distribution, T_min, q_max)
end

function get_insolation_form(parsed_args; setup_type = nothing)
    if !isnothing(setup_type)
        model = Setups.insolation_model(setup_type)
        !isnothing(model) && return model
    end
    insolation = parsed_args["insolation"]
    return if insolation == "idealized"
        IdealizedInsolation()
    elseif insolation == "timevarying"
        TimeVaryingInsolation()
    elseif insolation == "rcemipii"
        RCEMIPIIInsolation()
    elseif insolation == "gcmdriven"
        GCMDrivenInsolation()
    elseif insolation == "externaldriventv"
        ExternalTVInsolation()
    elseif insolation == "larcform1"
        Larcform1Insolation()
    else
        error(
            """Unknown insolation `$insolation`. Expected: "idealized", "timevarying", "rcemipii", "gcmdriven", "externaldriventv", or "larcform1".""",
        )
    end
end

function get_hyperdiffusion_model(parsed_args, ::Type{FT}) where {FT}
    hyperdiff_name = parsed_args["hyperdiff"]
    if hyperdiff_name == "Hyperdiffusion"
        return Hyperdiffusion{FT}(;
            ν₄_vorticity_coeff = parsed_args["vorticity_hyperdiffusion_coefficient"],
            divergence_damping_factor = parsed_args["divergence_damping_factor"],
            prandtl_number = parsed_args["hyperdiffusion_prandtl_number"],
        )
    elseif hyperdiff_name == "CAM_SE"
        # Ensure the user isn't trying to set the values manually from the config as CAM_SE defines a set of hyperdiffusion coefficients
        cam_se_hyperdiff = cam_se_hyperdiffusion(FT)
        coeff_pairs = [
            (cam_se_hyperdiff.ν₄_vorticity_coeff, "vorticity_hyperdiffusion_coefficient"),
            (cam_se_hyperdiff.divergence_damping_factor, "divergence_damping_factor"),
            (cam_se_hyperdiff.prandtl_number, "hyperdiffusion_prandtl_number"),
        ]

        for (cam_coef, config_coef) in coeff_pairs
            # check to machine precision
            config_val = FT(parsed_args[config_coef])
            @assert isapprox(cam_coef, config_val, atol = 1e-8) "CAM_SE hyperdiffusion overwrites $config_coef, use `hyperdiff: Hyperdiffusion` to set this value manually in the config instead."
        end
        return cam_se_hyperdiff
    elseif isnothing(hyperdiff_name)
        return nothing
    else
        error(
            """Uncaught hyperdiff `$hyperdiff_name`. Expected: ~ | "Hyperdiffusion" | "CAM_SE".""",
        )
    end
end

function get_vertical_diffusion_model(
    disable_momentum_vertical_diffusion,
    parsed_args,
    params,
    ::Type{FT},
) where {FT}
    vert_diff_name = parsed_args["vert_diff"]
    vdp = CAP.vert_diff_params(params)
    return if isnothing(vert_diff_name)
        nothing
    elseif vert_diff_name == "VerticalDiffusion"
        VerticalDiffusion{disable_momentum_vertical_diffusion, FT}(;
            C_E = vdp.C_E,
        )
    elseif vert_diff_name == "DecayWithHeightDiffusion"
        DecayWithHeightDiffusion{disable_momentum_vertical_diffusion, FT}(;
            H = vdp.H,
            D₀ = vdp.D₀,
        )
    else
        error(
            """Uncaught vert_diff `$vert_diff_name`. Expected: ~ | "VerticalDiffusion" | "DecayWithHeightDiffusion".""",
        )
    end
end

function get_non_orographic_gravity_wave_model(
    parsed_args,
    params,
    ::Type{FT},
) where {FT}
    nogw_name = parsed_args["non_orographic_gravity_wave"]
    return if nogw_name == true
        (;
            source_pressure,
            damp_pressure,
            source_height,
            Bw,
            Bn,
            dc,
            cmax,
            c0,
            nk,
            cw,
            cw_tropics,
            cn,
            Bt_0,
            Bt_n,
            Bt_s,
            Bt_eq,
            ϕ0_n,
            ϕ0_s,
            dϕ_n,
            dϕ_s,
        ) = params.non_orographic_gravity_wave_params
        NonOrographicGravityWave{FT}(;
            source_pressure,
            damp_pressure,
            source_height,
            Bw,
            Bn,
            dc,
            cmax,
            c0,
            nk,
            cw,
            cw_tropics,
            cn,
            Bt_0,
            Bt_n,
            Bt_s,
            Bt_eq,
            ϕ0_n,
            ϕ0_s,
            dϕ_n,
            dϕ_s,
        )
    else
        nothing
    end
end

function get_orographic_gravity_wave_model(parsed_args, params, ::Type{FT}) where {FT}
    ogw_name = parsed_args["orographic_gravity_wave"]
    isnothing(ogw_name) && return nothing
    return if ogw_name == "raw_topo" || ogw_name == "gfdl_restart"
        (; γ, ϵ, β, h_frac, ρscale, L0, a0, a1, Fr_crit) =
            params.orographic_gravity_wave_params
        topo_info = Val(Symbol(parsed_args["orographic_gravity_wave"]))
        topography = Val(Symbol(parsed_args["topography"]))
        FullOrographicGravityWave{FT, typeof(topo_info), typeof(topography)}(;
            γ,
            ϵ,
            β,
            h_frac,
            ρscale,
            L0,
            a0,
            a1,
            Fr_crit,
            topo_info,
            topography,
        )
    elseif ogw_name == "linear"
        LinearOrographicGravityWave(; topo_info = Val(:linear))
    else
        error(
            """Unknown orographic_gravity_wave `$ogw_name`. Expected: ~, "gfdl_restart", "raw_topo", or "linear".""",
        )
    end
end

function get_radiation_mode(parsed_args, ::Type{FT}; setup_type = nothing) where {FT}
    radiation_name = parsed_args["rad"]
    # Use setup default only when config doesn't explicitly set rad
    if isnothing(radiation_name) && !isnothing(setup_type)
        model = Setups.radiation_model(setup_type, FT)
        !isnothing(model) && return model
    end
    idealized_h2o = parsed_args["idealized_h2o"]
    idealized_clouds = parsed_args["idealized_clouds"]
    cloud = get_cloud_in_radiation(parsed_args)
    if idealized_clouds && (cloud isa PrescribedCloudInRadiation)
        error(
            "idealized_clouds and prescribe_clouds_in_radiation cannot be true at the same time",
        )
    end
    add_isothermal_boundary_layer = parsed_args["add_isothermal_boundary_layer"]
    aerosol_radiation = parsed_args["aerosol_radiation"]
    reset_rng_seed = parsed_args["radiation_reset_rng_seed"]
    deep_atmosphere = parsed_args["deep_atmosphere"]
    if !(radiation_name in ("allsky", "allskywithclear")) && reset_rng_seed
        @warn "reset_rng_seed does not have any effect with $radiation_name radiation option"
    end
    if !(radiation_name in ("allsky", "allskywithclear")) &&
       (cloud isa PrescribedCloudInRadiation)
        @warn "prescribe_clouds_in_radiation does not have any effect with $radiation_name radiation option"
    end
    return if radiation_name == "gray"
        RRTMGPI.GrayRadiation(;
            add_isothermal_boundary_layer,
            deep_atmosphere,
        )
    elseif radiation_name == "clearsky"
        RRTMGPI.ClearSkyRadiation(;
            idealized_h2o,
            add_isothermal_boundary_layer,
            aerosol_radiation,
            deep_atmosphere,
        )
    elseif radiation_name == "allsky"
        RRTMGPI.AllSkyRadiation(;
            idealized_h2o,
            idealized_clouds,
            cloud,
            add_isothermal_boundary_layer,
            aerosol_radiation,
            reset_rng_seed,
            deep_atmosphere,
        )
    elseif radiation_name == "allskywithclear"
        RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics(;
            idealized_h2o,
            idealized_clouds,
            cloud,
            add_isothermal_boundary_layer,
            aerosol_radiation,
            reset_rng_seed,
            deep_atmosphere,
        )
    elseif radiation_name == "held_suarez"
        HeldSuarezForcing()
    elseif radiation_name == "DYCOMS"
        RadiationDYCOMS{FT}()
    elseif radiation_name == "TRMM_LBA"
        RadiationTRMM_LBA(FT)
    elseif radiation_name == "ISDAC"
        RadiationISDAC{FT}()
    elseif isnothing(radiation_name)
        nothing
    else
        error(
            """Unknown rad `$radiation_name`. Expected: ~, "clearsky", "gray", "allsky", "allskywithclear", "held_suarez", "DYCOMS", "TRMM_LBA", or "ISDAC".""",
        )
    end
end


"""
    get_sgs_distribution(parsed_args)

Parse the SGS distribution type from configuration.

# Config value mapping

  - `"lognormal"` → `LogNormalSGS()`
  - `"gaussian"` → `GaussianSGS()`
  - `"mean"` → `GridMeanSGS()` (grid-mean only, no SGS sampling)
"""
function get_sgs_distribution(parsed_args)
    dist_name = parsed_args["sgs_distribution"]
    return if dist_name == "lognormal"
        LogNormalSGS()
    elseif dist_name == "gaussian"
        GaussianSGS()
    elseif dist_name == "mean"
        GridMeanSGS()
    else
        error("Invalid sgs_distribution $(dist_name). Use: lognormal, gaussian, mean")
    end
end

function get_tracer_nonnegativity_method(parsed_args)
    method = parsed_args["tracer_nonnegativity_method"]
    isnothing(method) && return nothing
    qtot = endswith(method, "_qtot")  # whether to apply tracer nonnegativity to qtot as well
    method = qtot ? chop(method; tail = 5) : method
    return if method == "elementwise_constraint"
        TracerNonnegativityElementConstraint{qtot}()
    elseif method == "vapor_constraint"
        TracerNonnegativityVaporConstraint{qtot}()
    elseif method == "vapor_tendency"
        qtot && warn("`tracer_nonnegativity_method` $(method) does not support \
                        `_qtot` suffix. qtot will be ignored.")
        TracerNonnegativityVaporTendency()
    elseif method == "vertical_water_borrowing"
        qtot && warn("`tracer_nonnegativity_method` $(method) does not support \
                        `_qtot` suffix. qtot will be ignored.")
        TracerNonnegativityVerticalWaterBorrowing()
    else
        error("Invalid `tracer_nonnegativity_method` $(method)")
    end
end

function get_cloud_model(parsed_args, params)
    cloud_model = parsed_args["cloud_model"]
    FT = parsed_args["FLOAT_TYPE"] == "Float64" ? Float64 : Float32

    return if cloud_model == "grid_scale"
        GridScaleCloud()
    elseif cloud_model == "quadrature"
        QuadratureCloud()
    elseif cloud_model == "MLCloud"
        nn_filepath = joinpath(
            @clima_artifact("cloud_fraction_nn"),
            parsed_args["cloud_nn_architecture"],
        )
        nn_model_data = JLD2.load(nn_filepath)
        nn_architecture = nn_model_data["re"]

        nn_param_vec = FT.(CAP.cloud_fraction_param_vec(params))
        # build the model
        cf_nn_model = nn_architecture(nn_param_vec)
        MLCloud_constructor(cf_nn_model)
    else
        error("Invalid cloud_model $(cloud_model)")
    end
end

function get_cloud_in_radiation(parsed_args)
    isnothing(parsed_args["prescribe_clouds_in_radiation"]) && return nothing
    return parsed_args["prescribe_clouds_in_radiation"] ?
           PrescribedCloudInRadiation() : InteractiveCloudInRadiation()
end


function get_subsidence_model(::Type{FT}; setup_type = nothing) where {FT}
    isnothing(setup_type) && return nothing
    profile = Setups.subsidence_forcing(setup_type, FT)
    return isnothing(profile) ? nothing : Subsidence(profile)
end

function get_large_scale_advection_model(
    ::Type{FT}; setup_type = nothing,
) where {FT}
    isnothing(setup_type) && return nothing
    data = Setups.large_scale_advection_forcing(setup_type, FT)
    isnothing(data) && return nothing
    prof_dqtdt = (_, _, _, z) -> data.prof_dqtdt(z)
    prof_dTdt =
        (thermo_params, p, _, z) ->
            data.prof_dTdt(TD.exner_given_pressure(thermo_params, p), z)
    return LargeScaleAdvection(prof_dTdt, prof_dqtdt)
end

function get_external_forcing_model(
    parsed_args,
    ::Type{FT};
    setup_type = nothing,
) where {FT}
    external_forcing = parsed_args["external_forcing"]
    if isnothing(external_forcing) && !isnothing(setup_type)
        model = Setups.external_forcing(setup_type, FT)
        !isnothing(model) && return model
    end
    if external_forcing in
       ("ReanalysisTimeVarying", "ReanalysisMonthlyAveragedDiurnal")
        @assert parsed_args["config"] == "column" "ReanalysisTimeVarying and ReanalysisMonthlyAveragedDiurnal are only supported in column mode."
    end
    if !isnothing(parsed_args["era5_diurnal_warming"])
        @assert external_forcing == "ReanalysisMonthlyAveragedDiurnal" "era5_diurnal_warming is only supported for ReanalysisMonthlyAveragedDiurnal."
        @assert parsed_args["era5_diurnal_warming"] isa Number "era5_diurnal_warming is expected to be a number, but was supplied as a $(typeof(parsed_args["era5_diurnal_warming"]))"
    end
    return if isnothing(external_forcing)
        nothing
    elseif external_forcing == "GCM"
        cfsite_number_str = parsed_args["cfsite_number"]

        GCMForcing{FT}(parsed_args["external_forcing_file"], cfsite_number_str)

    elseif external_forcing == "ReanalysisTimeVarying"
        # File is already generated by get_setup_type; reuse its path.
        external_forcing_file = setup_type.external_forcing_file
        ExternalDrivenTVForcing{FT}(external_forcing_file)
    elseif external_forcing == "ReanalysisMonthlyAveragedDiurnal"
        external_forcing_file =
            get_external_monthly_forcing_file_path(parsed_args)
        # generate single file from monthly averaged diurnal data if it doesn't exist
        # we'll use ClimaUtilities.TimeVaryingInputs downstream to repeat the data.
        if !isfile(external_forcing_file) ||
           !check_monthly_forcing_times(external_forcing_file, parsed_args)
            generate_external_forcing_file(
                parsed_args,
                external_forcing_file,
                FT,
                input_data_dir = joinpath(
                    @clima_artifact("era5_hourly_atmos_raw"),
                    "monthly",
                ),
                data_strs = [
                    "monthly_diurnal_profiles",
                    "monthly_diurnal_inst",
                    "monthly_diurnal_accum",
                ],
            )
        end
        ExternalDrivenTVForcing{FT}(external_forcing_file)

    elseif external_forcing == "ISDAC"
        ISDACForcing()
    elseif external_forcing == "ARMVARANAL"
        ARMVARANALForcing{FT}(parsed_args["external_forcing_file"])
    else
        error(
            """Unknown external_forcing `$external_forcing`. Expected: ~, "GCM", "ARMVARANAL", "ISDAC", "ReanalysisTimeVarying", or "ReanalysisMonthlyAveragedDiurnal".""",
        )
    end
end

function get_scm_coriolis(::Type{FT}; setup_type = nothing) where {FT}
    isnothing(setup_type) && return nothing
    return Setups.coriolis_forcing(setup_type, FT)
end

function get_turbconv_model(FT, parsed_args, turbconv_params)
    turbconv = parsed_args["turbconv"]
    n_updrafts = parsed_args["updraft_number"]
    prognostic_tke = parsed_args["prognostic_tke"]
    area_fraction = turbconv_params.min_area
    return if turbconv == "prognostic_edmfx"
        PrognosticEDMFX(; n_updrafts, prognostic_tke, area_fraction)
    elseif turbconv == "edonly_edmfx"
        EDOnlyEDMFX()
    elseif isnothing(turbconv) || turbconv == "edmfx"
        nothing
    else
        error(
            """Unknown turbconv `$turbconv`. Expected: ~, "edmfx", "prognostic_edmfx", or "edonly_edmfx".""",
        )
    end
end

function get_entrainment_model(parsed_args)
    entr_model = parsed_args["edmfx_entr_model"]
    return if entr_model == nothing || entr_model == "nothing"
        NoEntrainment()
    elseif entr_model == "PiGroups"
        PiGroupsEntrainment()
    elseif entr_model == "Generalized"
        InvZEntrainment()
    else
        error("Invalid entr_model $(entr_model)")
    end
end

function get_detrainment_model(parsed_args)
    detr_model = parsed_args["edmfx_detr_model"]
    return if detr_model == nothing || detr_model == "nothing"
        NoDetrainment()
    elseif detr_model == "PiGroups"
        PiGroupsDetrainment()
    elseif detr_model == "Generalized"
        BuoyancyVelocityDetrainment()
    elseif detr_model == "SmoothArea"
        SmoothAreaDetrainment()
    else
        error("Invalid detr_model $(detr_model)")
    end
end

function get_tracers(parsed_args)
    aerosol_names = Tuple(parsed_args["prescribed_aerosols"])
    time_varying_trace_gas_names = Tuple(parsed_args["time_varying_trace_gases"])
    return (; aerosol_names, time_varying_trace_gas_names)
end

function check_case_consistency(parsed_args)
    ic = parsed_args["initial_condition"]
    surf = parsed_args["surface_setup"]
    rad = parsed_args["rad"]
    microphysics = parsed_args["microphysics_model"]
    extf = parsed_args["external_forcing"]
    imp_vert_diff = parsed_args["implicit_diffusion"]
    vert_diff = parsed_args["vert_diff"]
    turbconv = parsed_args["turbconv"]
    topography = parsed_args["topography"]
    prescribed_flow = parsed_args["prescribed_flow"]
    config = parsed_args["config"]

    # Geometry consistency (always checked, independent of the case-specific
    # checks below)
    valid_configs = ("sphere", "column", "box", "plane")
    @assert(
        config in valid_configs,
        "Unknown `config = $(repr(config))`. Valid options are: $(join(valid_configs, ", "))."
    )

    if parsed_args["edmfx_sgs_horizontal_diffusive_flux"] && (
        !isnothing(parsed_args["smagorinsky_lilly"]) || parsed_args["amd_les"]
    )
        error(
            "`edmfx_sgs_horizontal_diffusive_flux` cannot be combined with \
             `smagorinsky_lilly` or `amd_les`, which already apply horizontal \
             SGS diffusion to the same fields",
        )
    end

    # ISDAC consistency: when initial_condition is ISDAC, surface/rad/external
    # forcing must all be set to the matching ISDAC variants. Subsidence,
    # scm_coriolis, and ls_adv are owned by the setup, not the YAML schema.
    ISDAC_mandatory = (ic, surf, rad, extf)
    if "ISDAC" in ISDAC_mandatory
        @assert(
            allequal(ISDAC_mandatory) &&
            microphysics != "dry",
            "ISDAC setup not consistent"
        )
    elseif imp_vert_diff
        # Implicit vertical diffusion is only supported for specific models:
        @assert(
            !isnothing(turbconv) || !isnothing(vert_diff),
            "Implicit vertical diffusion is only supported when using a " *
            "turbulence convection model or vertical diffusion model.",
        )
    elseif !isnothing(prescribed_flow)
        @assert(topography == "NoWarp",
            "Prescribed flow elides `set_velocity_at_surface!` and `set_velocity_at_top!` \
             which is needed for topography. Thus, prescribed flow must have flat surface."
        )
        @assert(
            !parsed_args["implicit_microphysics"] &&
            !parsed_args["implicit_diffusion"],
            "Prescribed flow does not use the implicit solver."
        )
    end
end
# AtmosConfig-aware constructors for the AtmosModel group structs.
# Each consolidates the YAML→typed-object translation for one group.

function AtmosWater(config::AtmosConfig, params, ::Type{FT}) where {FT}
    pa = config.parsed_args
    microphysics_model = get_microphysics_model(pa)
    sgs_quadrature = get_sgs_quadrature(pa, params)

    if microphysics_model isa DryModel
        @warn "Running simulations without any moisture present."
    end
    if microphysics_model isa EquilibriumMicrophysics0M && isnothing(sgs_quadrature)
        error(
            "EquilibriumMicrophysics0M requires use_sgs_quadrature: true. " *
            "GridMeanSGS fallback is not supported for 0-moment microphysics.",
        )
    end

    cloud_model = get_cloud_model(pa, params)

    terminal_velocity_mode =
        pa["fixed_terminal_velocity"] ?
        FixedTerminalVelocity{FT}(
            CAP.fixed_cloud_liquid_terminal_velocity(params),
            CAP.fixed_cloud_ice_terminal_velocity(params),
            CAP.fixed_rain_terminal_velocity(params),
            CAP.fixed_snow_terminal_velocity(params),
        ) : DiagnosticTerminalVelocity()

    implicit_microphysics = pa["implicit_microphysics"]

    return AtmosWater(;
        microphysics_model,
        cloud_model,
        microphysics_tendency_timestepping = implicit_microphysics ? Implicit() :
                                             Explicit(),
        tracer_nonnegativity_method = get_tracer_nonnegativity_method(pa),
        sgs_quadrature,
        terminal_velocity_mode,
    )
end

function AtmosRadiation(config::AtmosConfig, ::Type{FT}; setup_type = nothing) where {FT}
    pa = config.parsed_args
    return AtmosRadiation(;
        radiation_mode = get_radiation_mode(pa, FT; setup_type),
        insolation = get_insolation_form(pa; setup_type),
    )
end

function AtmosGravityWave(config::AtmosConfig, params, ::Type{FT}) where {FT}
    pa = config.parsed_args
    return AtmosGravityWave(;
        non_orographic_gravity_wave = get_non_orographic_gravity_wave_model(pa, params, FT),
        orographic_gravity_wave = get_orographic_gravity_wave_model(pa, params, FT),
    )
end

function AtmosTurbconv(config::AtmosConfig, params, ::Type{FT}) where {FT}
    pa = config.parsed_args
    turbconv_params = CAP.turbconv_params(params)

    scale_blending_method =
        if pa["edmfx_scale_blending"] == "SmoothMinimum"
            SmoothMinimumBlending()
        elseif pa["edmfx_scale_blending"] == "HardMinimum"
            HardMinimumBlending()
        else
            error("Unknown edmfx_scale_blending method: $(pa["edmfx_scale_blending"])")
        end

    edmfx_model = EDMFXModel(;
        entr_model = get_entrainment_model(pa),
        detr_model = get_detrainment_model(pa),
        sgs_mass_flux = pa["edmfx_sgs_mass_flux"],
        sgs_diffusive_flux = pa["edmfx_sgs_diffusive_flux"],
        sgs_diffusive_flux_horizontal = pa["edmfx_sgs_horizontal_diffusive_flux"],
        nh_pressure = pa["edmfx_nh_pressure"],
        vertical_diffusion = pa["edmfx_vertical_diffusion"],
        horizontal_diffusion = pa["edmfx_horizontal_diffusion"],
        filter = pa["edmfx_filter"],
        scale_blending_method,
    )

    n = pa["smagorinsky_lilly"]
    smagorinsky_lilly =
        isnothing(n) ? nothing : SmagorinskyLilly(; axes = Symbol(n))

    amd_les_active = pa["amd_les"]
    amd_les = amd_les_active ? AnisotropicMinimumDissipation{FT}(pa["c_amd"]) : nothing

    chd_active = pa["constant_horizontal_diffusion"]
    constant_horizontal_diffusion =
        chd_active ?
        ConstantHorizontalDiffusion{FT}(CAP.constant_horizontal_diffusion_D(params)) :
        nothing

    return AtmosTurbconv(;
        edmfx_model,
        turbconv_model = get_turbconv_model(FT, pa, turbconv_params),
        smagorinsky_lilly,
        amd_les,
        constant_horizontal_diffusion,
    )
end

AtmosNumerics(config::AtmosConfig, ::Type{FT}) where {FT} =
    get_numerics(config.parsed_args, FT)

function SCMSetup(config::AtmosConfig, ::Type{FT};
    setup_type = nothing) where {FT}
    return SCMSetup(;
        subsidence = get_subsidence_model(FT; setup_type),
        external_forcing = get_external_forcing_model(config.parsed_args, FT; setup_type),
        ls_adv = get_large_scale_advection_model(FT; setup_type),
        advection_test = config.parsed_args["advection_test"],
        scm_coriolis = get_scm_coriolis(FT; setup_type),
    )
end

function AtmosSponge(config::AtmosConfig, params)
    pa = config.parsed_args

    viscous_sponge = pa["viscous_sponge"] ? ViscousSponge(params) : nothing
    rayleigh_sponge = pa["rayleigh_sponge"] ? RayleighSponge(params) : nothing

    return AtmosSponge(; viscous_sponge, rayleigh_sponge)
end

function AtmosSurface(
    config::AtmosConfig, params, ::Type{FT}; setup_type = nothing,
) where {FT}
    pa = config.parsed_args

    # Resolve setup-provided surface pieces (flux_scheme, temperature, overrides)
    setup_pieces =
        isnothing(setup_type) ?
        (; flux_scheme = nothing, temperature = nothing, overrides = nothing) :
        Setups.surface_condition(setup_type, params)

    temperature = if pa["prognostic_surface"] == "SlabOceanSST"
        SurfaceConditions.SlabOceanTemperature{FT}()
    elseif pa["prognostic_surface"] == "PrescribedSST"
        @something(setup_pieces.temperature, Setups.surface_temperature_model(setup_type))
    else
        error(
            """Uncaught prognostic_surface `$(pa["prognostic_surface"])`. Expected: "PrescribedSST" | "SlabOceanSST".""",
        )
    end

    flux_scheme = if !isnothing(setup_pieces.flux_scheme)
        setup_pieces.flux_scheme
    elseif pa["surface_setup"] == "PrescribedSurface"
        nothing
    else
        getproperty(SurfaceConditions, Symbol(pa["surface_setup"]))()(params)
    end

    boundary_overrides = @something(
        setup_pieces.overrides, SurfaceConditions.SurfaceBoundaryOverrides()
    )

    surface_albedo =
        if pa["albedo_model"] == "ConstantAlbedo"
            ConstantAlbedo{FT}(; α = params.idealized_ocean_albedo)
        elseif pa["albedo_model"] == "RegressionFunctionAlbedo"
            isnothing(pa["rad"]) && error(
                "Radiation model not specified, so cannot use RegressionFunctionAlbedo",
            )
            RegressionFunctionAlbedo{FT}(; n = params.water_refractive_index)
        elseif pa["albedo_model"] == "CouplerAlbedo"
            CouplerAlbedo()
        else
            error("Uncaught surface albedo model `$(pa["albedo_model"])`.")
        end

    return AtmosSurface(;
        flux_scheme, temperature, boundary_overrides, surface_albedo,
    )
end

function AtmosChem(config::AtmosConfig)
    chem = config.parsed_args["chemistry_model"]
    chemistry_model = if isnothing(chem)
        nothing
    elseif chem == "passive"
        GasPhaseChem()
    else
        error(
            """Unknown chemistry_model `$chem`. Expected: ~ | "passive".""",
        )
    end
    return AtmosChem(; chemistry_model)
end
