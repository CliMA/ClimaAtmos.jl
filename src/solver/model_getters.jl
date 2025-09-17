function get_moisture_model(parsed_args)
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

function get_sfc_temperature_form(parsed_args)
    surface_temperature = parsed_args["surface_temperature"]
    @assert surface_temperature in (
        "ZonallyAsymmetric",
        "ZonallySymmetric",
        "RCEMIPII",
        "ReanalysisTimeVarying",
    )
    return if surface_temperature == "ZonallyAsymmetric"
        ZonallyAsymmetricSST()
    elseif surface_temperature == "ZonallySymmetric"
        ZonallySymmetricSST()
    elseif surface_temperature == "RCEMIPII"
        RCEMIPIISST()
    elseif surface_temperature == "ReanalysisTimeVarying"
        ExternalTVColumnSST()
    end
end

function get_insolation_form(parsed_args)
    insolation = parsed_args["insolation"]
    @assert insolation in (
        "idealized",
        "timevarying",
        "rcemipii",
        "gcmdriven",
        "externaldriventv",
    )
    return if insolation == "idealized"
        IdealizedInsolation()
    elseif insolation == "timevarying"
        # TODO: Remove this argument once we have support for integer time and
        # we can easily convert from time to date
        start_date = parse_date(parsed_args["start_date"])
        TimeVaryingInsolation(start_date)
    elseif insolation == "rcemipii"
        RCEMIPIIInsolation()
    elseif insolation == "gcmdriven"
        GCMDrivenInsolation()
    elseif insolation == "externaldriventv"
        ExternalTVInsolation()
    end
end

function get_hyperdiffusion_model(parsed_args, ::Type{FT}) where {FT}
    hyperdiff_name = parsed_args["hyperdiff"]
    if hyperdiff_name in ("ClimaHyperdiffusion", "true", true)
        ν₄_vorticity_coeff =
            FT(parsed_args["vorticity_hyperdiffusion_coefficient"])
        ν₄_scalar_coeff = FT(parsed_args["scalar_hyperdiffusion_coefficient"])
        divergence_damping_factor = FT(parsed_args["divergence_damping_factor"])
        return ClimaHyperdiffusion(;
            ν₄_vorticity_coeff,
            ν₄_scalar_coeff,
            divergence_damping_factor,
        )
    elseif hyperdiff_name == "CAM_SE"
        # To match hyperviscosity coefficients in:
        #    https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2017MS001257
        #    for equation A18 and A19
        # Need to scale by (1.1e5 / (sqrt(4 * pi / 6) * 6.371e6 / (3*30)) )^3  ≈ 1.238
        # These are re-scaled by the grid resolution in function ν₄(hyperdiff, Y)
        # TODO: Why do we need to increase ν₄ by 30% for the new reconstruction?
        ν₄_vorticity_coeff = FT(0.150 * 1.238 * 1.3)
        ν₄_scalar_coeff = FT(0.751 * 1.238 * 1.3)
        divergence_damping_factor = FT(5)
        # Ensure the user isn't trying to set the values manually from the config as CAM_SE defines a set of hyperdiffusion coefficients
        coeff_pairs = [
            (ν₄_vorticity_coeff, "vorticity_hyperdiffusion_coefficient"),
            (ν₄_scalar_coeff, "scalar_hyperdiffusion_coefficient"),
            (divergence_damping_factor, "divergence_damping_factor"),
        ]

        for (cam_coef, config_coef) in coeff_pairs
            # check to machine precision
            config_val = FT(parsed_args[config_coef])
            @assert isapprox(cam_coef, config_val, atol = 1e-7) "CAM_SE hyperdiffusion overwrites $config_coef, use hyperdiff: ClimaHyperdiffusion to set this value manually in the config instead."
        end
        return ClimaHyperdiffusion(;
            ν₄_vorticity_coeff,
            ν₄_scalar_coeff,
            divergence_damping_factor,
        )
    elseif hyperdiff_name in ("none", "false", false)
        return nothing
    else
        error("Uncaught hyperdiffusion model type.")
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
    return if vert_diff_name in ("false", false, "none")
        nothing
    elseif vert_diff_name in ("true", true, "VerticalDiffusion")
        VerticalDiffusion{disable_momentum_vertical_diffusion, FT}(;
            C_E = vdp.C_E,
        )
    elseif vert_diff_name in ("DecayWithHeightDiffusion",)
        DecayWithHeightDiffusion{disable_momentum_vertical_diffusion, FT}(;
            H = vdp.H,
            D₀ = vdp.D₀,
        )
    else
        error("Uncaught diffusion model `$vert_diff_name`.")
    end
end

function get_surface_model(parsed_args)
    prognostic_surface_name = parsed_args["prognostic_surface"]
    return if prognostic_surface_name in ("false", false, "PrescribedSST")
        PrescribedSST()
    elseif prognostic_surface_name in ("true", true, "SlabOceanSST")
        SlabOceanSST()
    elseif prognostic_surface_name == "PrognosticSurfaceTemperature"
        @warn "The `PrognosticSurfaceTemperature` option is deprecated. Use `SlabOceanSST` instead."
        SlabOceanSST()
    elseif prognostic_surface_name == "PrescribedSurfaceTemperature"
        @warn "The `PrescribedSurfaceTemperature` option is deprecated. Use `PrescribedSST` instead."
        PrescribedSST()
    else
        error("Uncaught surface model `$prognostic_surface_name`.")
    end
end

function get_surface_albedo_model(parsed_args, params, ::Type{FT}) where {FT}
    albedo_name = parsed_args["albedo_model"]
    return if albedo_name in ("ConstantAlbedo",)
        ConstantAlbedo{FT}(; α = params.idealized_ocean_albedo)
    elseif albedo_name in ("RegressionFunctionAlbedo",)
        isnothing(parsed_args["rad"]) && error(
            "Radiation model not specified, so cannot use RegressionFunctionAlbedo",
        )
        RegressionFunctionAlbedo{FT}(; n = params.water_refractive_index)
    elseif albedo_name in ("CouplerAlbedo",)
        CouplerAlbedo()
    else
        error("Uncaught surface albedo model `$albedo_name`.")
    end
end

function get_viscous_sponge_model(parsed_args, params, ::Type{FT}) where {FT}
    vs_name = parsed_args["viscous_sponge"]
    return if vs_name in ("false", false, "none")
        nothing
    elseif vs_name in ("true", true, "ViscousSponge")
        zd = params.zd_viscous
        κ₂ = params.kappa_2_sponge
        ViscousSponge{FT}(; zd, κ₂)
    else
        error("Uncaught viscous sponge model `$vs_name`.")
    end
end

function get_smagorinsky_lilly_model(parsed_args)
    is_model_active = parsed_args["smagorinsky_lilly"]
    @assert is_model_active in (true, false)
    return is_model_active ? SmagorinskyLilly() : nothing
end

function get_rayleigh_sponge_model(parsed_args, params, ::Type{FT}) where {FT}
    rs_name = parsed_args["rayleigh_sponge"]
    return if rs_name in ("false", false)
        nothing
    elseif rs_name in ("true", true, "RayleighSponge")
        zd = params.zd_rayleigh
        α_uₕ = params.alpha_rayleigh_uh
        α_w = params.alpha_rayleigh_w
        α_sgs_tracer = params.alpha_rayleigh_sgs_tracer
        RayleighSponge{FT}(; zd, α_uₕ, α_w, α_sgs_tracer)
    else
        error("Uncaught rayleigh sponge model `$rs_name`.")
    end
end

function get_non_orographic_gravity_wave_model(
    parsed_args,
    ::Type{FT},
) where {FT}
    nogw_name = parsed_args["non_orographic_gravity_wave"]
    @assert nogw_name in (true, false)
    return if nogw_name == true
        if parsed_args["config"] == "column"
            NonOrographicGravityWave{FT}(; Bw = 1.2, Bn = 0.0, Bt_0 = 4e-3)
        elseif parsed_args["config"] == "sphere"
            NonOrographicGravityWave{FT}(;
                Bw = 0.4,
                Bn = 0.0,
                cw = 35.0,
                cw_tropics = 35.0,
                cn = 2.0,
                Bt_0 = 0.0043,
                Bt_n = 0.0,
                Bt_eq = 0.0043,
                Bt_s = 0.0,
                ϕ0_n = 15,
                ϕ0_s = -15,
                dϕ_n = 10,
                dϕ_s = -10,
            )
        else
            error("Uncaught case")
        end
    else
        nothing
    end
end

function get_orographic_gravity_wave_model(parsed_args, ::Type{FT}) where {FT}
    ogw_name = parsed_args["orographic_gravity_wave"]
    @assert ogw_name in (nothing, "gfdl_restart", "raw_topo")
    return if ogw_name == "gfdl_restart"
        OrographicGravityWave{FT, String}()
    elseif ogw_name == "raw_topo"
        OrographicGravityWave{FT, String}(topo_info = "raw_topo")
    else
        nothing
    end
end

function get_radiation_mode(parsed_args, ::Type{FT}) where {FT}
    idealized_h2o = parsed_args["idealized_h2o"]
    @assert idealized_h2o in (true, false)
    idealized_clouds = parsed_args["idealized_clouds"]
    @assert idealized_clouds in (true, false)
    cloud = get_cloud_in_radiation(parsed_args)
    if idealized_clouds && (cloud isa PrescribedCloudInRadiation)
        error(
            "idealized_clouds and prescribe_clouds_in_radiation cannot be true at the same time",
        )
    end
    add_isothermal_boundary_layer = parsed_args["add_isothermal_boundary_layer"]
    @assert add_isothermal_boundary_layer in (true, false)
    aerosol_radiation = parsed_args["aerosol_radiation"]
    @assert aerosol_radiation in (true, false)
    reset_rng_seed = parsed_args["radiation_reset_rng_seed"]
    @assert reset_rng_seed in (true, false)
    radiation_name = parsed_args["rad"]
    deep_atmosphere = parsed_args["deep_atmosphere"]
    @assert radiation_name in (
        nothing,
        "nothing",
        "clearsky",
        "gray",
        "allsky",
        "allskywithclear",
        "held_suarez",
        "DYCOMS",
        "TRMM_LBA",
        "ISDAC",
    )
    if !(radiation_name in ("allsky", "allskywithclear")) && reset_rng_seed
        @warn "reset_rng_seed does not have any effect with $radiation_name radiation option"
    end
    if !(radiation_name in ("allsky", "allskywithclear")) &&
       (cloud isa PrescribedCloudInRadiation)
        @warn "prescribe_clouds_in_radiation does not have any effect with $radiation_name radiation option"
    end
    return if radiation_name == "gray"
        RRTMGPI.GrayRadiation(add_isothermal_boundary_layer, deep_atmosphere)
    elseif radiation_name == "clearsky"
        RRTMGPI.ClearSkyRadiation(
            idealized_h2o,
            add_isothermal_boundary_layer,
            aerosol_radiation,
            deep_atmosphere,
        )
    elseif radiation_name == "allsky"
        RRTMGPI.AllSkyRadiation(
            idealized_h2o,
            idealized_clouds,
            cloud,
            add_isothermal_boundary_layer,
            aerosol_radiation,
            reset_rng_seed,
            deep_atmosphere,
        )
    elseif radiation_name == "allskywithclear"
        RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics(
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
    else
        nothing
    end
end

function get_microphysics_model(parsed_args)
    microphysics_model = parsed_args["precip_model"]
    return if isnothing(microphysics_model) || microphysics_model == "nothing"
        NoPrecipitation()
    elseif microphysics_model == "0M"
        Microphysics0Moment()
    elseif microphysics_model == "1M"
        Microphysics1Moment()
    elseif microphysics_model == "2M"
        Microphysics2Moment()
    else
        error("Invalid microphysics_model $(microphysics_model)")
    end
end

function get_cloud_model(parsed_args)
    cloud_model = parsed_args["cloud_model"]
    FT = parsed_args["FLOAT_TYPE"] == "Float64" ? Float64 : Float32
    return if cloud_model == "grid_scale"
        GridScaleCloud()
    elseif cloud_model == "quadrature"
        QuadratureCloud(SGSQuadrature(FT))
    elseif cloud_model == "quadrature_sgs"
        SGSQuadratureCloud(SGSQuadrature(FT))
    else
        error("Invalid cloud_model $(cloud_model)")
    end
end

function get_ozone(parsed_args)
    isnothing(parsed_args["prescribe_ozone"]) && return nothing
    return parsed_args["prescribe_ozone"] ? PrescribedOzone() : IdealizedOzone()
end

function get_co2(parsed_args)
    if isnothing(parsed_args["co2_model"])
        return nothing
    elseif lowercase(parsed_args["co2_model"]) == "fixed"
        return FixedCO2()
    elseif lowercase(parsed_args["co2_model"]) == "maunaloa"
        return MaunaLoaCO2()
    else
        error("The CO2 models supported are $(subtypes(AbstractCO2))")
    end
end

function get_cloud_in_radiation(parsed_args)
    isnothing(parsed_args["prescribe_clouds_in_radiation"]) && return nothing
    return parsed_args["prescribe_clouds_in_radiation"] ?
           PrescribedCloudInRadiation() : InteractiveCloudInRadiation()
end

function get_forcing_type(parsed_args)
    forcing = parsed_args["forcing"]
    @assert forcing in (nothing, "held_suarez")
    if forcing == "held_suarez"
        @warn "The 'held_suarez' forcing option is deprecated. Use rad='held_suarez' instead to set HeldSuarezForcing as a radiation mode."
        return HeldSuarezForcing()  # Still return the object for backward compatibility
    end
    return nothing
end

struct CallCloudDiagnosticsPerStage end
function get_call_cloud_diagnostics_per_stage(parsed_args)
    ccdps = parsed_args["call_cloud_diagnostics_per_stage"]
    @assert ccdps in (nothing, true, false)
    return if ccdps in (nothing, false)
        nothing
    elseif ccdps == true
        CallCloudDiagnosticsPerStage()
    end
end

function get_subsidence_model(parsed_args, radiation_mode, FT)
    subsidence = parsed_args["subsidence"]
    isnothing(subsidence) && return nothing

    prof = if subsidence == "Bomex"
        APL.Bomex_subsidence(FT)
    elseif subsidence == "LifeCycleTan2018"
        APL.LifeCycleTan2018_subsidence(FT)
    elseif subsidence == "Rico"
        APL.Rico_subsidence(FT)
    elseif subsidence == "DYCOMS"
        @assert radiation_mode isa RadiationDYCOMS
        # For DYCOMS case, subsidence is linearly proportional to height
        # with slope equal to the divergence rate specified in radiation mode
        z -> -z * radiation_mode.divergence
    elseif subsidence == "ISDAC"
        APL.ISDAC_subsidence(FT)
    else
        error("Uncaught case")
    end
    return Subsidence(prof)
end

function get_large_scale_advection_model(parsed_args, ::Type{FT}) where {FT}
    ls_adv = parsed_args["ls_adv"]
    ls_adv == nothing && return nothing

    (prof_dTdt₀, prof_dqtdt₀) = if ls_adv == "Bomex"
        (APL.Bomex_dTdt(FT), APL.Bomex_dqtdt(FT))
    elseif ls_adv == "LifeCycleTan2018"
        (APL.LifeCycleTan2018_dTdt(FT), APL.LifeCycleTan2018_dqtdt(FT))
    elseif ls_adv == "Rico"
        (APL.Rico_dTdt(FT), APL.Rico_dqtdt(FT))
    elseif ls_adv == "ARM_SGP"
        (APL.ARM_SGP_dTdt(FT), APL.ARM_SGP_dqtdt(FT))
    elseif ls_adv == "GATE_III"
        (APL.GATE_III_dTdt(FT), APL.GATE_III_dqtdt(FT))
    else
        error("Uncaught case")
    end
    # See https://clima.github.io/AtmosphericProfilesLibrary.jl/dev/
    # for which functions accept which arguments.
    prof_dqtdt = if ls_adv in ("Bomex", "LifeCycleTan2018", "Rico", "GATE_III")
        (thermo_params, ᶜts, t, z) -> prof_dqtdt₀(z)
    elseif ls_adv == "ARM_SGP"
        (thermo_params, ᶜts, t, z) ->
            prof_dqtdt₀(TD.exner(thermo_params, ᶜts), t, z)
    end
    prof_dTdt = if ls_adv in ("Bomex", "LifeCycleTan2018", "Rico")
        (thermo_params, ᶜts, t, z) ->
            prof_dTdt₀(TD.exner(thermo_params, ᶜts), z)
    elseif ls_adv == "ARM_SGP"
        (thermo_params, ᶜts, t, z) -> prof_dTdt₀(t, z)
    elseif ls_adv == "GATE_III"
        (thermo_params, ᶜts, t, z) -> prof_dTdt₀(z)
    end

    return LargeScaleAdvection(prof_dTdt, prof_dqtdt)
end

function get_external_forcing_model(parsed_args, ::Type{FT}) where {FT}
    external_forcing = parsed_args["external_forcing"]
    @assert external_forcing in (
        nothing,
        "GCM",
        "ReanalysisTimeVarying",
        "ReanalysisMonthlyAveragedDiurnal",
        "ISDAC",
    )
    reanalysis_required_fields = map(
        x -> parsed_args[x],
        ["surface_setup", "surface_temperature", "initial_condition"],
    )
    if external_forcing in
       ("ReanalysisTimeVarying", "ReanalysisMonthlyAveragedDiurnal")
        @assert parsed_args["config"] == "column" "ReanalysisTimeVarying and ReanalysisMonthlyAveragedDiurnal are only supported in column mode."
        @assert all(reanalysis_required_fields .== "ReanalysisTimeVarying") "All of external_forcing, surface_setup, surface_temperature and initial_condition must be set to ReanalysisTimeVarying."
    end
    return if isnothing(external_forcing)
        nothing
    elseif external_forcing == "GCM"
        cfsite_number_str = parsed_args["cfsite_number"]

        GCMForcing{FT}(parsed_args["external_forcing_file"], cfsite_number_str)

    elseif external_forcing == "ReanalysisTimeVarying"
        external_forcing_file =
            get_external_daily_forcing_file_path(parsed_args)
        if !isfile(external_forcing_file) ||
           !check_daily_forcing_times(external_forcing_file, parsed_args)
            @info "External forcing file $(external_forcing_file) does not exist or does not cover the expected time range. Generating it now."
            # generate forcing from provided era5 data paths
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
    end
end

function get_scm_coriolis(parsed_args, ::Type{FT}) where {FT}
    scm_coriolis = parsed_args["scm_coriolis"]
    scm_coriolis == nothing && return nothing
    (prof_u, prof_v) = if scm_coriolis == "Bomex"
        (APL.Bomex_geostrophic_u(FT), z -> FT(0))
    elseif scm_coriolis == "LifeCycleTan2018"
        (APL.LifeCycleTan2018_geostrophic_u(FT), z -> FT(0))
    elseif scm_coriolis == "Rico"
        (APL.Rico_geostrophic_ug(FT), APL.Rico_geostrophic_vg(FT))
    elseif scm_coriolis == "ARM_SGP"
        (z -> FT(10), z -> FT(0))
    elseif scm_coriolis == "DYCOMS_RF01"
        (z -> FT(7), z -> FT(-5.5))
    elseif scm_coriolis == "DYCOMS_RF02"
        (z -> FT(5), z -> FT(-5.5))
    elseif scm_coriolis == "GABLS"
        (APL.GABLS_geostrophic_ug(FT), APL.GABLS_geostrophic_vg(FT))
    else
        error("Uncaught case")
    end

    coriolis_params = Dict()
    coriolis_params["Bomex"] = FT(0.376e-4)
    coriolis_params["LifeCycleTan2018"] = FT(0.376e-4)
    coriolis_params["Rico"] = FT(4.5e-5)
    coriolis_params["ARM_SGP"] = FT(8.5e-5)
    coriolis_params["DYCOMS_RF01"] = FT(0) # TODO: check this
    coriolis_params["DYCOMS_RF02"] = FT(0) # TODO: check this
    coriolis_params["GABLS"] = FT(1.39e-4)
    coriolis_param = coriolis_params[scm_coriolis]
    return SCMCoriolis(prof_u, prof_v, coriolis_param)
end

function get_turbconv_model(FT, parsed_args, turbconv_params)
    turbconv = parsed_args["turbconv"]
    @assert turbconv in (
        nothing,
        "edmfx",
        "prognostic_edmfx",
        "diagnostic_edmfx",
        "edonly_edmfx",
    )

    return if turbconv == "prognostic_edmfx"
        N = parsed_args["updraft_number"]
        TKE = parsed_args["prognostic_tke"]
        PrognosticEDMFX{N, TKE}(turbconv_params.min_area)
    elseif turbconv == "diagnostic_edmfx"
        N = parsed_args["updraft_number"]
        TKE = parsed_args["prognostic_tke"]
        DiagnosticEDMFX{N, TKE}(turbconv_params.min_area)
    elseif turbconv == "edonly_edmfx"
        EDOnlyEDMFX()
    else
        nothing
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

function get_surface_thermo_state_type(parsed_args)
    dict = Dict()
    dict["GCMSurfaceThermoState"] = GCMSurfaceThermoState()
    return dict[parsed_args["surface_thermo_state_type"]]
end

function get_tracers(parsed_args)
    aerosol_names = Tuple(parsed_args["prescribed_aerosols"])
    return (; aerosol_names)
end

function check_case_consistency(parsed_args)
    # if any flags is ISDAC, check that all are ISDAC
    ic = parsed_args["initial_condition"]
    subs = parsed_args["subsidence"]
    surf = parsed_args["surface_setup"]
    rad = parsed_args["rad"]
    cor = parsed_args["scm_coriolis"]
    forc = parsed_args["forcing"]
    moist = parsed_args["moist"]
    ls_adv = parsed_args["ls_adv"]
    extf = parsed_args["external_forcing"]

    ISDAC_mandatory = (ic, subs, surf, rad, extf)
    if "ISDAC" in ISDAC_mandatory
        @assert(
            allequal(ISDAC_mandatory) &&
            all(isnothing, (cor, forc, ls_adv)) &&
            moist != "dry",
            "ISDAC setup not consistent"
        )
    end
end
