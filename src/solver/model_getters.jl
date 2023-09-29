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

function get_model_config(parsed_args)
    config = parsed_args["config"]
    @assert config in ("sphere", "column", "box", "plane")
    return if config == "sphere"
        SphericalModel()
    elseif config == "column"
        SingleColumnModel()
    elseif config == "box"
        BoxModel()
    elseif config == "plane"
        PlaneModel()
    end
end

function get_sfc_temperature_form(parsed_args)
    surface_temperature = parsed_args["surface_temperature"]
    @assert surface_temperature in ("ZonallyAsymmetric", "ZonallySymmetric")
    return if surface_temperature == "ZonallyAsymmetric"
        ZonallyAsymmetricSST()
    elseif surface_temperature == "ZonallySymmetric"
        ZonallySymmetricSST()
    end
end

function get_hyperdiffusion_model(parsed_args, ::Type{FT}) where {FT}
    hyperdiff_name = parsed_args["hyperdiff"]
    κ₄ = FT(parsed_args["kappa_4"])
    divergence_damping_factor = FT(parsed_args["divergence_damping_factor"])
    return if hyperdiff_name in ("ClimaHyperdiffusion", "true", true)
        ClimaHyperdiffusion(; κ₄, divergence_damping_factor)
    elseif hyperdiff_name in ("none", "false", false)
        nothing
    else
        error("Uncaught hyperdiffusion model type.")
    end
end

function get_vertical_diffusion_model(
    diffuse_momentum,
    parsed_args,
    params,
    ::Type{FT},
) where {FT}
    vert_diff_name = parsed_args["vert_diff"]
    return if vert_diff_name in ("false", false, "none")
        nothing
    elseif vert_diff_name in ("true", true, "VerticalDiffusion")
        VerticalDiffusion{diffuse_momentum, FT}(; C_E = params.C_E)
    else
        error("Uncaught diffusion model `$vert_diff_name`.")
    end
end

function get_surface_model(parsed_args)
    prognostic_surface_name = parsed_args["prognostic_surface"]
    return if prognostic_surface_name in
              ("false", false, "PrescribedSurfaceTemperature")
        PrescribedSurfaceTemperature()
    elseif prognostic_surface_name in
           ("true", true, "PrognosticSurfaceTemperature")
        PrognosticSurfaceTemperature()
    else
        error("Uncaught surface model `$prognostic_surface_name`.")
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

function get_rayleigh_sponge_model(parsed_args, params, ::Type{FT}) where {FT}
    rs_name = parsed_args["rayleigh_sponge"]
    return if rs_name in ("false", false)
        nothing
    elseif rs_name in ("true", true, "RayleighSponge")
        zd = params.zd_rayleigh
        α_uₕ = params.alpha_rayleigh_uh
        α_w = params.alpha_rayleigh_w
        RayleighSponge{FT}(; zd, α_uₕ, α_w)
    else
        error("Uncaught rayleigh sponge model `$rs_name`.")
    end
end

function get_non_orographic_gravity_wave_model(
    parsed_args,
    model_config,
    ::Type{FT},
) where {FT}
    nogw_name = parsed_args["non_orographic_gravity_wave"]
    @assert nogw_name in (true, false)
    return if nogw_name == true
        if model_config isa SingleColumnModel
            NonOrographyGravityWave{FT}(; Bw = 1.2, Bn = 0.0, Bt_0 = 4e-3)
        elseif model_config isa SphericalModel
            NonOrographyGravityWave{FT}(;
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

function get_perf_mode(parsed_args)
    return if parsed_args["perf_mode"] == "PerfExperimental"
        PerfExperimental()
    else
        PerfStandard()
    end
end

function get_energy_form(parsed_args, vert_diff)
    energy_name = parsed_args["energy_name"]
    @assert energy_name in ("rhoe", "rhotheta")
    if !isnothing(vert_diff)
        @assert energy_name == "rhoe"
    end
    return if energy_name == "rhoe"
        TotalEnergy()
    elseif energy_name == "rhotheta"
        PotentialTemperature()
    end
end

function get_radiation_mode(parsed_args, ::Type{FT}) where {FT}
    idealized_h2o = parsed_args["idealized_h2o"]
    @assert idealized_h2o in (true, false)
    idealized_insolation = parsed_args["idealized_insolation"]
    @assert idealized_insolation in (true, false)
    idealized_clouds = parsed_args["idealized_clouds"]
    @assert idealized_clouds in (true, false)
    radiation_name = parsed_args["rad"]
    @assert radiation_name in (
        nothing,
        "nothing",
        "clearsky",
        "gray",
        "allsky",
        "allskywithclear",
        "DYCOMS_RF01",
        "TRMM_LBA",
    )
    return if radiation_name == "clearsky"
        RRTMGPI.ClearSkyRadiation(
            idealized_h2o,
            idealized_insolation,
            idealized_clouds,
        )
    elseif radiation_name == "gray"
        RRTMGPI.GrayRadiation(
            idealized_h2o,
            idealized_insolation,
            idealized_clouds,
        )
    elseif radiation_name == "allsky"
        RRTMGPI.AllSkyRadiation(
            idealized_h2o,
            idealized_insolation,
            idealized_clouds,
        )
    elseif radiation_name == "allskywithclear"
        RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics(
            idealized_h2o,
            idealized_insolation,
            idealized_clouds,
        )
    elseif radiation_name == "DYCOMS_RF01"
        RadiationDYCOMS_RF01{FT}()
    elseif radiation_name == "TRMM_LBA"
        RadiationTRMM_LBA(FT)
    else
        nothing
    end
end

function get_precipitation_model(parsed_args)
    precip_model = parsed_args["precip_model"]
    return if precip_model == nothing || precip_model == "nothing"
        NoPrecipitation()
    elseif precip_model == "0M"
        Microphysics0Moment()
    elseif precip_model == "1M"
        Microphysics1Moment()
    else
        error("Invalid precip_model $(precip_model)")
    end
end

function get_forcing_type(parsed_args)
    forcing = parsed_args["forcing"]
    @assert forcing in (nothing, "held_suarez")
    return if forcing == nothing
        nothing
    elseif forcing == "held_suarez"
        HeldSuarezForcing()
    end
end

function get_subsidence_model(parsed_args, radiation_mode, FT)
    subsidence = parsed_args["subsidence"]
    subsidence == nothing && return nothing

    prof = if subsidence == "Bomex"
        APL.Bomex_subsidence(FT)
    elseif subsidence == "LifeCycleTan2018"
        APL.LifeCycleTan2018_subsidence(FT)
    elseif subsidence == "Rico"
        APL.Rico_subsidence(FT)
    elseif subsidence == "DYCOMS"
        @assert radiation_mode isa RadiationDYCOMS_RF01
        z -> -z * radiation_mode.divergence
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

function get_edmf_coriolis(parsed_args, ::Type{FT}) where {FT}
    edmf_coriolis = parsed_args["edmf_coriolis"]
    edmf_coriolis == nothing && return nothing
    (prof_u, prof_v) = if edmf_coriolis == "Bomex"
        (APL.Bomex_geostrophic_u(FT), z -> FT(0))
    elseif edmf_coriolis == "LifeCycleTan2018"
        (APL.LifeCycleTan2018_geostrophic_u(FT), z -> FT(0))
    elseif edmf_coriolis == "Rico"
        (APL.Rico_geostrophic_ug(FT), APL.Rico_geostrophic_vg(FT))
    elseif edmf_coriolis == "ARM_SGP"
        (z -> FT(10), z -> FT(0))
    elseif edmf_coriolis == "DYCOMS_RF01"
        (z -> FT(7), z -> FT(-5.5))
    elseif edmf_coriolis == "DYCOMS_RF02"
        (z -> FT(5), z -> FT(-5.5))
    elseif edmf_coriolis == "GABLS"
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
    coriolis_param = coriolis_params[edmf_coriolis]
    return EDMFCoriolis(prof_u, prof_v, coriolis_param)
end

function get_turbconv_model(
    FT,
    moisture_model,
    precip_model,
    parsed_args,
    turbconv_params,
)
    turbconv = parsed_args["turbconv"]
    @assert turbconv in (nothing, "edmf", "edmfx", "diagnostic_edmfx")

    return if turbconv == "edmf"
        TC.EDMFModel(
            FT,
            moisture_model,
            precip_model,
            parsed_args,
            turbconv_params,
        )
    elseif turbconv == "edmfx"
        N = turbconv_params.updraft_number
        TKE = parsed_args["prognostic_tke"]
        EDMFX{N, TKE}(turbconv_params.min_area)
    elseif turbconv == "diagnostic_edmfx"
        N = turbconv_params.updraft_number
        TKE = parsed_args["prognostic_tke"]
        DiagnosticEDMFX{N, TKE}(FT(0.1), turbconv_params.min_area)
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
    elseif entr_model == "ConstantCoefficient"
        ConstantCoefficientEntrainment()
    elseif entr_model == "ConstantCoefficientHarmonics"
        ConstantCoefficientHarmonicsEntrainment()
    elseif entr_model == "ConstantTimescale"
        ConstantTimescaleEntrainment()
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
    elseif detr_model == "ConstantCoefficient"
        ConstantCoefficientDetrainment()
    elseif detr_model == "ConstantCoefficientHarmonics"
        ConstantCoefficientHarmonicsDetrainment()
    else
        error("Invalid entr_model $(entr_model)")
    end
end

function get_surface_thermo_state_type(parsed_args)
    dict = Dict()
    dict["GCMSurfaceThermoState"] = GCMSurfaceThermoState()
    return dict[parsed_args["surface_thermo_state_type"]]
end
