using Flux
import JLD2

function get_microphysics_model(parsed_args, params = nothing)
    model_name = parsed_args["microphysics_model"]
    @assert model_name in ("dry", "0M", "1M", "2M", "2MP3")
    if model_name == "dry"
        DryModel()
    elseif model_name == "0M"
        EquilibriumMicrophysics0M()
    elseif model_name == "1M"
        NonEquilibriumMicrophysics1M()
    elseif model_name == "2M"
        NonEquilibriumMicrophysics2M()
    elseif model_name == "2MP3"
        NonEquilibriumMicrophysics2MP3()
    end
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

function get_sfc_temperature_form(parsed_args)
    surface_temperature = parsed_args["surface_temperature"]
    @assert surface_temperature in (
        "ZonallySymmetric",
        "RCEMIPII",
        "ReanalysisTimeVarying",
    )
    return if surface_temperature == "ZonallySymmetric"
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
    elseif hyperdiff_name ∈ ("false", false, nothing)
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

"""
    get_smagorinsky_lilly_model(parsed_args)

Get the Smagorinsky-Lilly turbulence model based on `parsed_args["smagorinsky_lilly"]`

The possible model configurations flags are:
- `UVW`: Applies the model to all spatial directions.
- `UV`: Applies the model to the horizontal direction only.
- `W`: Applies the model to the vertical direction only.
- `UV_W`: Applies the model to the horizontal and vertical directions separately.
"""
function get_smagorinsky_lilly_model(parsed_args)
    smag = parsed_args["smagorinsky_lilly"]
    isnothing(smag) && return nothing
    return SmagorinskyLilly(; axes = Symbol(smag))
end

function get_amd_les_model(parsed_args, ::Type{FT}) where {FT}
    is_model_active = parsed_args["amd_les"]
    @assert is_model_active in (true, false)
    return is_model_active ? AnisotropicMinimumDissipation{FT}(parsed_args["c_amd"]) :
           nothing
end

function get_constant_horizontal_diffusion_model(parsed_args, params, ::Type{FT}) where {FT}
    is_model_active = parsed_args["constant_horizontal_diffusion"]
    @assert is_model_active in (true, false)
    return is_model_active ?
           ConstantHorizontalDiffusion{FT}(CAP.constant_horizontal_diffusion_D(params)) :
           nothing
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
    params,
    ::Type{FT},
) where {FT}
    nogw_name = parsed_args["non_orographic_gravity_wave"]
    @assert nogw_name in (true, false)
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
    @assert ogw_name in (nothing, "gfdl_restart", "raw_topo", "linear")
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
    else
        nothing
    end
end


"""
    get_sgs_distribution(parsed_args)

Parse the SGS distribution type from configuration.

# Config value mapping
- `"gaussian"` or `nothing` → `GaussianSGS()` (default)
- `"lognormal"` → `LogNormalSGS()`
- `"mean"` → `GridMeanSGS()` (grid-mean only, no SGS sampling)
"""
function get_sgs_distribution(parsed_args)
    dist_name = get(parsed_args, "sgs_distribution", "gaussian")
    return if dist_name in (nothing, "gaussian")
        GaussianSGS()
    elseif dist_name == "lognormal"
        LogNormalSGS()
    elseif dist_name == "mean"
        GridMeanSGS()
    else
        error("Invalid sgs_distribution $(dist_name). Use: gaussian, lognormal, mean")
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

function get_forcing_type(parsed_args)
    forcing = parsed_args["forcing"]
    @assert forcing in (nothing, "held_suarez")
    if forcing == "held_suarez"
        @warn "The 'held_suarez' forcing option is deprecated. Use rad='held_suarez' instead to set HeldSuarezForcing as a radiation mode."
        return HeldSuarezForcing()  # Still return the object for backward compatibility
    end
    return nothing
end


function get_subsidence_model(parsed_args, radiation_mode, FT)
    subsidence = parsed_args["subsidence"]
    isnothing(subsidence) && return nothing

    prof = if subsidence == "Bomex"
        APL.Bomex_subsidence(FT)
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
    elseif ls_adv == "Rico"
        (APL.Rico_dTdt(FT), APL.Rico_dqtdt(FT))
    else
        error("Uncaught case")
    end
    # See https://clima.github.io/AtmosphericProfilesLibrary.jl/dev/
    # for which functions accept which arguments.
    # TODO: do not assume dry air?
    prof_dqtdt = (thermo_params, ᶜp, t, z) -> prof_dqtdt₀(z)
    prof_dTdt =
        (thermo_params, ᶜp, t, z) ->
            prof_dTdt₀(TD.exner_given_pressure(thermo_params, ᶜp), z)

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
    elseif scm_coriolis == "Rico"
        (APL.Rico_geostrophic_ug(FT), APL.Rico_geostrophic_vg(FT))
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
    coriolis_params["Rico"] = FT(4.5e-5)
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

    n_updrafts = parsed_args["updraft_number"]
    prognostic_tke = parsed_args["prognostic_tke"]
    area_fraction = turbconv_params.min_area
    return if turbconv == "prognostic_edmfx"
        PrognosticEDMFX(; n_updrafts, prognostic_tke, area_fraction)
    elseif turbconv == "diagnostic_edmfx"
        DiagnosticEDMFX(; n_updrafts, prognostic_tke, area_fraction)
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

function get_tracers(parsed_args)
    aerosol_names = Tuple(parsed_args["prescribed_aerosols"])
    time_varying_trace_gas_names = Tuple(parsed_args["time_varying_trace_gases"])
    return (; aerosol_names, time_varying_trace_gas_names)
end

function check_case_consistency(parsed_args)
    # if any flags is ISDAC, check that all are ISDAC
    ic = parsed_args["initial_condition"]
    subs = parsed_args["subsidence"]
    surf = parsed_args["surface_setup"]
    rad = parsed_args["rad"]
    cor = parsed_args["scm_coriolis"]
    forc = parsed_args["forcing"]
    microphysics = parsed_args["microphysics_model"]
    ls_adv = parsed_args["ls_adv"]
    extf = parsed_args["external_forcing"]
    imp_vert_diff = parsed_args["implicit_diffusion"]
    vert_diff = parsed_args["vert_diff"]
    turbconv = parsed_args["turbconv"]
    topography = parsed_args["topography"]
    prescribed_flow = parsed_args["prescribed_flow"]

    ISDAC_mandatory = (ic, subs, surf, rad, extf)
    if "ISDAC" in ISDAC_mandatory
        @assert(
            allequal(ISDAC_mandatory) &&
            all(isnothing, (cor, forc, ls_adv)) &&
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
            !parsed_args["implicit_diffusion"] &&
            !parsed_args["implicit_sgs_advection"] &&
            !parsed_args["implicit_sgs_entr_detr"] &&
            !parsed_args["implicit_sgs_nh_pressure"] &&
            !parsed_args["implicit_sgs_vertdiff"] &&
            !parsed_args["implicit_sgs_mass_flux"],
            "Prescribed flow does not use the implicit solver."
        )
    end
end
