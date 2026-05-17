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
        n_substeps = parsed_args["microphysics_n_substeps"]
        n_substeps_quad = parsed_args["microphysics_n_substeps_quadrature"]
        NonEquilibriumMicrophysics1M(; n_substeps, n_substeps_quad)
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
    α_max = isnothing(params) ? FT(1) : FT(CAP.water_filling_max_alpha(params))
    return SGSQuadrature(FT; quadrature_order, distribution, T_min, q_max, α_max)
end

function get_insolation_form(parsed_args; setup_type = nothing)
    if !isnothing(setup_type)
        model = Setups.insolation_model(setup_type)
        !isnothing(model) && return model
    end
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
    elseif isnothing(hyperdiff_name)
        return nothing
    else
        error(
            "Uncaught hyperdiff `$hyperdiff_name`. Expected: ~ | \"Hyperdiffusion\" | \"CAM_SE\".",
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
            "Uncaught vert_diff `$vert_diff_name`. Expected: ~ | \"VerticalDiffusion\" | \"DecayWithHeightDiffusion\".",
        )
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

function get_radiation_mode(parsed_args, ::Type{FT}; setup_type = nothing) where {FT}
    radiation_name = parsed_args["rad"]
    # Use setup default only when config doesn't explicitly set rad
    if isnothing(radiation_name) && !isnothing(setup_type)
        model = Setups.radiation_model(setup_type, FT)
        !isnothing(model) && return model
    end
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
    deep_atmosphere = parsed_args["deep_atmosphere"]
    @assert radiation_name in (
        nothing,
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
    @assert external_forcing in (
        nothing,
        "GCM",
        "ReanalysisTimeVarying",
        "ReanalysisMonthlyAveragedDiurnal",
        "ISDAC",
    )
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
    end
end

function get_scm_coriolis(::Type{FT}; setup_type = nothing) where {FT}
    isnothing(setup_type) && return nothing
    return Setups.coriolis_forcing(setup_type, FT)
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
    ic = parsed_args["initial_condition"]
    surf = parsed_args["surface_setup"]
    rad = parsed_args["rad"]
    forc = parsed_args["forcing"]
    microphysics = parsed_args["microphysics_model"]
    extf = parsed_args["external_forcing"]
    imp_vert_diff = parsed_args["implicit_diffusion"]
    vert_diff = parsed_args["vert_diff"]
    turbconv = parsed_args["turbconv"]
    topography = parsed_args["topography"]
    prescribed_flow = parsed_args["prescribed_flow"]

    # ISDAC consistency: when initial_condition is ISDAC, surface/rad/external
    # forcing must all be set to the matching ISDAC variants. Subsidence,
    # scm_coriolis, and ls_adv are owned by the setup, not the YAML schema.
    ISDAC_mandatory = (ic, surf, rad, extf)
    if "ISDAC" in ISDAC_mandatory
        @assert(
            allequal(ISDAC_mandatory) &&
            isnothing(forc) &&
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
