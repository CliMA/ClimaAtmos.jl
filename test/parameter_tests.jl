using Test
import ClimaAtmos as CA
import ClimaParams as CP
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD

@testset "ClimaAtmosParameters Construction" begin
    for FT in (Float32, Float64)
        params = CA.ClimaAtmosParameters(FT)
        @test params isa CAP.ClimaAtmosParameters
        @test CAP.eltype(params) == FT

        # Verify sub-components types
        @test params.thermodynamics_params isa CAP.TD.Parameters.ThermodynamicsParameters
        @test params.turbconv_params isa CAP.TurbulenceConvectionParameters
        @test params.sgs_quadrature_params isa CAP.SGSQuadratureParameters
        @test params.microphysics_cloud_params isa NamedTuple
        @test params.microphysics_cloud_params.liquid isa CAP.CM.Parameters.CloudLiquid

        # Verify physical constants (sanity check for Earth, would need to be changed for other planets)
        # R_d for dry air ~ 287 J/kg/K
        @test CAP.R_d(params) ≈ FT(287.0) rtol = 0.01
        # Planet radius ~ 6371 km
        @test CAP.planet_radius(params) ≈ FT(6.371e6) rtol = 0.01
        # Gravity ~ 9.81 m/s^2
        @test CAP.grav(params) ≈ FT(9.81) rtol = 0.01
    end
end

@testset "1-moment microphysics process options" begin
    CMP = CAP.CM.Parameters

    # Defaults come from CloudMicrophysics, except where `default_config.yml`
    # picks another variant
    defaults = CA.NonEquilibriumMicrophysics1M()
    @test defaults.processes.rain_autoconversion isa CMP.Kessler1M
    @test defaults.processes.cloud_ice_formation isa
          CMP.TemperatureDependentIceNumber
    @test defaults.processes.snow_deposition_sublimation isa
          CMP.DepositionWithoutCloudCondensate
    # Unknown process names are rejected where they are written
    @test_throws Exception CA.NonEquilibriumMicrophysics1M(; not_a_process = 1)

    # Building the model directly and from an unmodified config agree
    default_config = CA.AtmosConfig(
        Dict("microphysics_model" => "1M"),
        job_id = "parameter_test_1m_defaults",
    )
    @test CA.get_microphysics_model(default_config.parsed_args) == defaults

    # Options set on the model select the parameters loaded for it
    model = CA.NonEquilibriumMicrophysics1M(;
        n_substeps = 3,
        rain_autoconversion = CMP.PrescribedNd(),
        cloud_ice_formation = CMP.TemperatureDependent(),
        rain_snow_accretion = nothing,
    )
    @test model.n_substeps == 3
    params = CA.ClimaAtmosParameters(Float64; microphysics_model = model)
    processes = params.microphysics_1m_params.processes
    @test processes.rain_autoconversion isa CMP.PrescribedNd
    # `cloud_ice_formation` lives on `microphysics_1m_implicit_params` when
    # `implicit_microphysics: true` (the default) and on `microphysics_1m_params`
    # otherwise; `cloud_formation_1m_params` returns whichever set carries it.
    @test CAP.cloud_formation_1m_params(
        params,
    ).processes.cloud_ice_formation isa CMP.TemperatureDependent
    @test isnothing(processes.rain_snow_accretion)
    @test isnothing(
        params.microphysics_1m_params.process_params.rain_snow_accretion,
    )

    # The YAML keys reach the same place
    config = CA.AtmosConfig(
        Dict(
            "microphysics_model" => "1M",
            "rain_autoconversion" => "PrescribedNd",
            "rain_snow_accretion" => nothing,
        ),
        job_id = "parameter_test_1m_options",
    )
    yaml_processes =
        CA.ClimaAtmosParameters(config).microphysics_1m_params.processes
    @test yaml_processes.rain_autoconversion isa CMP.PrescribedNd

    # Khairoutdinov and Kogan (2000) warm-rain options
    kk_config = CA.AtmosConfig(
        Dict(
            "microphysics_model" => "1M",
            "rain_autoconversion" => "KK2000",
            "cloud_liquid_rain_accretion" => "KK2000",
        );
        job_id = "kk2000_options",
    )
    kk_processes = CA.get_microphysics_model(kk_config.parsed_args).processes
    @test kk_processes.rain_autoconversion isa CMP.KK2000Autoconversion
    @test kk_processes.cloud_liquid_rain_accretion isa CMP.KK2000Accretion
    kk_mp = CA.Parameters.microphysics_1m_params(CA.ClimaAtmosParameters(kk_config))
    # the prescribed droplet number is the one the radiation effective radius uses
    @test kk_mp.process_params.rain_autoconversion.Nc ==
          CA.Parameters.microphysics_cloud_params(
        CA.ClimaAtmosParameters(kk_config),
    ).N_cloud_liquid_droplets
    @test isnothing(yaml_processes.rain_snow_accretion)
end

@testset "TKE dissipation coefficient derived from Ri_crit" begin
    for FT in (Float32, Float64)
        params = CA.ClimaAtmosParameters(FT)
        tc = params.turbconv_params
        # ClimaParams default: Ri_c = 0.25 (mixing_length_Ri_crit)
        @test CAP.Ri_crit(tc) == FT(0.25)
        # c_d is derived, not independent: c_d = c_m c_b / Ri_c
        @test CA.tke_dissipation_coefficient(tc) ==
              CAP.tke_ed_coeff(tc) * CAP.static_stab_coeff(tc) /
              CAP.Ri_crit(tc)
    end

    # A TOML override of mixing_length_Ri_crit propagates into the derived c_d
    mktemp() do path, io
        write(
            io,
            """
  [mixing_length_Ri_crit]
  value = 0.5
  type = "float"
  """,
        )
        flush(io)
        config_dict = Dict("toml" => [path])
        config = CA.AtmosConfig(config_dict, job_id = "parameter_test_ri_crit")
        params = CA.ClimaAtmosParameters(config)
        tc = params.turbconv_params
        @test CAP.Ri_crit(tc) == 0.5
        @test CA.tke_dissipation_coefficient(tc) ==
              CAP.tke_ed_coeff(tc) * CAP.static_stab_coeff(tc) / 0.5
    end
end

@testset "Geometric SGS variance parameters" begin
    # Defaults from ClimaParams (name_map) or the provisional block of
    # `SGSQuadratureParameters`.
    for FT in (Float32, Float64)
        sq = CA.ClimaAtmosParameters(FT).sgs_quadrature_params
        @test CAP.sgs_variance_horizontal_scale_factor(sq) == FT(1.24)
        @test CAP.sgs_variance_geometric_coeff(sq) == FT(1 // 12)
        @test CAP.sgs_variance_max_rel_std(sq) == FT(0.5)
        @test CAP.sgs_variance_vertical_velocity_floor(sq) == FT(1.08)
        @test CAP.sgs_variance_floor_reference_resolution(sq) == FT(1.2e5)
        @test CAP.sgs_variance_floor_resolution_exponent(sq) == FT(2 // 3)
        @test CAP.sgs_variance_floor_time_scale(sq) == FT(300)
        @test CAP.sgs_variance_theta_horizontal_scale_ratio(sq) == FT(0.88)
        @test CAP.sgs_variance_theta_vertical_scale_ratio(sq) == FT(0.38)
        @test CAP.sgs_variance_horizontal_cross_correlation(sq) == FT(0.74)
        @test CAP.sgs_liquid_uniform_fraction(sq) == FT(1)
        @test CAP.sgs_ice_uniform_fraction(sq) == FT(1)
        @test CAP.sgs_precip_incloud_fraction(sq) == FT(1)
        @test CAP.sgs_precip_overlap_decay(sq) == FT(1)
        @test CAP.sgs_precip_fraction_floor(sq) == FT(0.1)
        @test CAP.sgs_displacement_truncation(sq) == FT(3)
        @test CAP.cloud_fraction_eps_rel(sq) == FT(0)
    end
    @test_throws ArgumentError CA.SGSQuadratureParameters(
        Float64,
        (; sgs_displacement_truncation = 0.5),
    )
    @test_throws ArgumentError CA.SGSQuadratureParameters(
        Float64,
        (; cloud_fraction_eps_rel = -0.05),
    )
    @test CAP.edmf_tail_entr_fraction(CA.TurbulenceConvectionParameters(Float64)) ==
          0.5
    for f in (0.0, -0.1, 1.5)
        @test_throws ArgumentError CA.TurbulenceConvectionParameters(
            Float64,
            (; edmf_tail_entr_fraction = f),
        )
    end
    @test CAP.edmf_buoyancy_clear_env_weight(
        CA.TurbulenceConvectionParameters(Float64),
    ) == 0
    for f_ev in (-0.1, 1.1)
        @test_throws ArgumentError CA.TurbulenceConvectionParameters(
            Float64,
            (; edmf_buoyancy_clear_env_weight = f_ev),
        )
    end
    # A run toml overrides the defaults.
    mktemp() do path, io
        write(
            io,
            """
  [sgs_variance_horizontal_scale_factor]
  value = 2.0
  type = "float"
  [sgs_ice_uniform_fraction]
  value = 0.5
  type = "float"
  [cloud_fraction_eps_rel]
  value = 0.25
  type = "float"
  """,
        )
        flush(io)
        config = CA.AtmosConfig(
            Dict("toml" => [path]),
            job_id = "parameter_test_sgs_variance_scale_factor",
        )
        sq = CA.ClimaAtmosParameters(config).sgs_quadrature_params
        @test CAP.sgs_variance_horizontal_scale_factor(sq) == 2.0
        @test CAP.sgs_ice_uniform_fraction(sq) == 0.5
        @test CAP.cloud_fraction_eps_rel(sq) == 0.25
        @test CAP.sgs_liquid_uniform_fraction(sq) == 1.0
    end
    # The parameter log leaves out the superseded ClimaParams value of `c_Δx`,
    # so a rerun from the log keeps the code default, while an override equal
    # to the ClimaParams value still wins.
    c_Δx(toml_dict) = CAP.sgs_variance_horizontal_scale_factor(
        CA.SGSQuadratureParameters(toml_dict),
    )
    toml_dict = CP.create_toml_dict(Float64)
    @test c_Δx(toml_dict) == 1.24
    log_file = tempname() * ".toml"
    CP.log_parameter_information(toml_dict, log_file)
    @test c_Δx(CP.create_toml_dict(Float64; override_file = log_file)) == 1.24
    climaparams_c_Δx =
        toml_dict.data["sgs_variance_horizontal_scale_factor"]["value"]
    override = Dict(
        "sgs_variance_horizontal_scale_factor" =>
            Dict("value" => climaparams_c_Δx, "type" => "float"),
    )
    @test c_Δx(CP.create_toml_dict(Float64; override_file = override)) ==
          climaparams_c_Δx
end

@testset "0M updraft precipitation parameters" begin
    m0 = CA.EquilibriumMicrophysics0M()
    for FT in (Float32, Float64)
        # The ClimaParams defaults give the updrafts the grid-mean parameters.
        params = CA.ClimaAtmosParameters(
            CP.create_toml_dict(FT);
            microphysics_model = m0,
        )
        (; precip, precip_updraft) = CAP.microphysics_0m_params(params)
        @test precip_updraft == precip
        @test precip_updraft.τ_precip isa FT

        # The updraft timescale is the ratio times the grid-mean timescale;
        # the thresholds are shared.
        toml_dict = CP.create_toml_dict(
            FT;
            override_file = Dict(
                "precipitation_timescale_updraft_ratio" =>
                    Dict("value" => 0.25, "type" => "float"),
            ),
        )
        params = CA.ClimaAtmosParameters(toml_dict; microphysics_model = m0)
        mp0 = CAP.microphysics_0m_params(params)
        @test mp0.precip_updraft.τ_precip ≈ FT(0.25) * precip.τ_precip
        @test mp0.precip_updraft.S_0 === mp0.precip.S_0
        @test mp0.precip_updraft.qc_0 === mp0.precip.qc_0
        @test mp0.precip == precip
        @test "CloudMicrophysics" in
              toml_dict.data["precipitation_timescale_updraft_ratio"]["used_in"]
    end

    # The key also reaches the parameters through a run toml.
    mktemp() do path, io
        write(
            io,
            """
  [precipitation_timescale]
  value = 1000.0

  [precipitation_timescale_updraft_ratio]
  value = 0.3
  """,
        )
        flush(io)
        config = CA.AtmosConfig(
            Dict("toml" => [path], "microphysics_model" => "0M"),
            job_id = "parameter_test_0m_updraft",
        )
        mp0 = CAP.microphysics_0m_params(CA.ClimaAtmosParameters(config))
        @test mp0.precip.τ_precip == 1000
        @test mp0.precip_updraft.τ_precip ≈ 300
    end
end

@testset "AtmosConfig Parameter Overrides" begin
    # Test overriding a parameter via configuration logic
    # CA.AtmosConfig merges dicts into the parameters

    # Create a temporary TOML file for overrides
    mktemp() do path, io
        write(
            io,
            """
  [planet_radius]
  value = 1000.0
  type = "float"
  """,
        )
        flush(io)

        # Pass the TOML file to AtmosConfig
        # Note: We need to pass the path as a list in the "toml" key
        config_dict = Dict("toml" => [path])
        config = CA.AtmosConfig(config_dict, job_id = "parameter_test_override")

        params = CA.ClimaAtmosParameters(config)

        # Check if override worked
        @test CAP.planet_radius(params) == 1000.0

        # Check that other parameters remained default-like (sanity check)
        @test CAP.R_d(params) ≈ 287.0 rtol = 0.01
    end
end

@testset "TOML Integration" begin
    # Iterate over all TOML files in the package to ensure they load without error
    # This preserves the original test intent but cleanly
    toml_path = joinpath(pkgdir(CA), "toml")
    for (index, toml_file) in enumerate(readdir(toml_path))
        # Skip if not a .toml file
        endswith(toml_file, ".toml") || continue

        config_dict = Dict("toml" => [joinpath(toml_path, toml_file)])
        config = CA.AtmosConfig(config_dict, job_id = "parameter_test_toml_$(index)")

        @test CA.ClimaAtmosParameters(config) isa CAP.ClimaAtmosParameters
    end
end

@testset "0M precipitation evaporation parameters" begin
    m0 = CA.EquilibriumMicrophysics0M()
    override = Dict(
        "precipitation_evaporation_coefficient" => 2e-4,
        "precipitation_evaporation_rh_crit" => 0.85,
        "precipitation_evaporation_area_fraction" => 0.5,
        "precipitation_evaporation_flux_scale" => 4e-3,
        "precipitation_evaporation_exponent" => 0.6,
    )
    override_toml(FT) = CP.create_toml_dict(
        FT;
        override_file = Dict(
            k => Dict("value" => v, "type" => "float") for (k, v) in override
        ),
    )
    for FT in (Float32, Float64)
        # Option off (the default): no evaporation parameters are built and
        # none is read, even with a positive coefficient in the TOML.
        for toml_dict in (CP.create_toml_dict(FT), override_toml(FT))
            params =
                CA.ClimaAtmosParameters(toml_dict; microphysics_model = m0)
            @test isnothing(CAP.microphysics_0m_params(params).evaporation)
            @test isbits(CAP.microphysics_0m_params(params))
        end

        # Option on: the evaporation parameters come from the TOML.
        toml_dict = override_toml(FT)
        params = CA.ClimaAtmosParameters(
            toml_dict;
            microphysics_model = m0,
            precipitation_evaporation_0m = true,
        )
        evaporation = CAP.microphysics_0m_params(params).evaporation
        @test evaporation isa CAP.CM.Parameters.Evaporation0M{FT}
        @test isbits(CAP.microphysics_0m_params(params))
        @test evaporation.k_E === FT(2e-4)
        @test evaporation.RH_c === FT(0.85)
        @test evaporation.a_p === FT(0.5)
        @test evaporation.α₂ === FT(4e-3)
        @test evaporation.α₃ === FT(0.6)
        for name in keys(override)
            @test "CloudMicrophysics" in toml_dict.data[name]["used_in"]
        end

        # Option on with the ClimaParams defaults: the default coefficient is
        # positive, so no TOML entry is needed.
        params = CA.ClimaAtmosParameters(
            CP.create_toml_dict(FT);
            microphysics_model = m0,
            precipitation_evaporation_0m = true,
        )
        @test CAP.microphysics_0m_params(params).evaporation.k_E === FT(1.36e-4)

        # Option on with a coefficient that is not positive is an error.
        @test_throws ErrorException CA.ClimaAtmosParameters(
            CP.create_toml_dict(
                FT;
                override_file = Dict(
                    "precipitation_evaporation_coefficient" =>
                        Dict("value" => 0.0, "type" => "float"),
                ),
            );
            microphysics_model = m0,
            precipitation_evaporation_0m = true,
        )
        # The option is only valid with 0M microphysics.
        @test_throws ErrorException CA.ClimaAtmosParameters(
            override_toml(FT);
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            precipitation_evaporation_0m = true,
        )
    end

    # The option is read from the YAML config.
    function config_0m(option)
        mktemp() do path, io
            write(
                io,
                """
      [precipitation_evaporation_coefficient]
      value = 2e-4
      """,
            )
            flush(io)
            config = CA.AtmosConfig(
                Dict(
                    "toml" => [path],
                    "microphysics_model" => "0M",
                    "precipitation_evaporation_0m" => option,
                ),
                job_id = "parameter_test_0m_evaporation",
            )
            CAP.microphysics_0m_params(CA.ClimaAtmosParameters(config))
        end
    end
    @test isnothing(config_0m(nothing).evaporation)
    @test config_0m("Kessler").evaporation.k_E === Float32(2e-4)  # default FLOAT_TYPE
    @test_throws ErrorException config_0m("NotAnOption")
end

@testset "Subgrid precipitation shaft velocity and shallow-support weight" begin
    for FT in (Float32, Float64)
        sq = CA.ClimaAtmosParameters(FT).sgs_quadrature_params
        @test CAP.edmf_shallow_support_height_scale(sq) == zero(FT)

        # Datum-shift invariance and asymptotic limits of `shallow_support_weight`
        # and `shallow_support_fraction`.
        @test CA.shallow_support_weight(FT(0.5), FT(4000), FT(500), zero(FT)) ==
              FT(0.5)
        @test CA.shallow_support_weight(FT(0.5), FT(500), FT(500), FT(2000)) ==
              FT(0.5)
        w_sea = CA.shallow_support_weight(FT(0.5), FT(1500), zero(FT), FT(2000))
        w_mtn = CA.shallow_support_weight(FT(0.5), FT(3500), FT(2000), FT(2000))
        @test w_sea ≈ w_mtn atol = eps(FT)
        @test zero(FT) < w_sea < FT(0.5)

        @test CA.shallow_support_fraction(FT(0.5), FT(4000), FT(500), zero(FT)) ===
              FT(0.5)
        @test CA.shallow_support_fraction(FT(0.5), FT(500), FT(500), FT(2000)) ===
              FT(0.5)
        f_sea = CA.shallow_support_fraction(FT(0.5), FT(1500), zero(FT), FT(2000))
        f_mtn = CA.shallow_support_fraction(FT(0.5), FT(3500), FT(2000), FT(2000))
        @test f_sea ≈ f_mtn atol = eps(FT)
        @test FT(0.5) < f_sea < one(FT)

        # Subgrid shaft terminal velocity limits and superlinear enhancement.
        const_tv(_, ρ, q) = FT(5)
        power_tv(_, ρ, q) = FT(20) * q^FT(0.25)
        @test CA.shaft_weighted_terminal_velocity(
            const_tv, :q_rai, FT(1), FT(1e-4), FT(0.75), FT(0.2), FT(0.1),
        ) ≈ FT(5)
        @test CA.shaft_weighted_terminal_velocity(
            power_tv, :q_rai, FT(1), FT(1e-4), zero(FT), FT(0.2), FT(0.1),
        ) == power_tv(:q_rai, FT(1), FT(1e-4))
        @test CA.shaft_weighted_terminal_velocity(
            power_tv, :q_rai, FT(1), FT(1e-4), FT(0.75), one(FT), FT(0.1),
        ) ≈ power_tv(:q_rai, FT(1), FT(1e-4))
        @test CA.shaft_weighted_terminal_velocity(
            power_tv, :q_rai, FT(1), FT(1e-4), FT(0.75), FT(0.2), FT(0.1),
        ) > power_tv(:q_rai, FT(1), FT(1e-4))
    end
end

@testset "Estimated Inversion Strength inhomogeneity parameters and helpers" begin
    for FT in (Float32, Float64)
        sq = CA.ClimaAtmosParameters(FT).sgs_quadrature_params
        @test CAP.cloud_inhomogeneity_eis_coeff(sq) == zero(FT)

        # Marine inversion column has higher EIS than unstable tropical column.
        eis_sc = CA.compute_wb06_eis_from_bot_700(
            FT(101300),
            FT(290),
            FT(0.80),
            FT(15),
            FT(286),
            FT(3000),
        )
        eis_cu = CA.compute_wb06_eis_from_bot_700(
            FT(101300),
            FT(301),
            FT(0.80),
            FT(15),
            FT(282),
            FT(3100),
        )
        @test eis_sc > eis_cu
        @test CA.eis_inhomogeneity_exponent(eis_cu, FT(0.68)) == one(FT)
        @test FT(0.15) <=
              CA.eis_inhomogeneity_exponent(eis_sc, FT(0.68)) <
              one(FT)
    end
end

@testset "Free-tropospheric subsaturation dynamical entrainment parameters and limits" begin
    for FT in (Float32, Float64)
        params_default = CA.ClimaAtmosParameters(FT)
        thp = CAP.thermodynamics_params(params_default)
        tc_default = params_default.turbconv_params
        @test CAP.entr_subsat_coeff(tc_default) == zero(FT)
        @test CAP.entr_subsat_power(tc_default) == FT(2)
        @test CAP.entr_subsat_height_min(tc_default) == FT(0.2)
        @test CAP.entr_subsat_height_width(tc_default) == FT(0.1)

        tc_active = CA.TurbulenceConvectionParameters(
            FT,
            (;
                entr_subsat_coeff = FT(5),
                entr_subsat_power = FT(2),
                entr_subsat_height_min = FT(0.2),
                entr_subsat_height_width = FT(0.1),
            ),
        )
        z_sfc = zero(FT)
        p_c = FT(70000)
        ρ_c = FT(0.9)
        a_j = FT(0.02)
        w_j = FT(2)
        rh_j = one(FT)
        b_j = FT(0.01)
        w_0 = zero(FT)
        b_0 = zero(FT)
        tke = FT(0.5)

        # Surface guard: z <= z_sfc returns 0
        @test CA.entrainment_velocity_scale(
            thp,
            tc_active,
            z_sfc,
            z_sfc,
            p_c,
            ρ_c,
            a_j,
            w_j,
            rh_j,
            b_j,
            w_0,
            FT(0.35),
            b_0,
            tke,
            CA.InvZEntrainment(),
        ) == zero(FT)

        # MBL height (z = 1000 m, Π₅ < 0.2): identical to baseline for any RH⁰
        z_mbl = FT(1000)
        base_mbl = CA.entrainment_velocity_scale(
            thp,
            tc_default,
            z_mbl,
            z_sfc,
            p_c,
            ρ_c,
            a_j,
            w_j,
            rh_j,
            b_j,
            w_0,
            FT(0.25),
            b_0,
            tke,
            CA.InvZEntrainment(),
        )
        @test CA.entrainment_velocity_scale(
            thp,
            tc_active,
            z_mbl,
            z_sfc,
            p_c,
            ρ_c,
            a_j,
            w_j,
            rh_j,
            b_j,
            w_0,
            FT(0.25),
            b_0,
            tke,
            CA.InvZEntrainment(),
        ) == base_mbl

        # Free troposphere (z = 3500 m, Π₅ > 0.2): zero increment at RH⁰ >= 1, positive for RH⁰ < 1
        z_ft = FT(3500)
        base_ft = CA.entrainment_velocity_scale(
            thp,
            tc_default,
            z_ft,
            z_sfc,
            p_c,
            ρ_c,
            a_j,
            w_j,
            rh_j,
            b_j,
            w_0,
            FT(0.35),
            b_0,
            tke,
            CA.InvZEntrainment(),
        )
        @test CA.entrainment_velocity_scale(
            thp,
            tc_active,
            z_ft,
            z_sfc,
            p_c,
            ρ_c,
            a_j,
            w_j,
            rh_j,
            b_j,
            w_0,
            one(FT),
            b_0,
            tke,
            CA.InvZEntrainment(),
        ) == base_ft
        @test CA.entrainment_velocity_scale(
            thp,
            tc_active,
            z_ft,
            z_sfc,
            p_c,
            ρ_c,
            a_j,
            w_j,
            rh_j,
            b_j,
            w_0,
            FT(1.1),
            b_0,
            tke,
            CA.InvZEntrainment(),
        ) == base_ft

        g = FT(CAP.grav(params_default))
        ref_H = p_c / (ρ_c * g)
        Π₅ = z_ft / ref_H
        δΠ₅ = Π₅ - FT(0.2)
        G_z = δΠ₅^2 / (δΠ₅^2 + FT(0.1)^2)
        U_a = CA.upper_area_limiter_factor(a_j, tc_active)
        expected_ft =
            base_ft + U_a * (FT(5) / ref_H) * G_z * (one(FT) - FT(0.35))^2
        @test CA.entrainment_velocity_scale(
            thp,
            tc_active,
            z_ft,
            z_sfc,
            p_c,
            ρ_c,
            a_j,
            w_j,
            rh_j,
            b_j,
            w_0,
            FT(0.35),
            b_0,
            tke,
            CA.InvZEntrainment(),
        ) ≈ expected_ft
    end
end

@testset "Turbulent-entrainment, form-drag, and subsaturation-width parameters" begin
    for FT in (Float32, Float64)
        p_def = CA.ClimaAtmosParameters(FT)
        tc_def = CAP.turbconv_params(p_def)
        @test CAP.turb_entr_height_scale(tc_def) == zero(FT)
        @test CAP.pressure_drag_plume_radius_slope(tc_def) == zero(FT)
        @test CAP.pressure_drag_plume_radius_max(tc_def) == zero(FT)
        @test CAP.cloud_fraction_subsat_width_coeff(p_def) == one(FT)
        @test CAP.cloud_fraction_subsat_free_coeff(p_def) == one(FT)
        @test CAP.cloud_fraction_subsat_inversion_gamma(p_def) == zero(FT)
        @test CA.turbulent_entrainment(tc_def, FT(0.002), FT(5000), FT(0)) ===
              CA.turbulent_entrainment(tc_def, FT(0.002))

        mktemp() do path, io
            write(
                io,
                """
[turb_entr_height_scale]
type = "float"
value = 3500.0
[pressure_drag_plume_radius_slope]
type = "float"
value = 0.2
[pressure_drag_plume_radius_max]
type = "float"
value = 1000.0
[cloud_fraction_subsat_width_coeff]
type = "float"
value = 0.25
[cloud_fraction_subsat_free_coeff]
type = "float"
value = 0.7
[cloud_fraction_subsat_inversion_gamma]
type = "float"
value = 0.008
""",
            )
            close(io)
            p_act = CA.ClimaAtmosParameters(
                CP.create_toml_dict(FT; override_file = path),
            )
            tc_act = CAP.turbconv_params(p_act)
            @test CAP.turb_entr_height_scale(tc_act) == FT(3500)
            @test CAP.pressure_drag_plume_radius_slope(tc_act) == FT(0.2)
            @test CAP.pressure_drag_plume_radius_max(tc_act) == FT(1000)
            @test CAP.cloud_fraction_subsat_width_coeff(p_act) == FT(0.25)
            @test CAP.cloud_fraction_subsat_free_coeff(p_act) == FT(0.7)
            @test CAP.cloud_fraction_subsat_inversion_gamma(p_act) == FT(0.008)
            r0 = CA.turbulent_entrainment(tc_act, FT(0.002), FT(0), FT(0))
            r5 = CA.turbulent_entrainment(tc_act, FT(0.002), FT(5000), FT(0))
            @test r5 ≈ r0 * exp(-(FT(5000) / FT(3500))^2)
        end

        for bad_key in (
            "turb_entr_height_scale",
            "pressure_drag_plume_radius_slope",
            "pressure_drag_plume_radius_max",
            "cloud_fraction_subsat_width_coeff",
            "cloud_fraction_subsat_free_coeff",
            "cloud_fraction_subsat_inversion_gamma",
        )
            mktemp() do path, io
                write(io, "[$bad_key]\ntype = \"float\"\nvalue = -0.1\n")
                close(io)
                @test_throws ArgumentError CA.ClimaAtmosParameters(
                    CP.create_toml_dict(FT; override_file = path),
                )
            end
        end
    end
end
