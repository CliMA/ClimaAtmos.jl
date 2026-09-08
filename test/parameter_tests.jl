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
    @test defaults.processes.cloud_ice_formation isa CMP.PrescribedIceNumber
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
    @test processes.cloud_ice_formation isa CMP.TemperatureDependent
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
