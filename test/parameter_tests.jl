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

@testset "Geometric SGS variance parameters" begin
    # The defaults come from ClimaParams.
    for FT in (Float32, Float64)
        sq = CA.ClimaAtmosParameters(FT).sgs_quadrature_params
        @test CAP.sgs_variance_horizontal_scale_factor(sq) == FT(1)
        @test CAP.sgs_variance_vertical_scale_factor(sq) == FT(1)
        @test CAP.sgs_variance_geometric_coeff(sq) == FT(1 // 12)
        @test CAP.sgs_variance_max_rel_std(sq) == FT(0.5)
        @test CAP.sgs_liquid_uniform_fraction(sq) == FT(1)
        @test CAP.sgs_ice_uniform_fraction(sq) == FT(1)
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
        @test CAP.sgs_liquid_uniform_fraction(sq) == 1.0
    end
end

@testset "0M updraft precipitation parameters" begin
    m0 = CA.EquilibriumMicrophysics0M()
    for FT in (Float32, Float64)
        # Defaults reproduce the grid-mean 0M parameters exactly, without
        # adding the provisional keys to the caller's dictionary.
        toml_dict = CP.create_toml_dict(FT)
        params = CA.ClimaAtmosParameters(toml_dict; microphysics_model = m0)
        @test !haskey(toml_dict.data, "precipitation_timescale_updraft")
        @test !haskey(
            toml_dict.data,
            "supersaturation_precipitation_threshold_updraft",
        )
        base = CAP.microphysics_0m_params(params)
        up = CAP.microphysics_0m_updraft_params(params)
        @test up === params.microphysics_0m_updraft_params
        @test up isa CAP.CM.Parameters.Microphysics0MParams
        @test typeof(up) == typeof(base)
        @test up.precip.τ_precip === base.precip.τ_precip
        @test up.precip.S_0 === base.precip.S_0
        @test up.precip.qc_0 === base.precip.qc_0
        @test up == base
        @test up.precip.τ_precip > 0

        # Overrides (with `type = "float"`) are read, converted to FT,
        # logged as used, and leave the grid-mean parameters untouched.
        override = Dict(
            "precipitation_timescale_updraft" =>
                Dict("value" => 250, "type" => "float"),
            "supersaturation_precipitation_threshold_updraft" =>
                Dict("value" => 0.0, "type" => "float"),
        )
        toml_dict = CP.create_toml_dict(FT; override_file = override)
        params = CA.ClimaAtmosParameters(toml_dict; microphysics_model = m0)
        up = CAP.microphysics_0m_updraft_params(params)
        @test up.precip.τ_precip === FT(250)
        @test up.precip.S_0 === FT(0)
        @test up.precip.qc_0 === base.precip.qc_0
        @test CAP.microphysics_0m_params(params) == base
        for name in keys(override)
            @test "ClimaAtmos" in toml_dict.data[name]["used_in"]
        end

        # Only one key set: the other keeps its grid-mean default.
        toml_dict = CP.create_toml_dict(
            FT;
            override_file = Dict(
                "precipitation_timescale_updraft" =>
                    Dict("value" => 400.0, "type" => "float"),
            ),
        )
        up = CA.ClimaAtmosParameters(toml_dict).microphysics_0m_updraft_params
        @test up.precip.τ_precip === FT(400)
        @test up.precip.S_0 === base.precip.S_0

        # Set values are validated: finite positive timescale, finite
        # non-negative threshold.
        for (name, bad) in (
            ("precipitation_timescale_updraft", (0.0, -1.0, Inf, NaN)),
            (
                "supersaturation_precipitation_threshold_updraft",
                (-0.01, Inf, NaN),
            ),
        )
            for v in bad
                toml_dict = CP.create_toml_dict(
                    FT;
                    override_file = Dict(
                        name => Dict("value" => v, "type" => "float"),
                    ),
                )
                @test_throws ErrorException CA.ClimaAtmosParameters(toml_dict)
            end
        end

        # An entry without `type` (coupler TOML style) gives an error naming
        # the key, rather than ClimaParams' generic message.
        toml_dict = CP.create_toml_dict(
            FT;
            override_file = Dict(
                "precipitation_timescale_updraft" => Dict("value" => 300.0),
            ),
        )
        err = try
            CA.ClimaAtmosParameters(toml_dict)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("precipitation_timescale_updraft", err.msg)
        @test occursin("type = \"float\"", err.msg)

        # Not built (nor read, logged or validated) for other models, so the
        # keys show up as unused overrides there.
        for name in keys(override)
            toml_dict = CP.create_toml_dict(
                FT;
                override_file = Dict(
                    name => Dict("value" => -1.0, "type" => "float"),
                ),
            )
            params = CA.ClimaAtmosParameters(
                toml_dict;
                microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            )
            @test isnothing(CAP.microphysics_0m_params(params))
            @test isnothing(CAP.microphysics_0m_updraft_params(params))
            @test !haskey(toml_dict.data[name], "used_in")
        end
    end

    # The keys also reach the parameters through a run toml.
    mktemp() do path, io
        write(
            io,
            """
  [precipitation_timescale_updraft]
  value = 300.0
  type = "float"
  """,
        )
        flush(io)
        config = CA.AtmosConfig(
            Dict("toml" => [path], "microphysics_model" => "0M"),
            job_id = "parameter_test_0m_updraft",
        )
        params = CA.ClimaAtmosParameters(config)
        @test CAP.microphysics_0m_updraft_params(params).precip.τ_precip == 300
        @test CAP.microphysics_0m_params(params).precip.τ_precip != 300
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

@testset "0M precipitation threshold floor" begin
    m0 = CA.EquilibriumMicrophysics0M()
    for FT in (Float32, Float64)
        toml_dict = CP.create_toml_dict(FT)
        params = CA.ClimaAtmosParameters(toml_dict; microphysics_model = m0)
        @test CAP.precipitation_threshold_floor(params) === FT(0)
        @test CA.precipitation_q_vap_sat_min(params) === FT(0)
        @test !haskey(toml_dict.data, "precipitation_threshold_floor")

        mk(v) = CP.create_toml_dict(
            FT;
            override_file = Dict(
                "precipitation_threshold_floor" => Dict("value" => v, "type" => "float"),
            ),
        )
        toml_dict = mk(1e-5)
        params = CA.ClimaAtmosParameters(toml_dict; microphysics_model = m0)
        @test CAP.precipitation_threshold_floor(params) === FT(1e-5)
        S_0 = CAP.microphysics_0m_params(params).precip.S_0
        @test CA.precipitation_q_vap_sat_min(params) ≈ FT(1e-5) / S_0
        @test "ClimaAtmos" in toml_dict.data["precipitation_threshold_floor"]["used_in"]
        @test_throws ErrorException CA.ClimaAtmosParameters(mk(-1e-5); microphysics_model = m0)
        @test_throws ErrorException CA.ClimaAtmosParameters(
            CP.create_toml_dict(FT; override_file = Dict(
                "precipitation_threshold_floor" => Dict("value" => 1e-5))),
            microphysics_model = m0,
        )
    end
end
