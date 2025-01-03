using Test
import ClimaParams as CP
import ClimaAtmos as CA

FT = Float64

default_args = CA.cli_defaults(CA.argparse_settings())

@testset "Test types" begin

    config_dict = Dict(
        "krylov_rtol" => Float64(1),
        "newton_rtol" => Float32(1),
        "max_newton_iters_ode" => Int64(1),
        "nh_poly" => Int32(1),
        "dt_save_state_to_disk" => "str",
        "bubble" => true,
    )
    parsed_args =
        CA.AtmosConfig(config_dict, job_id = "paremter_tests1").parsed_args
    @test parsed_args["krylov_rtol"] isa FT
    @test parsed_args["newton_rtol"] isa FT
    @test parsed_args["max_newton_iters_ode"] isa Int
    @test parsed_args["nh_poly"] isa Int
    @test parsed_args["dt_save_state_to_disk"] isa String
    @test parsed_args["bubble"] isa Bool
end

@testset "Test all parameter tomls in toml/" begin
    toml_path = joinpath(pkgdir(CA), "toml")
    for (index, toml) in enumerate(readdir(toml_path))
        config_dict = Dict("toml" => [joinpath(toml_path, toml)])
        config =
            CA.AtmosConfig(config_dict, job_id = "paremter_tests$(index + 1)")
        # Ensure that there are no errors
        @test CA.ClimaAtmosParameters(config) isa
              CA.Parameters.ClimaAtmosParameters
    end
end

@testset "Test that `override_precip_timescale` is handled properly" begin
    # precipitation_timescale should NOT be overridden by DT
    config = CA.AtmosConfig(
        Dict("dt" => "1secs", "override_precip_timescale" => false),
    )
    @test config.parsed_args["override_precip_timescale"] == false
    @test config.parsed_args["dt"] == "1secs"
    (; precipitation_timescale) =
        CP.get_parameter_values(config.toml_dict, "precipitation_timescale")
    parameters = CA.ClimaAtmosParameters(config)
    @test parameters.microphysics_0m_params.τ_precip == precipitation_timescale

    # precipitation_timescale should be overridden by DT
    config = CA.AtmosConfig(Dict("dt" => "1secs"))
    @test config.parsed_args["override_precip_timescale"] == true
    @test config.parsed_args["dt"] == "1secs"
    (; precipitation_timescale) =
        CP.get_parameter_values(config.toml_dict, "precipitation_timescale")
    parameters = CA.ClimaAtmosParameters(config)
    @test parameters.microphysics_0m_params.τ_precip == 1.0
end
