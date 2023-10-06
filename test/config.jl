using Test
import ClimaAtmos as CA

include(joinpath("..", "perf", "common.jl"))

@testset "AtmosCoveragePerfConfig" begin
    # Just default config overridden by `default_perf.yml`
    config = AtmosCoveragePerfConfig()
    # Test defaults overridden by `default_perf`
    @test config.parsed_args["dt"] == "1secs"
    # Test defaults not overridden by `default_perf`
    @test config.parsed_args["y_elem"] == 6

    # Test with `target_job`
    config = AtmosCoveragePerfConfig(
        CA.config_from_target_job("sphere_baroclinic_wave_rhoe"),
    )
    # Target job config overridden by `default_perf`
    @test config.parsed_args["dt"] == "400secs"
    # Target job config not overridden by `default_perf`
    @test config.parsed_args["regression_test"] == true

    # Test that config_dict overrides `default_perf`
    config_dict = Dict("dt" => "50secs", "turbconv_case" => "GABLS")
    config = AtmosCoveragePerfConfig(config_dict)
    @test config.parsed_args["dt"] == "50secs"
    @test config.parsed_args["turbconv_case"] == "GABLS"
    @test config.parsed_args["regression_test"] == false
end

@testset "Access TOML outside pkgdir(CA)" begin
    config_file = joinpath(pkgdir(CA), "test", "config_test.yml")
    config_dict = YAML.load_file(config_file)

    # Test AtmosConfig with TOML from outside pkgdir(CA)
    # .buildkite is chosen because we can safely assume it will exist
    cd(joinpath(pkgdir(CA), ".buildkite"))
    config = CA.AtmosConfig(config_dict)
    @test config.toml_dict["C_E"]["value"] == 99

    # Test TargetJobConfig 
    # If flame_perf_gw is removed, replace it with another job id with a toml file
    job_id = "flame_perf_gw"
    config = TargetJobConfig(job_id)
    @test config.toml_dict["zd_rayleigh"]["value"] == 30000.0

    # Check that base dir still works
    cd(pkgdir(CA))
    config = CA.AtmosConfig(config_dict)
    @test config.toml_dict["C_E"]["value"] == 99

    # Reset working dir for next tests
    cd(joinpath(pkgdir(CA), "test"))
end
