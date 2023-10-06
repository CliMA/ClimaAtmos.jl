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
