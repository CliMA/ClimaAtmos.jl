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
    @test config.parsed_args["initial_condition"] == "DryBaroclinicWave"

    # Test that config_dict overrides `default_perf`
    config_dict = Dict("dt" => "50secs", "initial_condition" => "Bomex")
    config = AtmosCoveragePerfConfig(config_dict)
    @test config.parsed_args["dt"] == "50secs"
    @test config.parsed_args["initial_condition"] == "Bomex"
    @test config.parsed_args["regression_test"] == false
end

function extract_job_ids(folder_path)
    job_id_dict = Dict()
    for (root, _, files) in walkdir(folder_path)
        for file in files
            filepath = joinpath(root, file)
            data = YAML.load_file(filepath)
            if haskey(data, "job_id")
                job_id_dict[filepath] = data["job_id"]
            end
        end
    end
    return job_id_dict
end

file_to_job_id = extract_job_ids("config")
# Check that all jobs have a unique job_id
value_to_keys = Dict()
for (key, value) in file_to_job_id
    if haskey(value_to_keys, value)
        push!(value_to_keys[value], key)
    else
        value_to_keys[value] = [key]
    end
end
# Filter the keys that have more than one associated key
repeated_job_ids = filter(kv -> length(kv[2]) > 1, value_to_keys)

@test isempty(repeated_job_ids)
