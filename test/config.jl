using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

include(joinpath("..", "perf", "common.jl"))

function extract_job_ids(folder_path)
    job_id_dict = Dict()
    for (root, _, files) in walkdir(folder_path)
        for file in files
            filepath = joinpath(root, file)
            data = CA.load_yaml_file(filepath)
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

file, io = mktemp()
config_err = ErrorException("File $(CA.normrelpath(file)) is empty or missing.")
@test_throws config_err CA.AtmosConfig(file)

@testset "Check that entries in `default_config.yml` have `help` and `value` keys" begin
    config = CA.load_yaml_file(CA.default_config_file)
    missing_help = String[]
    missing_value = String[]
    for (key, value) in config
        !haskey(value, "help") && push!(missing_help, key)
        !haskey(value, "value") && push!(missing_value, key)
    end
    # Every key in the default config should have a `help` and `value` key
    # If not, these tests will fail, indicating which keys are missing
    @test isempty(missing_help)
    @test isempty(missing_value)
end
