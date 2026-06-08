using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

@testset "Check that every available config file has a unique `job_id`" begin
    all_job_ids = String[]
    for (root, _, files) in walkdir(CA.config_path), f in files
        file = joinpath(root, f)
        endswith(file, ".yml") || continue
        job_id = CA.job_id_from_config_file(file)
        @test !(job_id in all_job_ids)
        push!(all_job_ids, job_id)
    end
end

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
