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

@testset "Base-name uniqueness of configuration files" begin
    configs = String[]
    for (root, _, files) in walkdir(CA.config_path), f in files
        endswith(f, ".yml") && push!(configs, joinpath(root, f))
    end
    stems = map(f -> first(splitext(basename(f))), configs)
    # A shipped configuration whose base name appears once, so the assertions
    # below rest on `is_unique_basename` and not on the contents of `config/`.
    shipped = configs[findfirst(s -> count(==(s), stems) == 1, stems)]
    shipped_id = first(splitext(basename(shipped)))
    # A shipped file is unique with respect to itself.
    @test CA.is_unique_basename(shipped)
    @test CA.job_id_from_config_file(shipped) == shipped_id
    mktempdir() do dir
        # A file elsewhere that shares a shipped base name is not unique, and
        # its job id is disambiguated by the full path.
        clash = joinpath(dir, basename(shipped))
        touch(clash)
        @test !CA.is_unique_basename(clash)
        @test CA.job_id_from_config_file(clash) != shipped_id
        unique_file = joinpath(dir, "a_base_name_no_shipped_config_uses.yml")
        touch(unique_file)
        @test CA.is_unique_basename(unique_file)
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

@testset "`AtmosNumerics` upwinding validation" begin
    num = CA.AtmosNumerics(;
        energy_humidity_upwinding = :first_order,
        edmfx_mse_humidity_upwinding = :none,
    )
    @test num.energy_humidity_upwinding === Val(:first_order)
    @test num.edmfx_mse_humidity_upwinding === Val(:none)

    # Removed keys are rejected as unrecognized keywords.
    @test_throws MethodError CA.AtmosNumerics(;
        energy_q_tot_upwinding = :first_order,
    )
    @test_throws MethodError CA.AtmosNumerics(;
        edmfx_mse_q_tot_upwinding = :none,
    )

    # Updraft reconstructions accept only `:none` and `:first_order`.
    @test_throws ErrorException CA.AtmosNumerics(;
        edmfx_mse_humidity_upwinding = :vanleer_limiter,
    )
    @test_throws ErrorException CA.AtmosNumerics(;
        edmfx_tracer_upwinding = :third_order,
    )
end
