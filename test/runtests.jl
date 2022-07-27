using Test
using ClimaAtmos

#####
##### Run tests!
#####

group = get(ENV, "TEST_GROUP", :all) |> Symbol

@testset "ClimaAtmos" begin
    if group == :unit || group == :all
        @testset "Unit tests" begin
            include("test_domains.jl")
            include("test_models.jl")
        end
    end
end

# Make sure default driver config
# works with latest package versions
@testset "Test latest packages" begin
    ca_dir = joinpath(dirname(@__DIR__))
    include(joinpath(ca_dir, "examples", "hybrid", "cli_options.jl"))

    filename = joinpath(ca_dir, "examples", "hybrid", "driver.jl")
    dict = parsed_args_per_job_id(; trigger = "benchmark.jl")
    parsed_args = dict["perf_target_unthreaded"]
    parsed_args["z_elem"] = 10 # lower resolution
    parsed_args["h_elem"] = 4 # lower resolution
    parsed_args["nh_poly"] = 2 # lower resolution
    parsed_args["dt_rad"] = "2secs" # make sure to call RRTMGP
    include(filename)
end

nothing
