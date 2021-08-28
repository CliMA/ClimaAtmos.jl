using Test

using ClimaAtmos

float_types = (Float32, Float64)

#####
##### Run tests!
#####

group = get(ENV, "TEST_GROUP", :all) |> Symbol

@testset "ClimaAtmos" begin
    if group == :unit || group == :all
        @testset "Unit tests" begin
            include("test_domains.jl")
            include("test_shallow_water.jl")
        end
    end

    # if group == :regression || group == :all
    #     @testset "Regression" begin
    #         include("test_regression.jl")
    #     end    end

    # if group == :validation || group == :all
    #     @testset "Validation" begin
    #         include("test_validation.jl")
    #     end
    # end

end
