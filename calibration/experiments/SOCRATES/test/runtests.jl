using Test

@testset "SOCRATES experiment tests" begin
    include("forcing_conversion_tests.jl")
    include("observation_map_tests.jl")
    include("calibration_pipeline_offline_tests.jl")
    include("ekp_failure_gating_tests.jl")
end
