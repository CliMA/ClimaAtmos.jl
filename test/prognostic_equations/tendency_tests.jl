#=
Core tendency computation unit tests for ClimaAtmos.jl

TODO: Implement tests for:
- Tendency accumulation correctness
- Individual tendency term verification
- Tendency budget closure (sum of parts = total)
- NaN/Inf detection in tendencies
- Performance regression tests

See src/prognostic_equations/remaining_tendency.jl for the implementation.
=#

using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

@testset "Tendency computations" begin
    @testset "Placeholder - tendency accumulation" begin
        @test_skip "Tendency accumulation tests not yet implemented"
    end

    @testset "Placeholder - tendency budget closure" begin
        @test_skip "Tendency budget closure tests not yet implemented"
    end

    @testset "Placeholder - NaN detection" begin
        @test_skip "NaN detection tests not yet implemented"
    end
end
