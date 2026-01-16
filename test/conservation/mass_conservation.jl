#=
Mass conservation tests for ClimaAtmos.jl

TODO: Implement tests for:
- Global dry air mass conservation
- Total water mass conservation (vapor + liquid + ice + rain + snow)
- Tracer mass conservation
- Mass flux consistency at column boundaries
- Conservation under different advection schemes

Reference: Total atmospheric mass should be conserved to machine precision
in the absence of sources/sinks.
=#

using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

@testset "Mass conservation" begin
    @testset "Placeholder - dry air mass" begin
        @test_skip "Dry air mass conservation tests not yet implemented"
    end

    @testset "Placeholder - total water mass" begin
        @test_skip "Total water mass conservation tests not yet implemented"
    end

    @testset "Placeholder - tracer mass" begin
        @test_skip "Tracer mass conservation tests not yet implemented"
    end
end
