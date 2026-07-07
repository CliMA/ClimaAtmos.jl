#=
Advection operator unit tests for ClimaAtmos.jl

TODO: Implement tests for:
- Horizontal advection operators
- Vertical advection with upwinding schemes
- Advection of known analytical functions (e.g., Gaussian blob)
- Convergence rate verification with mesh refinement
- Boundary condition handling

See src/prognostic_equations/advection.jl for the implementation.
=#

using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

@testset "Advection operators" begin
    @testset "Placeholder - horizontal advection" begin
        @test_skip "Horizontal advection tests not yet implemented"
    end

    @testset "Placeholder - vertical advection" begin
        @test_skip "Vertical advection tests not yet implemented"
    end

    @testset "Placeholder - advection convergence" begin
        @test_skip "Advection convergence tests not yet implemented"
    end
end
