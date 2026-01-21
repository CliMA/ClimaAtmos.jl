#=
Hyperdiffusion unit tests for ClimaAtmos.jl

TODO: Implement tests for:
- Scalar hyperdiffusion (∇⁴ operator)
- Vorticity hyperdiffusion
- Divergence damping
- Coefficient scaling with resolution
- Stability at high wavenumbers

See src/prognostic_equations/hyperdiffusion.jl for the implementation.
=#

using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

@testset "Hyperdiffusion" begin
    @testset "Placeholder - scalar hyperdiffusion" begin
        @test_skip "Scalar hyperdiffusion tests not yet implemented"
    end

    @testset "Placeholder - vorticity hyperdiffusion" begin
        @test_skip "Vorticity hyperdiffusion tests not yet implemented"
    end

    @testset "Placeholder - divergence damping" begin
        @test_skip "Divergence damping tests not yet implemented"
    end
end
