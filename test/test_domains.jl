if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(@__DIR__)))
end
using Test

using ClimaAtmos: Domains
using ClimaCore: Spaces, Meshes
import ClimaCore.Meshes:
    StretchingRule,
    Uniform,
    ExponentialStretching,
    GeneralizedExponentialStretching

float_types = (Float32, Float64)

@testset "Domains" begin
    for FT in float_types
        ## Column domains
        domain = Domains.Column(
            FT,
            zlim = (-1, π),
            nelements = 2,
            stretching = Uniform(),
        )
        @test domain.zlim == FT.((-1, π))
        @test domain.nelements == 2
        @test domain.stretching isa Meshes.StretchingRule
        @test domain.stretching isa Uniform

        ## Hybrid plane domains
        domain = Domains.HybridPlane(
            FT,
            xlim = (-1, π),
            zlim = (-2, 2π),
            nelements = (2, 3),
            npolynomial = 4,
            xperiodic = false,
            stretching = ExponentialStretching{FT}(π),
        )
        @test domain.xlim == FT.((-1, π))
        @test domain.zlim == FT.((-2, 2π))
        @test domain.nelements == (2, 3)
        @test domain.npolynomial == 4
        @test domain.xperiodic == false
        @test domain.stretching isa Meshes.StretchingRule
        @test domain.stretching isa ExponentialStretching

        ## Hybrid box domains
        domain = Domains.HybridBox(
            FT,
            xlim = (-1, π),
            ylim = (-1, π),
            zlim = (-2, 2π),
            nelements = (2, 2, 3),
            npolynomial = 4,
            xperiodic = false,
            yperiodic = true,
        )
        @test domain.xlim == FT.((-1, π))
        @test domain.ylim == FT.((-1, π))
        @test domain.zlim == FT.((-2, 2π))
        @test domain.nelements == (2, 2, 3)
        @test domain.npolynomial == 4
        @test domain.xperiodic == false
        @test domain.yperiodic == true
        @test domain.stretching isa StretchingRule
        @test domain.stretching isa Uniform

        ## Spherical shell domain
        domain = Domains.SphericalShell(
            FT,
            radius = 100.0,
            height = 30.0,
            nelements = (6, 20),
            npolynomial = 3,
            stretching = GeneralizedExponentialStretching(FT(1), FT(2)),
        )
        @test domain.radius == FT.(100)
        @test domain.height == FT.(30)
        @test domain.nelements == (6, 20)
        @test domain.npolynomial == 3
        @test domain.stretching isa StretchingRule
        @test domain.stretching isa GeneralizedExponentialStretching
    end
end
