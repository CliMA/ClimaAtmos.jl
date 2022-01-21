using Test

using ClimaAtmos: Domains
using ClimaCore: Spaces

float_types = (Float32, Float64)

@testset "Domains" begin
    for FT in float_types
        domain = Domains.Column(FT, zlim = (-1, π), nelements = 2)
        @test domain.zlim == FT.((-1, π))
        @test domain.nelements == 2
        @test Domains.make_function_space(domain) isa Tuple{
            Spaces.CenterFiniteDifferenceSpace,
            Spaces.FaceFiniteDifferenceSpace,
        }

        domain = Domains.HybridPlane(
            FT,
            xlim = (-1, π),
            zlim = (-2, 2π),
            nelements = (2, 3),
            npolynomial = 4,
            xperiodic = false,
        )
        @test domain.xlim == FT.((-1, π))
        @test domain.zlim == FT.((-2, 2π))
        @test domain.nelements == (2, 3)
        @test domain.npolynomial == 4
        @test domain.xperiodic == false
        @test Domains.make_function_space(domain) isa Tuple{
            Spaces.CenterExtrudedFiniteDifferenceSpace,
            Spaces.FaceExtrudedFiniteDifferenceSpace,
        }

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
        @test Domains.make_function_space(domain) isa Tuple{
            Spaces.CenterExtrudedFiniteDifferenceSpace,
            Spaces.FaceExtrudedFiniteDifferenceSpace,
        }

        domain = Domains.SphericalShell(
            FT,
            radius = 100.0,
            height = 30.0,
            nelements = (6, 10),
            npolynomial = 3,
        )
        @test domain.radius == FT.(100)
        @test domain.height == FT.(30)
        @test domain.nelements == (6, 10)
        @test domain.npolynomial == 3
        @test Domains.make_function_space(domain) isa Tuple{
            Spaces.CenterExtrudedFiniteDifferenceSpace,
            Spaces.FaceExtrudedFiniteDifferenceSpace,
        }
    end
end
