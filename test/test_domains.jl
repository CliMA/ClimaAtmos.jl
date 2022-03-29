if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(@__DIR__)))
end
using Test

using ClimaCore: Spaces, Meshes, Hypsography, Fields
using ClimaAtmos: Domains
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
        @test Domains.make_function_space(domain) isa Tuple{
            Spaces.CenterFiniteDifferenceSpace,
            Spaces.FaceFiniteDifferenceSpace,
        }

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
        @test Domains.make_function_space(domain) isa Tuple{
            Spaces.CenterExtrudedFiniteDifferenceSpace,
            Spaces.FaceExtrudedFiniteDifferenceSpace,
        }

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
        @test Domains.make_function_space(domain) isa Tuple{
            Spaces.CenterExtrudedFiniteDifferenceSpace,
            Spaces.FaceExtrudedFiniteDifferenceSpace,
        }

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
        @test Domains.make_function_space(domain) isa Tuple{
            Spaces.CenterExtrudedFiniteDifferenceSpace,
            Spaces.FaceExtrudedFiniteDifferenceSpace,
        }
    end
end

@testset "Meshes with Topography" begin
    # Mimics test in ClimaCore
    function warp_test_2d(coord)
        x = coord.x
        return sin(x) * eltype(x)(1 / 2)
    end
    function warp_test_3d(coord)
        x = coord.x
        y = coord.y
        return sin(x)^2 * cos(y)^2 * eltype(x)(1 / 2)
    end
    for FT in float_types
        # Extruded FD-Spectral Hybrid 2D
        domain = Domains.HybridPlane(
            FT,
            xlim = (-0, π),
            zlim = (0, 1),
            nelements = (8, 5),
            npolynomial = 4,
            xperiodic = true,
            topography = Domains.AnalyticalTopography(warp_test_2d),
        )
        cspace, fspace = Domains.make_function_space(domain)
        ᶜcoords = Fields.coordinate_field(cspace)
        z₀ = Fields.level(ᶜcoords.z, 1)
        # Check ∫ₓ(z_sfc)dx == known value from warp_test_2d
        @test sum(z₀ .- domain.zlim[2] / 5 / 2) - FT(1) <= FT(0.1 / 4 / 8 / 5)
        @test abs(maximum(z₀) - FT(0.5)) <= FT(0.125)
    end
    for FT in float_types
        # Extruded FD-Spectral Hybrid 3D
        levels = 5:10
        polynom = 2:4:10
        horzelem = 2:4:10
        for nl in levels, np in polynom, nh in horzelem
            domain = Domains.HybridBox(
                FT,
                xlim = (-0, π),
                ylim = (-0, π),
                zlim = (0, 1),
                nelements = (nh, nh, nl),
                npolynomial = np,
                xperiodic = true,
                topography = Domains.AnalyticalTopography(warp_test_3d),
            )
            cspace, fspace = Domains.make_function_space(domain)
            ᶜcoords = Fields.coordinate_field(cspace)
            z₀ = Fields.level(ᶜcoords.z, 1)
            # Check ∫ₛ(z_sfc)dS == known value from warp_test_3d
            # Assumes uniform stretching
            @test sum(z₀ .- domain.zlim[2] / 2 / domain.zlim[2]) -
                  FT(π^2 / 8) <= FT(0.1 / np * nh * nl)
            @test abs(maximum(z₀) - FT(0.5)) <= FT(0.125)
        end
    end
end
