#=
Hyperdiffusion unit tests for ClimaAtmos.jl

TODO: Implement tests for:
- Scalar hyperdiffusion (∇⁴ operator)
- Vorticity hyperdiffusion
- Divergence damping

See src/prognostic_equations/hyperdiffusion.jl for the implementation.
=#

using Test
import Logging
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
using ClimaCore: Geometry, Spaces
using ClimaCore.CommonSpaces

# grid_scale_factor, for regenerating table entries in the suite.
include(joinpath(@__DIR__, "hyperdiffusion_grid_factor.jl"))

@testset "Hyperdiffusion" begin
    FT = Float64

    # Grids for the measured stability limits: a uniform box at Δx ≈ 113 m
    # (2720 m over 8 degree-3 elements) and a cubed sphere at h_elem 6. See #4673.
    box = RectangleXYSpace(
        FT;
        x_min = 0,
        x_max = 2720,
        y_min = 0,
        y_max = 2720,
        periodic_x = true,
        periodic_y = true,
        n_quad_points = 4,
        x_elem = 8,
        y_elem = 8,
    )
    sphere = CubedSphereSpace(FT; radius = 6.371e6, n_quad_points = 4, h_elem = 6)

    coeff = 0.1857
    prandtl = 0.2
    div_factor = 5.0
    C_reduce = CA.HYPERDIFFUSION_FORWARD_EULER_STABILITY
    make(;
        dt_safety_factor = 0,
        prandtl_number = prandtl,
        divergence_damping_factor = div_factor,
    ) = CA.Hyperdiffusion{FT}(;
        ν₄_vorticity_coeff = coeff,
        divergence_damping_factor,
        prandtl_number,
        dt_safety_factor,
    )

    @testset "grid factor per degree" begin
        # Uniform-grid biharmonic factor; see docs/src/equations.md.
        @test CA.hyperdiffusion_grid_scale_factor(3) ≈ 4.0637 atol = 1e-4
        @test issorted(CA.hyperdiffusion_grid_scale_factor.(2:7))
        # Regenerate the inexpensive degrees against the assembled operator;
        # the standalone generator covers degrees 2 to 7.
        for degree in 2:3
            @test CA.hyperdiffusion_grid_scale_factor(degree) ≈
                  grid_scale_factor(degree) atol = 1e-4
        end

        # Uniform box: metric factor 1, so β equals the degree-3 uniform factor.
        @test CA.hyperdiffusion_grid_factor(box) ≈
              CA.hyperdiffusion_grid_scale_factor(3) rtol = 1e-3
        # Cubed sphere: metric non-uniformity raises β above the uniform factor.
        @test CA.hyperdiffusion_grid_factor(sphere) >
              CA.hyperdiffusion_grid_scale_factor(3)
    end

    @testset "untabulated degree" begin
        @test isnothing(CA.hyperdiffusion_grid_scale_factor(8))
        box8 = RectangleXYSpace(
            FT;
            x_min = 0,
            x_max = 2720,
            y_min = 0,
            y_max = 2720,
            periodic_x = true,
            periodic_y = true,
            n_quad_points = 9,
            x_elem = 2,
            y_elem = 2,
        )
        @test isnothing(CA.hyperdiffusion_grid_factor(box8))
        # No warning without a safety factor; construction errors with one.
        @test_logs min_level = Logging.Warn CA.warn_if_hyperdiffusion_over_dt_limit(
            make(), (; c = ones(box8)), 1.0e6, nothing,
        )
        @test_throws ErrorException CA.hyperdiffusion_cache(
            (; c = ones(box8)), make(; dt_safety_factor = 2), nothing, nothing,
        )
    end

    @testset "unsupported horizontal space" begin
        # Column (point) and 1D plane horizontal spaces have no limit.
        point = Spaces.PointSpace(Geometry.XPoint(FT(0)))
        @test isnothing(CA.hyperdiffusion_grid_factor(point))
        slice = SliceXZSpace(
            FT;
            z_elem = 2,
            x_min = 0,
            x_max = 2720,
            z_min = 0,
            z_max = 1000,
            periodic_x = true,
            n_quad_points = 4,
            x_elem = 8,
            staggering = CellCenter(),
        )
        @test isnothing(
            CA.hyperdiffusion_grid_factor(Spaces.horizontal_space(slice)),
        )
        # No warning without a safety factor; construction errors with one.
        @test_logs min_level = Logging.Warn CA.warn_if_hyperdiffusion_over_dt_limit(
            make(), (; c = ones(slice)), 1.0e6, nothing,
        )
        @test_throws "2D spectral-element" CA.hyperdiffusion_cache(
            (; c = ones(slice)), make(; dt_safety_factor = 2), nothing, nothing,
        )
    end

    @testset "measured stability limits" begin
        h_box = Spaces.node_horizontal_length_scale(box)
        box_limit = CA.hyperdiffusion_dt_limit(
            make(), h_box, CA.hyperdiffusion_grid_factor(box), C_reduce,
        )
        # Box: measured stable at 0.9 s, unstable at 1.2 s at Δx = 113 m.
        @test 0.85 < box_limit < 1.2

        h_sph = Spaces.node_horizontal_length_scale(sphere)
        sph_limit = CA.hyperdiffusion_dt_limit(
            make(), h_sph, CA.hyperdiffusion_grid_factor(sphere), C_reduce,
        )
        # Sphere: measured limit for the unreduced coefficient within
        # [720, 1600] s at h_elem 6.
        @test 720 < sph_limit < 1600
    end

    @testset "coefficients with no limit set" begin
        Y = (; c = ones(box))
        h = Spaces.node_horizontal_length_scale(box)
        gf = CA.hyperdiffusion_grid_factor(box)
        out = CA.ν₄(make(), Y, 1.0, gf)
        @test out.ν₄_vorticity == coeff * h^3
        @test out.ν₄_scalar == coeff * h^3 / prandtl
    end

    @testset "coefficient reduction inert below the limit, active above" begin
        Y = (; c = ones(box))
        h = Spaces.node_horizontal_length_scale(box)
        gf = CA.hyperdiffusion_grid_factor(box)
        limit = CA.hyperdiffusion_dt_limit(make(), h, gf, C_reduce)
        S = 2.0

        ref = CA.ν₄(make(), Y, limit / 10, gf)
        inert = CA.ν₄(make(; dt_safety_factor = S), Y, limit / 10, gf)
        @test inert.ν₄_vorticity == ref.ν₄_vorticity

        dt = 10 * limit
        active = CA.ν₄(make(; dt_safety_factor = S), Y, dt, gf)
        @test active.ν₄_vorticity < coeff * h^3
        @test active.ν₄_scalar ≈ active.ν₄_vorticity / prandtl
        # The reduced coefficient reaches its own limit exactly at S * dt.
        F = max(div_factor, inv(prandtl), 1)
        @test F * active.ν₄_vorticity * (gf / h)^4 * S * dt ≈ C_reduce
        # Monotone non-increasing in dt and in the safety factor.
        @test CA.ν₄(make(; dt_safety_factor = S), Y, 2dt, gf).ν₄_vorticity <=
              active.ν₄_vorticity
        @test CA.ν₄(make(; dt_safety_factor = 2S), Y, dt, gf).ν₄_vorticity <=
              active.ν₄_vorticity
    end

    @testset "limit follows the largest coefficient factor" begin
        h = Spaces.node_horizontal_length_scale(box)
        gf = CA.hyperdiffusion_grid_factor(box)
        # F = max(divergence_damping_factor, 1 / prandtl_number, 1).
        # 1/Pr is the larger factor (Pr 0.2, div 1): F = 5.
        base = CA.hyperdiffusion_dt_limit(
            make(; prandtl_number = 0.2, divergence_damping_factor = 1.0), h, gf,
            C_reduce,
        )
        # Halving Pr (1/Pr = 10) halves the limit.
        @test CA.hyperdiffusion_dt_limit(
            make(; prandtl_number = 0.1, divergence_damping_factor = 1.0), h, gf, C_reduce,
        ) ≈ base / 2
        # A divergence factor above 1/Pr becomes the larger factor: F = 20.
        @test CA.hyperdiffusion_dt_limit(
            make(; prandtl_number = 0.2, divergence_damping_factor = 20.0), h, gf, C_reduce,
        ) ≈ base / 4
    end

    @testset "warns above the limit when no safety factor is set" begin
        Y = (; c = ones(box))
        h = Spaces.node_horizontal_length_scale(box)
        gf = CA.hyperdiffusion_grid_factor(box)
        # The warning uses the ARS343 explicit-tableau bound, not the reduction bound.
        limit = CA.hyperdiffusion_dt_limit(
            make(), h, gf, CA.HYPERDIFFUSION_ARS343_STABILITY,
        )
        @test_logs (:warn, r"stability limit") CA.warn_if_hyperdiffusion_over_dt_limit(
            make(), Y, 10 * limit, gf,
        )
        @test_logs min_level = Logging.Warn CA.warn_if_hyperdiffusion_over_dt_limit(
            make(), Y, limit / 10, gf,
        )
        @test_logs min_level = Logging.Warn CA.warn_if_hyperdiffusion_over_dt_limit(
            make(; dt_safety_factor = 2), Y, 10 * limit, gf,
        )
    end
end
