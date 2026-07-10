using Test
import Logging
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
using ClimaCore: Spaces
using ClimaCore.CommonSpaces

@testset "Hyperdiffusion" begin
    FT = Float64

    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 4,
        z_min = 0,
        z_max = 1,
        radius = 6.371e6,
        h_elem = 30,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
    Y = (; c = ones(ᶜspace))
    h = Spaces.node_horizontal_length_scale(Spaces.horizontal_space(ᶜspace))

    coeff = 0.1857
    prandtl = 0.2
    div_factor = 5.0
    β = CA.HYPERDIFFUSION_MAX_WAVENUMBER_FACTOR
    F = max(div_factor, inv(prandtl))
    make(; dt_limit_safety = 0) = CA.Hyperdiffusion{FT}(;
        ν₄_vorticity_coeff = coeff,
        divergence_damping_factor = div_factor,
        prandtl_number = prandtl,
        dt_limit_safety,
    )
    limit = CA.hyperdiffusion_dt_limit(make(), h)

    @testset "coefficients with no limit set" begin
        out = CA.ν₄(make(), Y, limit / 10)
        @test out.ν₄_vorticity == coeff * h^3
        @test out.ν₄_scalar == coeff * h^3 / prandtl
    end

    @testset "warns above the limit when no safety is set" begin
        @test_logs (:warn, r"stability limit") CA.warn_if_hyperdiffusion_over_dt_limit(
            make(),
            Y,
            10 * limit,
        )
        @test_logs min_level = Logging.Warn CA.warn_if_hyperdiffusion_over_dt_limit(
            make(),
            Y,
            limit / 10,
        )
        @test_logs min_level = Logging.Warn CA.warn_if_hyperdiffusion_over_dt_limit(
            make(; dt_limit_safety = 2),
            Y,
            10 * limit,
        )
    end

    @testset "clamp inert below the limit" begin
        ref = CA.ν₄(make(), Y, limit / 10)
        out = CA.ν₄(make(; dt_limit_safety = 2), Y, limit / 10)
        @test out.ν₄_vorticity == ref.ν₄_vorticity
        @test out.ν₄_scalar == ref.ν₄_scalar
    end

    @testset "clamp active above the limit" begin
        S = 2.0
        dt = 10 * limit
        out = CA.ν₄(make(; dt_limit_safety = S), Y, dt)
        @test out.ν₄_vorticity ≈ min(coeff * h^3, 2 * h^4 / (F * β^4 * S * dt))
        @test out.ν₄_vorticity < coeff * h^3
        @test out.ν₄_scalar ≈ out.ν₄_vorticity / prandtl
        # monotone non-increasing in dt and in the safety factor
        @test CA.ν₄(make(; dt_limit_safety = S), Y, 2dt).ν₄_vorticity <=
              out.ν₄_vorticity
        @test CA.ν₄(make(; dt_limit_safety = 2S), Y, dt).ν₄_vorticity <=
              out.ν₄_vorticity
    end

    @testset "strongest coefficient meets the limit at safety * dt" begin
        S = 2.0
        dt = 10 * limit
        ν = CA.ν₄(make(; dt_limit_safety = S), Y, dt).ν₄_vorticity
        @test F * ν * (β / h)^4 * S * dt ≈ 2
    end

    @testset "dt-limit calibration" begin
        # 2 Δx / (F β⁴ coeff) ≈ 0.95 s at Δx = 113 m, coeff 0.1857, F = 5
        @test CA.hyperdiffusion_dt_limit(make(), 113.0) ≈ 0.95 atol = 0.01
    end
end
