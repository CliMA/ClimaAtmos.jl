using Test

import Distributions

import ClimaAtmos

function percentile_bounds_mean_norm_distributions(
    low_percentile::FT,
    high_percentile::FT,
) where {FT <: Real}
    gauss_int(x) = -exp(-x * x / 2) / sqrt(2 * pi)
    xp_low = Distributions.quantile(Distributions.Normal(), low_percentile)
    xp_high = Distributions.quantile(Distributions.Normal(), high_percentile)
    return (gauss_int(xp_high) - gauss_int(xp_low)) / max(
        Distributions.cdf(Distributions.Normal(), xp_high) -
        Distributions.cdf(Distributions.Normal(), xp_low),
        eps(FT),
    )
end

@testset "Gauss quantile" begin
    for p in 0.0:0.1:1.0
        @test ClimaAtmos.gauss_quantile(p) ≈
              Distributions.quantile(Distributions.Normal(), p) rtol = 1e-3
    end

    for p in 0.0:0.1:1.0
        for q in 0.0:0.1:1.0
            @test ClimaAtmos.percentile_bounds_mean_norm(p, q) ≈
                  percentile_bounds_mean_norm_distributions(p, q) rtol = 1e-3
        end
    end
end

@testset "surface_mass_flux_coefficient and surface_mass_flux" begin
    # `surface_mass_flux_coefficient` returns the effective surface area
    # fraction a_s = a_s_max · w*³ / (w*³ + c_u·u*³), with
    # w*³ = max(z_i · ⟨w'b'⟩_s, 0). `surface_mass_flux` returns
    # F_surf = a_s · ρ · w*.
    for FT in (Float32, Float64)
        @testset "FT = $FT" begin
            z_i = FT(1000)
            ρ = FT(1.2)
            a_s_max = FT(0.1)
            c_u = FT(1)

            @testset "stable BL (⟨w'b'⟩_s ≤ 0)" begin
                for bf in FT[-1e-3, 0]
                    for ustar in FT[0.1, 0.5]
                        a_s = ClimaAtmos.surface_mass_flux_coefficient(
                            bf, z_i, ustar, a_s_max, c_u,
                        )
                        F = ClimaAtmos.surface_mass_flux(
                            bf, ρ, z_i, ustar, a_s_max, c_u,
                        )
                        @test a_s == FT(0)
                        @test F == FT(0)
                    end
                end
            end

            @testset "free-convection limit (w* ≫ u*)" begin
                bf = FT(1e-2)
                ustar = FT(1e-6)
                a_s = ClimaAtmos.surface_mass_flux_coefficient(
                    bf, z_i, ustar, a_s_max, c_u,
                )
                @test a_s ≈ a_s_max rtol = FT(1e-3)
                # F_surf = a_s · ρ · w* with w* = cbrt(z_i · bf)
                w_star = cbrt(z_i * bf)
                F = ClimaAtmos.surface_mass_flux(
                    bf, ρ, z_i, ustar, a_s_max, c_u,
                )
                @test F ≈ a_s_max * ρ * w_star rtol = FT(1e-3)
            end

            @testset "shear-dominated limit (u*³ ≫ w*³)" begin
                bf = FT(1e-5)   # gives w*³ = z_i · bf = 1e-2
                ustar = FT(1)   # u*³ = 1, so w*³/(w*³+c_u·u*³) ≈ 1e-2
                a_s = ClimaAtmos.surface_mass_flux_coefficient(
                    bf, z_i, ustar, a_s_max, c_u,
                )
                @test a_s < FT(0.02) * a_s_max
                F = ClimaAtmos.surface_mass_flux(
                    bf, ρ, z_i, ustar, a_s_max, c_u,
                )
                @test F > FT(0)
            end

            @testset "type stability" begin
                bf = FT(1e-3)
                ustar = FT(0.3)
                a_s = ClimaAtmos.surface_mass_flux_coefficient(
                    bf, z_i, ustar, a_s_max, c_u,
                )
                F = ClimaAtmos.surface_mass_flux(
                    bf, ρ, z_i, ustar, a_s_max, c_u,
                )
                @test a_s isa FT
                @test F isa FT
            end

            @testset "monotonicity in buoyancy_flux at fixed u*" begin
                ustar = FT(0.2)
                prev_a_s = FT(-1)
                prev_F = FT(-1)
                for bf in FT[1e-5, 1e-4, 1e-3, 1e-2]
                    a_s = ClimaAtmos.surface_mass_flux_coefficient(
                        bf, z_i, ustar, a_s_max, c_u,
                    )
                    F = ClimaAtmos.surface_mass_flux(
                        bf, ρ, z_i, ustar, a_s_max, c_u,
                    )
                    @test a_s > prev_a_s
                    @test F > prev_F
                    @test FT(0) ≤ a_s ≤ a_s_max
                    prev_a_s = a_s
                    prev_F = F
                end
            end

            @testset "monotonicity decreasing in u* at fixed buoyancy_flux" begin
                bf = FT(1e-3)
                prev_a_s = FT(2)  # > a_s_max so first iteration passes
                for ustar in FT[1e-3, 1e-2, 1e-1, 1]
                    a_s = ClimaAtmos.surface_mass_flux_coefficient(
                        bf, z_i, ustar, a_s_max, c_u,
                    )
                    @test a_s < prev_a_s
                    @test FT(0) ≤ a_s ≤ a_s_max
                    prev_a_s = a_s
                end
            end

            @testset "a_s scales linearly with a_s_max" begin
                bf = FT(1e-3)
                ustar = FT(0.3)
                a_s_1 = ClimaAtmos.surface_mass_flux_coefficient(
                    bf, z_i, ustar, FT(0.1), c_u,
                )
                a_s_2 = ClimaAtmos.surface_mass_flux_coefficient(
                    bf, z_i, ustar, FT(0.2), c_u,
                )
                @test a_s_2 ≈ 2 * a_s_1 rtol = FT(1e-6)
            end
        end
    end
end
