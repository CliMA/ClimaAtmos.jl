#=
Unit tests for the SGS quadrature infrastructure.

Tests `_sgs_saturation_moments`, verifying:
  - μ_S equals the linearized analytic mean q_tot_mean − q_sat(T_mean, ρ)
  - σ_S is the linearized width `_sgs_linear_sigma_S`
  - σ_S ≥ ϵ_numerics in the zero-variance limit (T′T′ = q′q′ = 0)
  - σ_S is monotonically increasing with σ_q
  - grid-mean fallback: nothing and GridMeanSGS give the same result
=#

using Test
using ClimaAtmos
import Thermodynamics as TD
import ClimaParams as CP

const CA = ClimaAtmos

@testset "SGS Moments" begin

    @testset "μ_S and the linearized σ_S" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.2)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean + FT(2e-3)
                mu_S = q_tot_mean - q_sat_mean

                # `_sgs_saturation_moments` exposes μ_S = analytic value.
                mom = CA._sgs_saturation_moments(
                    thp, ρ, T_mean, q_tot_mean, nothing, FT(0), FT(0), FT(0),
                )
                @test mom.mu_S ≈ mu_S rtol = FT(1e-6)
                # with SGS sampling, σ_S is the linearized width
                quad = CA.SGSQuadrature(FT; quadrature_order = 3)
                (T′T′, q′q′, corr) = (FT(0.5), FT(4e-7), FT(0.4))
                mom = CA._sgs_saturation_moments(
                    thp, ρ, T_mean, q_tot_mean, quad, T′T′, q′q′, corr,
                )
                @test mom.sigma_S ==
                      CA._sgs_linear_sigma_S(thp, ρ, T_mean, T′T′, q′q′, corr)
            end
        end
    end

    @testset "_sgs_saturation_moments: zero-variance limit (σ_S → ϵ_numerics)" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.2)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean + FT(2e-3)

                for quad in (
                    CA.SGSQuadrature(
                        FT;
                        distribution = CA.GaussianSGS(),
                        quadrature_order = 3,
                    ),
                )
                    mom = CA._sgs_saturation_moments(
                        thp, ρ, T_mean, q_tot_mean, quad, FT(0), FT(0), FT(0),
                    )
                    @test mom.sigma_S ≥ 0
                    @test mom.sigma_S < FT(1e-10)
                end
            end
        end
    end

    @testset "_sgs_saturation_moments: σ_S increases monotonically with σ_q" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.2)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean - FT(1e-4)
                quad = CA.SGSQuadrature(
                    FT;
                    distribution = CA.GaussianSGS(),
                    quadrature_order = 3,
                )

                prev = FT(-1)
                for σ_q in FT[1e-4, 5e-4, 1e-3, 2e-3]
                    mom = CA._sgs_saturation_moments(
                        thp, ρ, T_mean, q_tot_mean, quad, FT(0), σ_q^2, FT(0),
                    )
                    @test mom.sigma_S ≥ 0
                    @test isfinite(mom.sigma_S)
                    @test mom.sigma_S > prev
                    prev = mom.sigma_S
                end
            end
        end
    end

    @testset "_sgs_saturation_moments: nothing / GridMeanSGS give the same result" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                ρ = FT(1.0)
                T_mean = FT(285.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean + FT(1e-3)

                m1 = CA._sgs_saturation_moments(thp, ρ, T_mean, q_tot_mean,
                    nothing, FT(1), FT(1e-6), FT(0))
                m2 = CA._sgs_saturation_moments(thp, ρ, T_mean, q_tot_mean,
                    CA.GridMeanSGS(), FT(1), FT(1e-6), FT(0))

                @test isfinite(m1.sigma_S)
                @test m1.sigma_S ≥ 0
                @test m2.sigma_S ≈ m1.sigma_S rtol = FT(1e-6)
            end
        end
    end
end
