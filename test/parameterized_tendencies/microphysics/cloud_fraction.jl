#=
Unit tests for cloud_fraction.jl

Tests cover:
1. compute_cloud_fraction_hybrid - hybrid CF using cached SGS moments
=#

using Test
using ClimaAtmos
import ClimaAtmos as CA

import Thermodynamics as TD
import ClimaParams as CP

@testset "Cloud Fraction" begin

    @testset "compute_cloud_fraction_hybrid" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                T = FT(280.0)
                ρ = FT(1.0)
                q_sat = TD.q_vap_saturation(thp, T, ρ)
                q_min = FT(1e-10)
                cf_guess = FT(1.0)
                coeff = FT(1)

                # Helper to build a moments NamedTuple of the right type.
                make_moments(mu_S, sigma_S_sq, M_l, M_i) =
                    (mu_S = FT(mu_S), sigma_S_sq = FT(sigma_S_sq),
                        M_l = FT(M_l), M_i = FT(M_i))

                for sgs_dist in (CA.GaussianSGS(), CA.LogNormalSGS())
                    @testset "No condensate → zero cloud fraction ($sgs_dist)" begin
                        # μ_S < 0 (subsaturated) and q_c = 0 → has_condensate
                        # gate fires regardless of moments.
                        moments = make_moments(-1e-3, 1e-8, 0, 0)
                        cf = CA.compute_cloud_fraction_hybrid(
                            thp, T, ρ, q_sat - FT(1e-3),
                            FT(0), FT(0), moments, cf_guess, coeff, q_min, sgs_dist,
                        )
                        @test cf == FT(0)
                    end

                    @testset "Liquid condensate, saturated mean → cf > 0 ($sgs_dist)" begin
                        excess = FT(1e-3)
                        q_tot = q_sat + excess
                        mu_S = sgs_dist isa CA.LogNormalSGS ?
                               log(q_tot / q_sat) : excess
                        moments = make_moments(mu_S, 1e-7, 1e-3, 0)
                        cf = CA.compute_cloud_fraction_hybrid(
                            thp, T, ρ, q_tot,
                            FT(1e-3), FT(0), moments, cf_guess, coeff, q_min, sgs_dist,
                        )
                        @test FT(0) < cf <= FT(1)
                    end

                    @testset "Zero variance, condensate present → cf ≈ 1 ($sgs_dist)" begin
                        excess = FT(1e-3)
                        q_tot = q_sat + excess
                        mu_S = sgs_dist isa CA.LogNormalSGS ?
                               log(q_tot / q_sat) : excess
                        moments = make_moments(mu_S, 0, excess, 0)
                        cf = CA.compute_cloud_fraction_hybrid(
                            thp, T, ρ, q_tot,
                            FT(1e-3), FT(0), moments, cf_guess, coeff, q_min, sgs_dist,
                        )
                        @test cf > FT(0.99)
                    end

                    @testset "Large condensate, small variance → cf ≈ 1 ($sgs_dist)" begin
                        excess = FT(1e-2)
                        q_tot = q_sat + excess
                        mu_S = sgs_dist isa CA.LogNormalSGS ?
                               log(q_tot / q_sat) : excess
                        moments = make_moments(mu_S, 1e-10, excess, 0)
                        cf = CA.compute_cloud_fraction_hybrid(
                            thp, T, ρ, q_tot,
                            FT(1e-2), FT(0), moments, cf_guess, coeff, q_min, sgs_dist,
                        )
                        @test cf > FT(0.99)
                    end

                    @testset "Ice condensate → positive cloud fraction ($sgs_dist)" begin
                        T_cold = FT(240.0)
                        q_sat_cold = TD.q_vap_saturation(thp, T_cold, ρ)
                        excess = FT(1e-4)
                        q_tot = q_sat_cold + excess
                        mu_S =
                            sgs_dist isa CA.LogNormalSGS ?
                            log(q_tot / q_sat_cold) : excess
                        moments = make_moments(mu_S, 1e-9, 0, 1e-3)
                        cf = CA.compute_cloud_fraction_hybrid(
                            thp, T_cold, ρ, q_tot,
                            FT(0), FT(1e-3), moments, cf_guess, coeff, q_min, sgs_dist,
                        )
                        @test FT(0) < cf <= FT(1)
                    end

                    @testset "Type stability ($sgs_dist)" begin
                        moments = make_moments(1e-3, 1e-7, 1e-3, 0)
                        cf = CA.compute_cloud_fraction_hybrid(
                            thp, T, ρ, q_sat + FT(1e-3),
                            FT(1e-3), FT(0), moments, cf_guess, coeff, q_min, sgs_dist,
                        )
                        @test cf isa FT
                    end

                    @testset "CF monotone in σ_S² at fixed (q_c, μ_S) ($sgs_dist)" begin
                        # Saturated mean (μ_S > 0): as σ_S² grows, cloud
                        # fraction approaches 0.5 from above (broader PDF
                        # spills more of its mass into the clear branch).
                        excess = FT(1e-3)
                        q_tot = q_sat + excess
                        mu_S = sgs_dist isa CA.LogNormalSGS ?
                               log(q_tot / q_sat) : excess
                        sig_vals = FT[1e-9, 1e-8, 1e-7, 1e-6]
                        cfs = [
                            CA.compute_cloud_fraction_hybrid(
                                thp, T, ρ, q_tot, FT(1e-3), FT(0),
                                make_moments(mu_S, σ², 1e-3, 0),
                                cf_guess, coeff, q_min, sgs_dist,
                            ) for σ² in sig_vals
                        ]
                        for i in 2:length(cfs)
                            @test cfs[i] <= cfs[i - 1] + FT(1e-6)
                        end
                    end
                end

                @testset "GridMeanSGS: degenerate (binary) limit" begin
                    # σ_S² → 0, μ_S < 0, q_c > 0 but |μ_S| > q_c → Q_eff < 0 → cf → 0.
                    moments = make_moments(-2e-3, 0, 0, 0)
                    cf_dry = CA.compute_cloud_fraction_hybrid(
                        thp, T, ρ, q_sat - FT(2e-3),
                        FT(1e-4), FT(0), moments, FT(0.1), coeff, q_min,
                        CA.GridMeanSGS(),
                    )
                    @test cf_dry < FT(0.05)

                    # Saturated, q_c > 0 → Q_eff > 0 → cf → 1.
                    moments = make_moments(1e-3, 0, 1e-3, 0)
                    cf_wet = CA.compute_cloud_fraction_hybrid(
                        thp, T, ρ, q_sat + FT(1e-3),
                        FT(1e-3), FT(0), moments, cf_guess, coeff, q_min,
                        CA.GridMeanSGS(),
                    )
                    @test cf_wet > FT(0.99)
                end

                @testset "Gaussian vs Lognormal agree in σ → 0 limit" begin
                    # At σ → 0, both distributions collapse to a single-point
                    # evaluation; with q_c equal to the linear excess the
                    # hybrid CF saturates at 1 in both coordinates.
                    excess = FT(2e-3)
                    q_tot = q_sat + excess
                    moments_g = make_moments(excess, 1e-12, excess, 0)
                    moments_ln = make_moments(log(q_tot / q_sat), 1e-12, excess, 0)
                    cf_g = CA.compute_cloud_fraction_hybrid(
                        thp, T, ρ, q_tot, excess, FT(0),
                        moments_g, cf_guess, coeff, q_min, CA.GaussianSGS(),
                    )
                    cf_ln = CA.compute_cloud_fraction_hybrid(
                        thp, T, ρ, q_tot, excess, FT(0),
                        moments_ln, cf_guess, coeff, q_min, CA.LogNormalSGS(),
                    )
                    @test cf_g > FT(0.99)
                    @test cf_ln > FT(0.99)
                end
            end
        end
    end

end
