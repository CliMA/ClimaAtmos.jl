#=
Unit tests for cloud_fraction.jl

Tests cover:
1. compute_cloud_fraction_sd - Sommeria-Deardorff moment-matching cloud fraction
=#

using Test
using ClimaAtmos
import ClimaAtmos as CA

import Thermodynamics as TD
import ClimaParams as CP

@testset "Cloud Fraction" begin

    @testset "compute_cloud_fraction_sd" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                T = FT(280.0)
                ρ = FT(1.0)
                # For tests we need a nominal q_tot. Let's assume q_tot = q_cond + 0.01
                q_tot_base = FT(0.01)
                q_min = FT(1e-10)

                for sgs_dist in (CA.GaussianSGS(), CA.LogNormalSGS())
                    @testset "No condensate → zero cloud fraction ($sgs_dist)" begin
                        cf = CA.compute_cloud_fraction_sd(
                            thp, T, ρ, q_tot_base,
                            FT(0), FT(0),              # q_liq, q_ice
                            FT(1.0), FT(1e-6), FT(0),  # T'T', q'q', T'q'
                            FT(1), q_min, sgs_dist,
                        )
                        @test cf == FT(0)
                    end

                    @testset "Liquid condensate → positive cloud fraction ($sgs_dist)" begin
                        cf = CA.compute_cloud_fraction_sd(
                            thp, T, ρ, q_tot_base + FT(1e-3),
                            FT(1e-3), FT(0),           # q_liq present
                            FT(1.0), FT(1e-6), FT(0),
                            FT(1), q_min, sgs_dist,
                        )
                        @test cf > FT(0)
                        @test cf <= FT(1)
                    end

                    @testset "Ice condensate → positive cloud fraction ($sgs_dist)" begin
                        T_cold = FT(240.0)
                        cf = CA.compute_cloud_fraction_sd(
                            thp, T_cold, ρ, q_tot_base + FT(1e-3),
                            FT(0), FT(1e-3),           # q_ice present
                            FT(1.0), FT(1e-6), FT(0),
                            FT(1), q_min, sgs_dist,
                        )
                        @test cf > FT(0)
                        @test cf <= FT(1)
                    end

                    @testset "Large condensate, small variance → cf ≈ 1 ($sgs_dist)" begin
                        cf = CA.compute_cloud_fraction_sd(
                            thp, T, ρ, q_tot_base + FT(1e-2),
                            FT(1e-2), FT(0),           # large q_liq
                            FT(0.01), FT(1e-10), FT(0),# small variance
                            FT(1), q_min, sgs_dist,
                        )
                        @test cf > FT(0.99)
                    end

                    @testset "Zero variance, nonzero condensate → cf = 1 ($sgs_dist)" begin
                        cf = CA.compute_cloud_fraction_sd(
                            thp, T, ρ, q_tot_base + FT(1e-3),
                            FT(1e-3), FT(0),
                            FT(0), FT(0), FT(0),
                            FT(1), q_min, sgs_dist,
                        )
                        @test cf > FT(0.99)
                    end

                    @testset "Both liquid and ice → max overlap ($sgs_dist)" begin
                        cf_both = CA.compute_cloud_fraction_sd(
                            thp, FT(260.0), ρ, q_tot_base + FT(1.5e-3),
                            FT(1e-3), FT(5e-4),
                            FT(1.0), FT(1e-6), FT(0),
                            FT(1), q_min, sgs_dist,
                        )
                        cf_liq = CA.compute_cloud_fraction_sd(
                            thp, FT(260.0), ρ, q_tot_base + FT(1.5e-3),
                            FT(1e-3), FT(0),
                            FT(1.0), FT(1e-6), FT(0),
                            FT(1), q_min, sgs_dist,
                        )
                        # Max overlap: combined cf ≥ liquid-only cf
                        @test cf_both >= cf_liq
                    end

                    @testset "Type stability ($sgs_dist)" begin
                        cf = CA.compute_cloud_fraction_sd(
                            thp, T, ρ, q_tot_base + FT(1e-3),
                            FT(1e-3), FT(0),
                            FT(1.0), FT(1e-6), FT(0),
                            FT(1), q_min, sgs_dist,
                        )
                        @test cf isa FT
                    end

                    @testset "Cloud fraction increases with T-q correlation ($sgs_dist)" begin
                        # Physical reasoning: positive corr(T',q') means T and q
                        # perturbations partially cancel in the saturation deficit
                        # s = q − q_sat(T).  The PDF width σ_s² = σ_q² + b²σ_T²
                        # − 2b·corr·σ_T·σ_q shrinks when corr increases (b > 0),
                        # so Q̂ = q_cond / σ_s grows ⇒ cf grows.
                        corr_vals = FT[-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]

                        # Liquid-only case
                        cf_liq = [
                            CA.compute_cloud_fraction_sd(
                                thp, T, ρ, q_tot_base + FT(1e-3),
                                FT(1e-3), FT(0),
                                FT(1.0), FT(1e-6), c,
                                FT(1), q_min, sgs_dist,
                            ) for c in corr_vals
                        ]
                        for i in 2:length(cf_liq)
                            @test cf_liq[i] > cf_liq[i - 1]
                        end

                        # Ice-only case (cold temperature)
                        T_cold = FT(240.0)
                        cf_ice = [
                            CA.compute_cloud_fraction_sd(
                                thp, T_cold, ρ, q_tot_base + FT(1e-3),
                                FT(0), FT(1e-3),
                                FT(1.0), FT(1e-6), c,
                                FT(1), q_min, sgs_dist,
                            ) for c in corr_vals
                        ]
                        for i in 2:length(cf_ice)
                            @test cf_ice[i] > cf_ice[i - 1]
                        end
                    end
                end

                @testset "LogNormalSGS: CF decreases as Cv increases (constant absolute condensate and q_tot)" begin
                    # Hold q_cond and q_tot constant. 
                    # Vary absolute variance q_var to vary sig_s, which varies Cv = sig_s / q_tot.
                    # Larger variance -> larger Cv -> more skewed tail & smaller Q_hat -> smaller CF.

                    q_cond = FT(1e-3)
                    qt = FT(0.01)

                    # Increasing q_var array to increase Cv and sig_s
                    q_var_vals = FT[1e-6, 1e-5, 1e-4, 1e-3]

                    cfs = [
                        CA.compute_cloud_fraction_sd(
                            thp, T, ρ, qt,
                            q_cond, FT(0),
                            FT(0), q_var, FT(0),
                            FT(1), q_min, CA.LogNormalSGS(),
                        ) for q_var in q_var_vals
                    ]

                    # As variance (and Cv) increases, CF should decrease
                    for i in 2:length(cfs)
                        @test cfs[i] < cfs[i - 1]
                    end
                end
            end
        end
    end

end
