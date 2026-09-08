#=
Unit tests for microphysics_wrappers.jl
Tests wrapper functions for physical correctness, sign convention, and type stability.

Sign convention: all microphysics tendencies representing SINKS should be ≤ 0.
=#

using Test
using ClimaAtmos

import Thermodynamics as TD
import CloudMicrophysics as CM
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.Microphysics0M as CM0
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

# Import limiters
import ClimaAtmos: limit_sink

const CA = ClimaAtmos

# Functors returning one component of the conditioned point state, so the
# assignment can be integrated over the quadrature (structs cannot be declared
# inside a testset).
struct ConditionedRainProbe{E}
    evaluator::E
end
(p::ConditionedRainProbe)(T_hat, q_hat) =
    CA._conditioned_point_state(p.evaluator, T_hat, q_hat).q_rai

struct ConditionedSnowProbe{E}
    evaluator::E
end
(p::ConditionedSnowProbe)(T_hat, q_hat) =
    CA._conditioned_point_state(p.evaluator, T_hat, q_hat).q_sno

# Vapor the point presents to CloudMicrophysics, which subtracts rather than
# receives it: q_v = q_tot − (q_lcl + q_rai) − (q_icl + q_sno).
diagnosed_q_vap(state) =
    state.q_tot - state.q_lcl - state.q_icl - state.q_rai - state.q_sno

@testset "Microphysics Wrappers" begin

    @testset "BMT 0M sign convention" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics0MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                dt = FT(60.0)  # 1 minute timestep

                @testset "dq_tot_dt is always ≤ 0 (sink)" begin
                    # Condensate present → precipitation removes water (sink)
                    T = FT(280.0)
                    ρ = FT(1.0)
                    q_liq = FT(0.001)
                    q_ice = FT(0.0005)

                    # 3-arg form (condensate threshold)
                    result = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, T, q_liq, q_ice,
                    )
                    @test result <= FT(0)
                    @test isfinite(result)

                    # 4-arg form (supersaturation threshold)
                    q_vap_sat = TD.q_vap_saturation(thp, T, ρ)
                    result_sat = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, T, q_liq, q_ice, q_vap_sat,
                    )
                    @test result_sat <= FT(0)
                    @test isfinite(result_sat)
                end

                @testset "dq_tot_dt is zero when no condensate" begin
                    result = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, FT(280.0), FT(0), FT(0),
                    )
                    @test result == FT(0)
                end

                @testset "limit_sink preserves sign from BMT" begin
                    T = FT(280.0)
                    q_tot = FT(0.015)
                    q_liq = FT(0.001)
                    q_ice = FT(0.0005)

                    bmt_result = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, T, q_liq, q_ice,
                    )
                    limited = limit_sink(bmt_result, q_tot, dt, 1)

                    # limit_sink should keep the tendency negative (sink)
                    @test limited <= FT(0)

                    # Should not remove more water than available
                    @test limited * dt >= -q_tot

                    # Should be finite
                    @test isfinite(limited)
                end

                @testset "limit_sink with tiny q_tot" begin
                    # Edge case: very small q_tot should limit the magnitude
                    T = FT(280.0)
                    q_tot = FT(1e-6)   # Very small amount
                    q_liq = FT(0.01)   # Large condensate (edge case)
                    q_ice = FT(0)

                    bmt_result = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, T, q_liq, q_ice,
                    )
                    limited = limit_sink(bmt_result, q_tot, dt, 1)

                    # Should still be a sink
                    @test limited <= FT(0)

                    # Should be limited to available water
                    @test limited * dt >= -q_tot
                end

                @testset "type stability" begin
                    result = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, FT(280.0), FT(0.001), FT(0.0005),
                    )
                    @test typeof(result) == FT

                    # 4-arg form
                    q_vap_sat = TD.q_vap_saturation(thp, FT(280.0), FT(1.0))
                    result_sat = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        mp, thp, FT(280.0), FT(0.001), FT(0.0005), q_vap_sat,
                    )
                    @test typeof(result_sat) == FT

                    limited = limit_sink(result, FT(0.01), FT(60.0), 1)
                    @test typeof(limited) == FT
                end
            end
        end
    end

    @testset "BMT 1M sign convention" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict;
                    rain_autoconversion = CMP.PrescribedNd(toml_dict),
                )
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                ρ = FT(1.0)
                T = FT(280.0)
                q_tot = FT(0.015)
                q_liq = FT(0.001)
                q_ice = FT(0.0005)
                q_rai = FT(0.0001)
                q_sno = FT(0.00005)
                dt = FT(60.0)

                result = BMT.bulk_microphysics_tendencies(
                    BMT.LinearizedAverage(),
                    BMT.Microphysics1Moment(),
                    mp, thp, ρ, T,
                    q_tot, q_liq, q_ice, q_rai, q_sno, dt,
                )

                @testset "return type" begin
                    @test haskey(result, :dq_lcl_dt)
                    @test haskey(result, :dq_icl_dt)
                    @test haskey(result, :dq_rai_dt)
                    @test haskey(result, :dq_sno_dt)
                end

                @testset "finite values" begin
                    @test isfinite(result.dq_lcl_dt)
                    @test isfinite(result.dq_icl_dt)
                    @test isfinite(result.dq_rai_dt)
                    @test isfinite(result.dq_sno_dt)
                end

                @testset "type stability" begin
                    @test typeof(result.dq_lcl_dt) == FT
                    @test typeof(result.dq_icl_dt) == FT
                    @test typeof(result.dq_rai_dt) == FT
                    @test typeof(result.dq_sno_dt) == FT
                end
            end
        end
    end

    @testset "e_tot_0M_precipitation_sources_helper" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                @testset "Warm conditions (all liquid)" begin
                    T = FT(290.0)
                    q_liq = FT(0.001)
                    q_ice = FT(0)
                    Φ = FT(1000.0)

                    energy = ClimaAtmos.e_tot_0M_precipitation_sources_helper(
                        thp, T, q_liq, q_ice, Φ,
                    )

                    @test isfinite(energy)
                    I_liq = TD.internal_energy_liquid(thp, T)
                    @test energy ≈ I_liq + Φ rtol = FT(1e-5)
                end

                @testset "Cold conditions (all ice)" begin
                    T = FT(240.0)
                    q_liq = FT(0)
                    q_ice = FT(0.001)
                    Φ = FT(5000.0)

                    energy = ClimaAtmos.e_tot_0M_precipitation_sources_helper(
                        thp, T, q_liq, q_ice, Φ,
                    )

                    @test isfinite(energy)
                    I_ice = TD.internal_energy_ice(thp, T)
                    @test energy ≈ I_ice + Φ rtol = FT(1e-5)
                end

                @testset "Type stability" begin
                    energy = ClimaAtmos.e_tot_0M_precipitation_sources_helper(
                        thp, FT(280.0), FT(0.001), FT(0.0005), FT(1000.0),
                    )
                    @test typeof(energy) == FT
                end
            end
        end
    end

    @testset "Microphysics1MEvaluator Lagrange-Multiplier Logic" begin
        import CloudMicrophysics.Parameters as CMP
        import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
        import Thermodynamics as TD
        import ClimaParams as CP
        using ClimaAtmos: Microphysics1MEvaluator

        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                ρ = FT(1.0)
                T_mean = FT(280.0)
                q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
                q_tot_mean = q_sat_mean + FT(2e-3)
                # mu_S centres S′: at the grid-mean point S′_hat = 0 by construction.
                mu_S = q_tot_mean - q_sat_mean
                dt = FT(60)
                nsubs = 1

                @testset "Large-negative λ_lagrange → no condensate" begin
                    # λ_lagrange << 0 means even the most saturated quadrature
                    # point has q_c_hat = max(0, λ_lagrange + α·S′_hat) = 0, so
                    # BMT receives zero cloud condensate (only rain/snow
                    # evaporation).
                    eval_clear = Microphysics1MEvaluator(
                        BMT.Microphysics1Moment(), mp, thp, ρ,
                        FT(0), FT(0),           # q_rai, q_sno
                        FT(1), FT(-1), mu_S, FT(1),  # λ, λ_lagrange, mu_S, α
                        FT(1), FT(1), FT(1),  # w_cloudy, w_clear, ε_w
                        dt, nsubs, (),
                    )
                    # At the grid-mean point S′_hat = 0, so shifted_excess = -1
                    # and q_c_hat = max(0, -1) = 0.
                    result = eval_clear(T_mean, q_tot_mean)
                    ref = BMT.bulk_microphysics_tendencies(
                        BMT.LinearizedAverage(),
                        BMT.Microphysics1Moment(), mp, thp, ρ, T_mean,
                        q_tot_mean, FT(0), FT(0), FT(0), FT(0), dt, nsubs,
                    )
                    @test result.dq_lcl_dt ≈ ref.dq_lcl_dt rtol = FT(1e-4)
                    @test result.dq_icl_dt ≈ ref.dq_icl_dt rtol = FT(1e-4)
                end

                @testset "Positive λ_lagrange → condensate at grid-mean point" begin
                    # At the grid-mean quadrature point S′_hat = 0, so
                    # shifted_excess = λ_lagrange.  With λ=1 (all liquid) and
                    # q_rai=0 we get q_lcl_hat = λ_lagrange exactly.
                    q_c = FT(1e-3)
                    eval_cloud = Microphysics1MEvaluator(
                        BMT.Microphysics1Moment(), mp, thp, ρ,
                        FT(0), FT(0),            # q_rai, q_sno
                        FT(1), q_c, mu_S, FT(1), # λ, λ_lagrange, mu_S, α
                        FT(1), FT(1), FT(1),  # w_cloudy, w_clear, ε_w
                        dt, nsubs, (),
                    )
                    result = eval_cloud(T_mean, q_tot_mean)
                    ref = BMT.bulk_microphysics_tendencies(
                        BMT.LinearizedAverage(),
                        BMT.Microphysics1Moment(), mp, thp, ρ, T_mean,
                        q_tot_mean, q_c, FT(0), FT(0), FT(0), dt, nsubs,
                    )
                    @test result.dq_lcl_dt ≈ ref.dq_lcl_dt rtol = FT(1e-4)
                    @test result.dq_icl_dt ≈ ref.dq_icl_dt rtol = FT(1e-4)
                end

                @testset "Precipitation does not reduce reconstructed condensate" begin
                    # λ_lagrange enforces E[q_c_hat] = q_c on *cloud*
                    # condensate only, so at the mean point (S′_hat = 0), the
                    # reconstruction must recover q_lcl_hat = λ·q_c and
                    # q_icl_hat = (1−λ)·q_c regardless of q_rai / q_sno.
                    #
                    # Mixed-phase temperature so 0 < λ < 1 and both partitions
                    # (liquid vs q_rai, ice vs q_sno) are exercised.
                    T_mix = FT(263.15)
                    q_sat_mix = TD.q_vap_saturation(thp, T_mix, ρ)
                    q_tot_mix = q_sat_mix + FT(2e-3)
                    mu_S_mix = q_tot_mix - q_sat_mix
                    q_c = FT(1.5e-4)
                    # Temperature-ramp liquid fraction (≈0.75 at 263.15 K). Used
                    # identically below in the evaluator and the reference, so
                    # the exact value only needs to lie strictly in (0, 1).
                    λ_mix = TD.liquid_fraction_ramp(thp, T_mix)
                    q_rai = FT(1e-3)
                    q_sno = FT(5e-4)

                    eval_precip = Microphysics1MEvaluator(
                        BMT.Microphysics1Moment(), mp, thp, ρ,
                        q_rai, q_sno,               # q_rai, q_sno
                        λ_mix, q_c, mu_S_mix, FT(1),  # λ, λ_lagrange, mu_S, α
                        FT(1), FT(1), FT(1),  # w_cloudy, w_clear, ε_w
                        dt, nsubs, (),
                    )
                    result = eval_precip(T_mix, q_tot_mix)
                    # Reference: the condensate the closure must reconstruct at
                    # the mean point, plus the true precipitation.
                    ref = BMT.bulk_microphysics_tendencies(
                        BMT.LinearizedAverage(),
                        BMT.Microphysics1Moment(), mp, thp, ρ, T_mix,
                        q_tot_mix, λ_mix * q_c, (1 - λ_mix) * q_c,
                        q_rai, q_sno, dt, nsubs,
                    )
                    @test result.dq_lcl_dt ≈ ref.dq_lcl_dt rtol = FT(1e-4)
                    @test result.dq_icl_dt ≈ ref.dq_icl_dt rtol = FT(1e-4)
                    @test result.dq_rai_dt ≈ ref.dq_rai_dt rtol = FT(1e-4)
                    @test result.dq_sno_dt ≈ ref.dq_sno_dt rtol = FT(1e-4)
                end

                @testset "Output is finite NamedTuple" begin
                    eval = Microphysics1MEvaluator(
                        BMT.Microphysics1Moment(), mp, thp, ρ,
                        FT(0), FT(0),
                        FT(1), FT(5e-4), mu_S, FT(1),
                        FT(1), FT(1), FT(1),  # w_cloudy, w_clear, ε_w
                        dt, nsubs, (),
                    )
                    result = eval(T_mean, q_tot_mean)
                    @test result isa NamedTuple
                    @test isfinite(result.dq_lcl_dt)
                    @test isfinite(result.dq_icl_dt)
                    @test isfinite(result.dq_rai_dt)
                    @test isfinite(result.dq_sno_dt)
                end
            end
        end
    end


    @testset "Precipitation conditioning" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                α = FT(1)
                corr = FT(0.6)
                ρ = FT(1.1)
                dt = FT(60)
                nsubs = 1
                sigma_S = FT(3e-4)
                ε_w = CA.discrete_cloudy_weight_width(α, sigma_S)

                @testset "weights: bounds and degenerate limits" begin
                    for a_p in FT[0.05, 0.2, 0.5, 1], CF_d in FT[0.01, 0.3, 0.9]
                        w_cloudy, w_clear =
                            CA.precip_conditioning_weights(a_p, CF_d)
                        # A ∈ [1, 1/a_p] and B ∈ [0, 1].
                        @test one(FT) <= w_cloudy <= one(FT) / a_p
                        @test zero(FT) <= w_clear <= one(FT)
                        # Exact conservation under the discrete measure:
                        # A·CF_d + B·(1 − CF_d) = 1.
                        @test w_cloudy * CF_d + w_clear * (1 - CF_d) ≈
                              one(FT) rtol = sqrt(eps(FT))
                    end

                    # a_p = 1 (shaft fills the cell) → uniform, any CF_d.
                    for CF_d in FT[0, 0.4, 1]
                        @test CA.precip_conditioning_weights(FT(1), CF_d) ==
                              (one(FT), one(FT))
                    end

                    # In-cloud (a_p = CF_d): rain concentrated on the cloudy
                    # points at 1/a_p, nothing in the clear ones.
                    for a_p in FT[0.05, 0.31, 0.8]
                        w_cloudy, w_clear =
                            CA.precip_conditioning_weights(a_p, a_p)
                        @test w_cloudy ≈ one(FT) / a_p rtol = sqrt(eps(FT))
                        @test w_clear == zero(FT)
                    end

                    # No shaft, condensate-free, and overcast cells all fall
                    # back to uniform rather than dividing by zero. Below cloud
                    # base (CF_d = 0) that is the known limitation: this
                    # closure does not touch below-cloud evaporation.
                    for (a_p, CF_d) in
                        ((FT(0), FT(0.3)), (FT(0.4), FT(0)), (FT(0.4), FT(1)))
                        @test CA.precip_conditioning_weights(a_p, CF_d) ==
                              (one(FT), one(FT))
                    end
                end

                @testset "⟨q_rai_hat⟩ = q_rai exactly under the quadrature" begin
                    # Mass conservation of the assignment, over the same
                    # regimes and orders as the discrete-λ fit, with `CF_d`
                    # taken from `_compute_sgs_moments` so that the evaluator
                    # and the normalizer share one measure.
                    q_rai = FT(3e-4)
                    q_sno = FT(1e-4)
                    # (T, saturation offset, q_c, T′T′, q′q′)
                    regimes = (
                        (FT(288), FT(2e-3), FT(3e-4), FT(1), FT(1e-6)),
                        (FT(288), FT(-2e-3), FT(5e-5), FT(1), FT(1e-6)),
                        (FT(233), FT(1e-4), FT(2e-5), FT(1), FT(1e-7)),
                        (FT(288), FT(0), FT(2e-4), FT(9), FT(4e-6)),
                    )
                    for order in (1, 3, 5),
                        (T, dq, q_c, T′T′, q′q′) in regimes

                        quad = CA.SGSQuadrature(FT; quadrature_order = order)
                        q_tot = TD.q_vap_saturation(thp, T, ρ) + dq
                        m = CA._compute_sgs_moments(
                            thp, ρ, T, q_tot, q_c, quad, T′T′, q′q′, corr, α,
                        )
                        mu_S = q_tot - TD.q_vap_saturation(thp, T, ρ)
                        λ = TD.liquid_fraction(thp, T, q_c, FT(0))
                        # Shaft fractions from "exactly the cloudy points" to
                        # "the whole cell", plus one inherited from above.
                        for a_p in
                            (m.CF_d, min(one(FT), m.CF_d + FT(0.3)), one(FT))
                            a_p <= 0 && continue
                            evaluator = CA.Microphysics1MEvaluator(
                                BMT.Microphysics1Moment(), mp, thp, ρ,
                                q_rai, q_sno, λ, m.λ_lagrange, mu_S, α,
                                CA.precip_conditioning_weights(a_p, m.CF_d)...,
                                CA.discrete_cloudy_weight_width(α, m.sigma_S),
                                dt, nsubs, (),
                            )
                            rai_mean = CA.integrate_over_sgs(
                                ConditionedRainProbe(evaluator),
                                quad, q_tot, T, q′q′, T′T′, corr,
                            )
                            sno_mean = CA.integrate_over_sgs(
                                ConditionedSnowProbe(evaluator),
                                quad, q_tot, T, q′q′, T′T′, corr,
                            )
                            @test rai_mean ≈ q_rai rtol = sqrt(eps(FT))
                            @test sno_mean ≈ q_sno rtol = sqrt(eps(FT))
                        end
                    end
                end

                @testset "per-point q_v is invariant to the conditioning" begin
                    # The overlay exists so that moving rain between points
                    # cannot move vapor. Stronger than the mean test: the
                    # diagnosed vapor must be the same at every point, for any
                    # (a_p, CF_d). Analytically `q_v = q_np − q_c` drops the
                    # conditioning entirely; CloudMicrophysics reaches it by
                    # subtracting the conditioned values from the conditioned
                    # total, so the realized invariance is a few ulps of
                    # `q_tot` rather than bitwise. A missing overlay would
                    # shift `q_v` by `q_rai·(1/a_p − 1)`, five orders of
                    # magnitude above this tolerance.
                    T = FT(285)
                    q_sat = TD.q_vap_saturation(thp, T, ρ)
                    q_tot = q_sat + FT(1e-3)
                    q_c = FT(4e-4)
                    q_rai = FT(3e-4)
                    q_sno = FT(1e-4)
                    mu_S = q_tot - q_sat
                    λ = TD.liquid_fraction(thp, T, q_c, FT(0))
                    make_eval =
                        w -> CA.Microphysics1MEvaluator(
                            BMT.Microphysics1Moment(), mp, thp, ρ,
                            q_rai, q_sno, λ, q_c, mu_S, α,
                            w[1], w[2], ε_w, dt, nsubs, (),
                        )
                    eval_uniform = make_eval((FT(1), FT(1)))
                    for (a_p, CF_d) in (
                        (FT(0.31), FT(0.31)),
                        (FT(0.6), FT(0.2)),
                        (FT(0.05), FT(0.05)),
                        (FT(1), FT(0.5)),
                    )
                        ev = make_eval(
                            CA.precip_conditioning_weights(a_p, CF_d),
                        )
                        for (T_hat, q_hat) in (
                            (T, q_tot),
                            (T - FT(2), q_tot + FT(1e-3)),
                            (T + FT(2), q_tot - FT(2e-3)),
                            (T, FT(0)),
                        )
                            s_cond =
                                CA._conditioned_point_state(ev, T_hat, q_hat)
                            s_unif = CA._conditioned_point_state(
                                eval_uniform, T_hat, q_hat,
                            )
                            ulps = 32 * eps(FT) * q_tot
                            @test diagnosed_q_vap(s_cond) ≈
                                  diagnosed_q_vap(s_unif) atol = ulps rtol = 0
                            # Cloud condensate is untouched entirely: it never
                            # sees the conditioning, so this one is bitwise.
                            @test s_cond.q_lcl === s_unif.q_lcl
                            @test s_cond.q_icl === s_unif.q_icl
                            # Total water still carries the conditioned precip.
                            @test s_cond.q_tot ≈
                                  s_unif.q_tot - q_rai - q_sno +
                                  s_cond.q_rai +
                                  s_cond.q_sno atol = ulps rtol = 0
                        end
                    end
                end

                @testset "uniform conditioning reproduces constant precip" begin
                    T = FT(285)
                    q_tot = TD.q_vap_saturation(thp, T, ρ) + FT(1e-3)
                    q_c = FT(4e-4)
                    q_rai = FT(3e-4)
                    q_sno = FT(1e-4)
                    mu_S = q_tot - TD.q_vap_saturation(thp, T, ρ)
                    λ = TD.liquid_fraction(thp, T, q_c, FT(0))
                    # `a_p = 1` and the explicit uniform limit agree, and both
                    # hand every point the cell-mean precipitation.
                    for (w_cloudy, w_clear) in (
                        (FT(1), FT(1)),
                        CA.precip_conditioning_weights(FT(1), FT(0.4)),
                    )
                        ev = CA.Microphysics1MEvaluator(
                            BMT.Microphysics1Moment(), mp, thp, ρ,
                            q_rai, q_sno, λ, q_c, mu_S, α,
                            w_cloudy, w_clear, ε_w, dt, nsubs, (),
                        )
                        for q_hat in (q_tot, q_tot + FT(2e-3), q_tot - FT(2e-3))
                            st = CA._conditioned_point_state(ev, T, q_hat)
                            @test st.q_rai ≈ q_rai rtol = sqrt(eps(FT))
                            @test st.q_sno ≈ q_sno rtol = sqrt(eps(FT))
                        end
                    end
                end

                @testset "in-cloud conditioning concentrates rain" begin
                    # `a_p = CF_d`: cloudy points get q_rai/a_p, clear points
                    # get nothing. This is the in-cloud overlap fix.
                    T = FT(288)
                    q_sat = TD.q_vap_saturation(thp, T, ρ)
                    q_tot = q_sat + FT(2e-4)
                    q_c = FT(2e-4)
                    q_rai = FT(3e-4)
                    quad = CA.SGSQuadrature(FT; quadrature_order = 3)
                    T′T′, q′q′ = FT(1), FT(1e-6)
                    m = CA._compute_sgs_moments(
                        thp, ρ, T, q_tot, q_c, quad, T′T′, q′q′, corr, α,
                    )
                    @test 0 < m.CF_d < 1  # partially cloudy, or the test is vacuous
                    ev = CA.Microphysics1MEvaluator(
                        BMT.Microphysics1Moment(), mp, thp, ρ,
                        q_rai, FT(0), FT(1), m.λ_lagrange,
                        q_tot - q_sat, α,
                        CA.precip_conditioning_weights(m.CF_d, m.CF_d)...,
                        CA.discrete_cloudy_weight_width(α, m.sigma_S),
                        dt, nsubs, (),
                    )
                    # Deep inside the cloudy branch: q_rai/CF_d.
                    st_wet = CA._conditioned_point_state(
                        ev, T, q_tot + 10 * m.sigma_S,
                    )
                    @test st_wet.q_rai ≈ q_rai / m.CF_d rtol = FT(1e-3)
                    # Deep in the clear branch: no rain at all.
                    st_dry = CA._conditioned_point_state(
                        ev, T, q_tot - 10 * m.sigma_S,
                    )
                    @test st_dry.q_rai < FT(1e-3) * q_rai
                end
            end
        end
    end

end
