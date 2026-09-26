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
import ClimaAtmos:
    limit_sink, microphysics_tendencies_1m, Microphysics1MEvaluator, sgs_local_condensate

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
                    mp, thp, ρ, T, FT(0),
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

    @testset "BMT 1M vertical-velocity plumbing" begin
        # CloudMicrophysics 0.41: the Kessler1M rain autoconversion blends its timescale
        # between a stratiform and a convective value with the subdomain vertical velocity
        # (f(w) = w⁴ / (w⁴ + w₀⁴), w₀ = 1.5 m/s by default). The ClimaParams defaults set the
        # two values equal (classic Kessler), so `w` must be made active here by giving the
        # stratiform regime a longer timescale. The tests then check that `w` reaches
        # CloudMicrophysics through every 1M wrapper: an updraft (w = 10 m/s) must convert
        # cloud liquid to rain faster than quiescent air (w = 0), and each wrapper must agree
        # with the direct BMT call made with the same `w`.
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT;
                    override_file = Dict(
                        "rain_autoconversion_timescale_stratiform" =>
                            Dict("value" => 10000.0, "type" => "float"),
                        "rain_autoconversion_timescale" =>
                            Dict("value" => 1000.0, "type" => "float"),
                    ),
                )
                mp = CMP.Microphysics1MParams(toml_dict)  # default Kessler1M
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                @test mp.processes.rain_autoconversion isa CMP.Kessler1M

                ρ = FT(1.0)
                T = FT(280.0)          # warm: cloud water is liquid
                q_lcl = FT(1.5e-3)     # above the autoconversion threshold
                q_icl = FT(0)
                q_rai = FT(1e-4)
                q_sno = FT(0)
                q_tot = TD.q_vap_saturation(thp, T, ρ) + q_lcl + q_rai  # saturated
                dt = FT(60.0)
                nsubs = 1
                w_rest = FT(0)
                w_up = FT(10)

                # Reference: direct BMT calls with the two velocities
                bmt(w) = BMT.bulk_microphysics_tendencies(
                    BMT.LinearizedAverage(), BMT.Microphysics1Moment(),
                    mp, thp, ρ, T, w, q_tot, q_lcl, q_icl, q_rai, q_sno, dt, nsubs,
                )
                ref_rest = bmt(w_rest)
                ref_up = bmt(w_up)
                @test ref_up.dq_rai_dt > ref_rest.dq_rai_dt > FT(0)

                @testset "non-quadrature wrapper forwards w" begin
                    r_rest = microphysics_tendencies_1m(
                        ρ, q_tot, q_lcl, q_icl, q_rai, q_sno, T, w_rest, mp, thp, dt,
                        nsubs,
                    )
                    r_up = microphysics_tendencies_1m(
                        ρ, q_tot, q_lcl, q_icl, q_rai, q_sno, T, w_up, mp, thp, dt,
                        nsubs,
                    )
                    @test r_up.dq_rai_dt > r_rest.dq_rai_dt
                    @test r_rest.dq_rai_dt == ref_rest.dq_rai_dt
                    @test r_up.dq_rai_dt == ref_up.dq_rai_dt
                end

                # Lagrange-multiplier moments at zero SGS variance: the single quadrature
                # point sits at the mean, so the quadrature wrapper must reproduce the
                # direct call (see sgs_quadrature.jl "Single Point = Grid Mean").
                λ = TD.liquid_fraction(thp, T, q_lcl, q_icl)
                mu_S = q_tot - TD.q_vap_saturation(thp, T, ρ)
                λ_lagrange = q_lcl + q_icl
                α = FT(1)

                @testset "quadrature wrapper forwards w" begin
                    quad = ClimaAtmos.SGSQuadrature(
                        FT;
                        quadrature_order = 1,
                        distribution = ClimaAtmos.GaussianSGS(),
                    )
                    quad_1m(w) = microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(), quad, mp, thp, ρ, T, w,
                        q_tot, q_lcl, q_icl, q_rai, q_sno,
                        FT(0), FT(0), FT(0), λ_lagrange, α, FT(0), FT(0), dt, nsubs,
                    )
                    q_rest = quad_1m(w_rest)
                    q_up = quad_1m(w_up)
                    @test q_up.dq_rai_dt > q_rest.dq_rai_dt
                    @test q_rest.dq_rai_dt ≈ ref_rest.dq_rai_dt rtol = FT(1e-5)
                    @test q_up.dq_rai_dt ≈ ref_up.dq_rai_dt rtol = FT(1e-5)
                end

                @testset "Microphysics1MEvaluator forwards w" begin
                    evaluator(w) = Microphysics1MEvaluator(
                        BMT.Microphysics1Moment(), mp, thp, ρ, w,
                        q_rai, q_sno, λ, FT(0), FT(0), FT(0), FT(0),
                        λ_lagrange, mu_S, α, dt, nsubs, (),
                    )
                    e_rest = evaluator(w_rest)(T, q_tot)
                    e_up = evaluator(w_up)(T, q_tot)
                    @test e_up.dq_rai_dt > e_rest.dq_rai_dt
                    @test e_rest.dq_rai_dt ≈ ref_rest.dq_rai_dt rtol = FT(1e-5)
                    @test e_up.dq_rai_dt ≈ ref_up.dq_rai_dt rtol = FT(1e-5)
                end

                @testset "even in w" begin
                    @test bmt(-w_up).dq_rai_dt == ref_up.dq_rai_dt
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

                @testset "Large-negative λ_lagrange → zero shifted excess → no condensate" begin
                    # λ_lagrange << 0 means even the most saturated quadrature
                    # point has max(0, λ_lagrange + α·S′_hat) = 0, so BMT
                    # receives zero cloud condensate (only rain/snow evaporation).
                    eval_clear = Microphysics1MEvaluator(
                        BMT.Microphysics1Moment(), mp, thp, ρ, FT(0),
                        FT(0), FT(0),           # q_rai, q_sno
                        FT(1), FT(0), FT(0), FT(0), FT(0),  # λ, ξ_liq, ξ_ice, q_lcl, q_icl
                        FT(-1), mu_S, FT(1),         # λ_lagrange, mu_S, α
                        dt, nsubs, (),
                    )
                    # At the grid-mean point S′_hat = 0, so shifted_excess = max(0,-1) = 0.
                    result = eval_clear(T_mean, q_tot_mean)
                    ref = BMT.bulk_microphysics_tendencies(
                        BMT.LinearizedAverage(),
                        BMT.Microphysics1Moment(), mp, thp, ρ, T_mean, FT(0),
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
                        BMT.Microphysics1Moment(), mp, thp, ρ, FT(0),
                        FT(0), FT(0),            # q_rai, q_sno
                        FT(1), FT(0), FT(0), FT(0), FT(0),  # λ, ξ_liq, ξ_ice, q_lcl, q_icl
                        q_c, mu_S, FT(1),            # λ_lagrange, mu_S, α
                        dt, nsubs, (),
                    )
                    result = eval_cloud(T_mean, q_tot_mean)
                    ref = BMT.bulk_microphysics_tendencies(
                        BMT.LinearizedAverage(),
                        BMT.Microphysics1Moment(), mp, thp, ρ, T_mean, FT(0),
                        q_tot_mean, q_c, FT(0), FT(0), FT(0), dt, nsubs,
                    )
                    @test result.dq_lcl_dt ≈ ref.dq_lcl_dt rtol = FT(1e-4)
                    @test result.dq_icl_dt ≈ ref.dq_icl_dt rtol = FT(1e-4)
                end

                @testset "Precipitation does not reduce reconstructed condensate" begin
                    # λ_lagrange enforces E[shifted_excess] = q_c on *cloud*
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
                        BMT.Microphysics1Moment(), mp, thp, ρ, FT(0),
                        q_rai, q_sno,               # q_rai, q_sno
                        λ_mix, FT(0), FT(0), FT(0), FT(0),  # λ, ξ_liq, ξ_ice, q_lcl, q_icl
                        q_c, mu_S_mix, FT(1),        # λ_lagrange, mu_S, α
                        dt, nsubs, (),
                    )
                    result = eval_precip(T_mix, q_tot_mix)
                    # Reference: the condensate the closure must reconstruct at
                    # the mean point, plus the true precipitation.
                    ref = BMT.bulk_microphysics_tendencies(
                        BMT.LinearizedAverage(),
                        BMT.Microphysics1Moment(), mp, thp, ρ, T_mix, FT(0),
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
                        BMT.Microphysics1Moment(), mp, thp, ρ, FT(0),
                        FT(0), FT(0),
                        FT(1), FT(0), FT(0), FT(0), FT(0),
                        FT(5e-4), mu_S, FT(1),
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

    @testset "uniform fractions of the condensate reconstruction" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
                mp = CMP.Microphysics1MParams(toml_dict)
                dt = FT(60)
                nsubs = 1
                ρ = FT(0.6)
                w = FT(0)

                @testset "sgs_local_condensate: limits and blend" begin
                    for λ in (FT(0), FT(0.3), FT(1)), se in (FT(0), FT(2e-4))
                        q_l, q_i = FT(4e-5), FT(5e-5)
                        # ξ = 0: the excess split, bitwise
                        ql, qi = sgs_local_condensate(λ, se, FT(0), FT(0), q_l, q_i)
                        @test ql === λ * se && qi === (FT(1) - λ) * se
                        # ξ = 1: the subdomain mean at the node
                        ql, qi = sgs_local_condensate(λ, se, FT(1), FT(1), q_l, q_i)
                        @test ql == q_l && qi == q_i
                        # mixed: liquid excess, ice uniform (the run-15 configuration)
                        ql, qi = sgs_local_condensate(λ, se, FT(0), FT(1), q_l, q_i)
                        @test ql === λ * se && qi == q_i
                        # interior blend and type stability
                        ql, qi = sgs_local_condensate(λ, se, FT(0.25), FT(0.5), q_l, q_i)
                        @test ql ≈ FT(0.75) * λ * se + FT(0.25) * q_l
                        @test qi ≈ FT(0.5) * (FT(1) - λ) * se + FT(0.5) * q_i
                        @test ql isa FT && qi isa FT
                    end
                end

                # ---- ice-only cell at −25 °C (λ = 0)
                T = FT(248)
                q_icl = FT(3e-5)
                q_c = q_icl
                λ_i = FT(TD.liquid_fraction(thp, T, FT(0), q_icl))
                @test λ_i == 0
                q_sat = TD.q_vap_saturation(thp, T, ρ)
                q_tot = q_sat + q_c
                mu_S = q_tot - q_sat
                make(ξ_l, ξ_i) = Microphysics1MEvaluator(
                    BMT.Microphysics1Moment(), mp, thp, ρ, w, FT(0), FT(0), λ_i,
                    ξ_l, ξ_i, FT(0), q_icl, q_c, mu_S, FT(1), dt, nsubs, (),
                )
                ev_e = make(FT(0), FT(0))
                ev_u = make(FT(0), FT(1))
                ev_h = make(FT(0), FT(0.5))

                @testset "dry node: ice sublimates when uniform, absent when excess" begin
                    q̂ = FT(0.9) * TD.q_vap_saturation(thp, T, ρ, TD.Ice())
                    @test q_c + q̂ - q_tot < 0   # shifted_excess = 0 at this node
                    out_e = ev_e(T, q̂)
                    out_u = ev_u(T, q̂)
                    out_h = ev_h(T, q̂)
                    @test out_e.dq_icl_dt == 0
                    @test out_u.dq_icl_dt < 0
                    @test out_h.dq_icl_dt < 0 && out_h.dq_icl_dt > out_u.dq_icl_dt
                end

                @testset "mean node: all fractions hold q_icl" begin
                    ref = BMT.bulk_microphysics_tendencies(
                        BMT.LinearizedAverage(),
                        BMT.Microphysics1Moment(), mp, thp, ρ, T, w,
                        q_tot, FT(0), q_icl, FT(0), FT(0), dt, nsubs,
                    )
                    for ev in (ev_e, ev_u, ev_h)
                        out = ev(T, q_tot)
                        @test out.dq_icl_dt ≈ ref.dq_icl_dt rtol = FT(1e-4)
                    end
                end

                @testset "quadrature wrapper: explicit λ, mu_S equal their defaults; ξ_ice = 1 differs" begin
                    quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3)
                    T′T′ = FT(1)
                    q′q′ = (FT(0.1) * q_tot)^2
                    base = microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(), quad, mp, thp, ρ, T, w, q_tot,
                        FT(0), q_icl, FT(0), FT(0), T′T′, q′q′, FT(0.6),
                        q_c, FT(1), FT(0), FT(0), dt, nsubs,
                    )
                    same = microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(), quad, mp, thp, ρ, T, w, q_tot,
                        FT(0), q_icl, FT(0), FT(0), T′T′, q′q′, FT(0.6),
                        q_c, FT(1), FT(0), FT(0), dt, nsubs, λ_i, mu_S,
                    )
                    @test same === base
                    uni = microphysics_tendencies_1m(
                        BMT.Microphysics1Moment(), quad, mp, thp, ρ, T, w, q_tot,
                        FT(0), q_icl, FT(0), FT(0), T′T′, q′q′, FT(0.6),
                        q_c, FT(1), FT(0), FT(1), dt, nsubs, λ_i, mu_S,
                    )
                    @test all(isfinite, values(uni))
                    @test uni.dq_icl_dt != base.dq_icl_dt
                end
            end
        end
    end
end
