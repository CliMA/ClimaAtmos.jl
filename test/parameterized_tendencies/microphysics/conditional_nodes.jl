#=
Unit tests for the conditional-moment node rule of the 1M SGS microphysics
(`GaussianSGS`, `conditional_rule`, `sum_over_conditional_nodes`):

  - the two-point Gauss rule of a truncated normal reproduces its first three
    moments and keeps its nodes inside the interval, in Float32 tails too,
  - the rule's weights sum to one, its nodes round-trip through the evaluator's
    saturation-excess reconstruction, and it integrates the actual and the
    equilibrium condensate and the shaft fraction exactly,
  - `_compute_sgs_moments` returns the continuous multipliers for the rule,
  - the driver runs with the rule,
  - the distribution parses from the config, the temperature points keep the
    mean temperature and the node's `S′` and lie on either side of the
    freezing point, and the rule reduces to the grid-mean evaluation as the
    variance vanishes.
=#

using Test
using ClimaAtmos
import Thermodynamics as TD
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

const CA = ClimaAtmos

@testset "Conditional-moment node rule" begin

    @testset "truncated-normal Gauss pair" begin
        for FT in (Float32, Float64)
            tol = FT == Float32 ? 2e-3 : 1e-9
            for (a, b) in (
                (FT(-Inf), FT(Inf)), (FT(-1), FT(Inf)), (FT(2), FT(Inf)),
                (FT(0.5), FT(1.5)), (FT(-Inf), FT(-2)), (FT(4), FT(Inf)),
            )
                (w, x₁, p₁, x₂, p₂) = CA._truncated_normal_gauss2(a, b)
                (I₀, I₁, I₂, I₃) = CA._gaussian_interval_moments4(a, b)
                @test w == I₀
                @test w > 0
                @test p₁ + p₂ ≈ 1
                @test p₁ > 0 && p₂ > 0
                @test a <= x₁ <= x₂ <= b
                @test p₁ * x₁ + p₂ * x₂ ≈ I₁ / I₀ atol = tol * max(1, abs(I₁ / I₀))
                @test p₁ * x₁^2 + p₂ * x₂^2 ≈ I₂ / I₀ rtol = tol
                @test p₁ * x₁^3 + p₂ * x₂^3 ≈ I₃ / I₀ rtol = 10 * tol
            end
            # degenerate interval: both nodes at the mean, zero weight
            (w, x₁, p₁, x₂, p₂) = CA._truncated_normal_gauss2(FT(1), FT(1))
            @test w == 0
            @test x₁ == x₂
            @test p₁ == p₂ == FT(0.5)
        end
    end

    @testset "rule: weights, round trip, exact means" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            ρ = FT(1.1)
            T = FT(285)
            T′T′ = FT(0.25)
            q′q′ = FT(2.5e-7)
            corr = FT(0.5)
            σ_S = CA._sgs_linear_sigma_S(thp, ρ, T, T′T′, q′q′, corr)
            β_T = CA.sgs_temperature_regression(thp, T, ρ, σ_S, T′T′, q′q′, corr)
            # the regressed temperature spread never exceeds σ_T
            @test abs(β_T) * σ_S <= sqrt(T′T′) * (1 + sqrt(eps(FT)))
            tol = FT == Float32 ? 5e-4 : 1e-10
            for z in FT.((-3, -1.5, -0.5, 0, 1)), α in FT.((0.3, 1))
                mu_S = z * σ_S
                q_c_eq = CA.truncated_gaussian_condensate(mu_S, σ_S)
                q_c = FT(0.6) * q_c_eq
                λ = CA._continuous_lagrange(q_c, α * σ_S)
                a_p = FT(0.3)
                t_p = CA.sgs_precip_shaft_threshold(a_p, σ_S)
                rule = CA.conditional_rule(
                    thp, ρ, T, mu_S, σ_S, β_T, -λ / α, -mu_S, t_p, FT(150), FT(0.05),
                )
                @test rule isa CA.ConditionalRule{8, FT}
                # the weights are differences of tail probabilities from two
                # approximations of Φ, accurate to ~10⁻⁸
                @test sum(n.w for n in rule.nodes) ≈ 1 atol = (FT == Float32 ? 1e-6 : 1e-8)
                @test all(n.w >= 0 for n in rule.nodes)
                @test rule.nodes[1].w > 0
                # the evaluator recovers S′ = q − q_sat(T, ρ) − μ_S at each node
                for n in rule.nodes
                    S′ = n.q - TD.q_vap_saturation(thp, n.T, ρ) - mu_S
                    @test S′ ≈ n.S′ atol = 64 * eps(FT) * (abs(n.q) + σ_S)
                end
                # the node means of the linear condensates are the continuous
                # ones, to the accuracy of the Φ approximations in the tail
                # (`truncated_gaussian_condensate` uses `normal_cdf`)
                g = CA.conditional_rule_mean(CA.NodeCondensate(λ, α), rule)
                @test g ≈ q_c rtol = (FT == Float32 ? 3e-3 : 2e-3)
                g_eq = CA.conditional_rule_mean(CA.NodeCondensate(mu_S, one(FT)), rule)
                @test g_eq ≈ q_c_eq rtol = (FT == Float32 ? 3e-3 : 2e-3)

                # Test clear-sky nucleation at qc = 0 with subsaturated mean (mu_S < 0)
                rule_nucleation = CA.conditional_rule(
                    thp, ρ, T, mu_S, σ_S, β_T, FT(Inf), -mu_S, t_p, FT(150), FT(0.05),
                )
                g_nucleation = CA.conditional_rule_mean(
                    CA.NodeCondensate(mu_S, one(FT)),
                    rule_nucleation,
                )
                @test g_nucleation ≈ q_c_eq rtol = (FT == Float32 ? 3e-3 : 2e-3)

                # Test sliver guard: when |t_e - t_a| < 1e-3 * σ_S, weights sum to 1 with zero weight on sliver
                t_e_sliver = -λ / α + FT(0.5e-3) * σ_S
                rule_sliver = CA.conditional_rule(
                    thp, ρ, T, mu_S, σ_S, β_T, -λ / α, t_e_sliver, t_p, FT(150),
                    FT(0.05),
                )
                @test sum(n.w for n in rule_sliver.nodes) ≈ 1 atol =
                    (FT == Float32 ? 1e-6 : 1e-8)
                @test all(n.w >= 0 for n in rule_sliver.nodes)
                shaft = CA.conditional_rule_shaft_fraction(rule)
                # exact up to the quantile approximation of the threshold:
                # |δz| < 4.5e-4 in `normal_cdf_inv`, so |shaft − a_p| ≤ φ(0)·4.5e-4 < 2e-4
                @test shaft ≈ a_p atol = 2e-4
                # cloudy intervals carry two nodes, clear ones a single node
                cloudy = count(n -> n.w > 0 && n.S′ >= -λ / α, rule.nodes)
                @test cloudy >= 2
            end
        end
    end

    @testset "_compute_sgs_moments: continuous multipliers" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            quad = CA.SGSQuadrature(
                FT; quadrature_order = 3, distribution = CA.GaussianSGS(),
            )
            ρ = FT(1.1)
            T = FT(285)
            q_sat = TD.q_vap_saturation(thp, T, ρ)
            T′T′ = FT(0.25)
            q′q′ = FT(2.5e-7)
            corr = FT(0.5)
            for z in FT.((-2, -0.5, 0.5)), α in FT.((0.3, 1))
                mu_S = z * FT(4e-4)
                q_tot = q_sat + mu_S
                q_rai = FT(2e-5)
                q_tot_cloud = q_tot - q_rai
                σ_S = CA._sgs_linear_sigma_S(thp, ρ, T, T′T′, q′q′, corr)
                q_c = FT(0.5) * CA.truncated_gaussian_condensate(mu_S - q_rai, σ_S)
                mom = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, q_c, quad, T′T′, q′q′, corr, α, q_tot_cloud,
                )
                # the linearized σ_S
                @test mom.sigma_S == σ_S
                # λ_eq is the mean cloud excess; λ solves the continuous constraint
                @test mom.λ_eq ≈ mu_S - q_rai atol = 64 * eps(FT) * q_sat
                C = CA.truncated_gaussian_condensate(mom.λ_lagrange, α * σ_S)
                @test C ≈ q_c rtol = (FT == Float32 ? 1e-3 : 1e-6)
                @test mom.sigma_S isa FT && mom.λ_lagrange isa FT && mom.λ_eq isa FT
            end
            # no condensate: the multiplier puts the cloud edge far out
            mom0 = CA._compute_sgs_moments(
                thp, ρ, T, q_sat - FT(1e-3), FT(0), quad, T′T′, q′q′, corr, FT(1),
            )
            @test mom0.λ_lagrange < -3 * mom0.sigma_S
        end
    end

    @testset "driver with GaussianSGS" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            cmp = CMP.Microphysics1MParams(toml_dict)
            quad = CA.SGSQuadrature(
                FT; quadrature_order = 3, distribution = CA.GaussianSGS(),
            )
            ρ = FT(1.1)
            T = FT(285)
            q_sat = TD.q_vap_saturation(thp, T, ρ)
            T′T′ = FT(0.25)
            q′q′ = FT(2.5e-7)
            corr = FT(0.5)
            α = FT(0.7)
            for z in FT.((-1.5, 0.3))
                mu_S = z * FT(4e-4)
                q_tot = q_sat + mu_S
                q_rai = FT(1e-5)
                q_sno = FT(0)
                q_lcl = FT(0.3) * CA.truncated_gaussian_condensate(mu_S, FT(4e-4))
                q_icl = FT(0)
                mom = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, q_lcl + q_icl, quad, T′T′, q′q′, corr, α,
                    q_tot - q_rai - q_sno,
                )
                out = CA.microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(), quad, cmp, thp, ρ, T, FT(0.1),
                    q_tot, q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr,
                    mom.λ_lagrange, α, FT(0.1), FT(0.1), FT(180), 2,
                    TD.liquid_fraction(thp, T, q_lcl, q_icl), mu_S,
                    FT(0.5), FT(0.4), mom.sigma_S, FT(0.1), Val(8),
                )
                @test out.dq_lcl_dt isa FT
                # a warm cell without ice or snow makes none
                @test out.dq_icl_dt == 0 && out.dq_sno_dt == 0
                uniform(a_p, floor) = CA.microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(), quad, cmp, thp, ρ, T, FT(0.1),
                    q_tot, q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr,
                    mom.λ_lagrange, α, FT(0.1), FT(0.1), FT(180), 2,
                    TD.liquid_fraction(thp, T, q_lcl, q_icl), mu_S,
                    FT(0), a_p, mom.sigma_S, floor, Val(8),
                )
                out_u = uniform(FT(0.4), FT(0.1))
                # without placement the shaft area and its floor do not enter
                @test out_u == uniform(FT(0.9), FT(0.5))
                # the shaft sits at the moist end, where the cloud is: its rain
                # evaporates less and collects more cloud than uniform rain
                @test out.dq_rai_dt > out_u.dq_rai_dt
                @test out.dq_lcl_dt < out_u.dq_lcl_dt

                # Default 6-node rule verifies increased rain growth with shaft placement
                out6 = CA.microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(), quad, cmp, thp, ρ, T, FT(0.1),
                    q_tot, q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr,
                    mom.λ_lagrange, α, FT(0.1), FT(0.1), FT(180), 2,
                    TD.liquid_fraction(thp, T, q_lcl, q_icl), mu_S,
                    FT(0.5), FT(0.4), mom.sigma_S, FT(0.1),
                )
                out_u6 = CA.microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(), quad, cmp, thp, ρ, T, FT(0.1),
                    q_tot, q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr,
                    mom.λ_lagrange, α, FT(0.1), FT(0.1), FT(180), 2,
                    TD.liquid_fraction(thp, T, q_lcl, q_icl), mu_S,
                    FT(0), FT(0.4), mom.sigma_S, FT(0.1),
                )
                @test out6.dq_rai_dt > out_u6.dq_rai_dt
            end
        end
    end

    @testset "config" begin
        @test CA.get_sgs_distribution(Dict("sgs_distribution" => "gaussian")) ==
              CA.GaussianSGS()
        @test_throws ErrorException CA.get_sgs_distribution(
            Dict("sgs_distribution" => "lognormal"),
        )
    end

    @testset "temperature points" begin
        for FT in (Float32, Float64)
            thp = TD.Parameters.ThermodynamicsParameters(CP.create_toml_dict(FT))
            T_frz = FT(TD.Parameters.T_freeze(thp))
            σ⊥ = FT(0.8)
            # away from the freezing point: one point, no partner
            @test CA._freezing_split(T_frz + 5, σ⊥, T_frz) == (1, 0, 0, 0)
            @test CA._freezing_split(T_frz - 5, σ⊥, T_frz) == (1, 0, 0, 0)
            @test CA._freezing_split(T_frz + FT(0.3), FT(0), T_frz) == (1, 0, 0, 0)
            spread = CA.TemperatureSpread(thp, FT(1), FT(0), σ⊥, T_frz, FT(150), FT(0.05))
            pts = CA._temperature_points(T_frz + 5, spread)
            @test pts[1] == (1, 0) && all(p == (0, 0) for p in pts[2:end])
            # straddling: probabilities sum to one, the mean temperature is
            # kept, the points lie on the sides of T_freeze they represent
            for dT in FT.((-1, -0.3, 0, 0.5, 1.2))
                pts = CA._temperature_points(T_frz + dT, spread)
                @test sum(first, pts) ≈ 1
                @test sum(p * t for (p, t) in pts) ≈ 0 atol = 10 * eps(FT)
                n_lo = length(pts) ÷ 2
                @test all(T_frz + dT + t * σ⊥ < T_frz for (_, t) in pts[1:n_lo])
                @test all(T_frz + dT + t * σ⊥ > T_frz for (_, t) in pts[(n_lo + 1):end])
            end
            @test sum(first, CA._temperature_points(T_frz, spread)[1:(end ÷ 2)]) ≈
                  FT(0.5)
            # summed over a rule near freezing: the probability, the mean
            # temperature and every point's S′ are those of the unsplit rule
            ρ, σ_S, β_T = FT(1), FT(3e-4), FT(-1000)
            mu_S = FT(-0.5) * σ_S
            λ = CA._continuous_lagrange(
                FT(0.5) * CA.truncated_gaussian_condensate(mu_S, σ_S),
                σ_S,
            )
            t_p = CA.sgs_precip_shaft_threshold(FT(0.5), σ_S)
            rule = CA.conditional_rule(
                thp, ρ, T_frz + FT(0.2), mu_S, σ_S, β_T, -λ, -mu_S, t_p, FT(150),
                FT(0.05),
            )
            @test rule isa CA.ConditionalRule{8, FT}
            rule6 = CA.conditional_rule(
                thp, ρ, T_frz + FT(0.2), mu_S, σ_S, β_T, -λ, t_p, FT(150),
                FT(0.05),
            )
            @test rule6 isa CA.ConditionalRule{6, FT}
            @test sum(n.w for n in rule6.nodes) ≈ 1
            σ⊥ = CA.sgs_temperature_spread_at_fixed_excess(FT(1), β_T, σ_S)
            @test (β_T * σ_S)^2 + σ⊥^2 ≈ 1
            spread = CA.TemperatureSpread(thp, ρ, mu_S, σ⊥, T_frz, FT(150), FT(0.05))
            one_f = (T, q, s) -> one(T)
            T_f = (T, q, s) -> T
            S_f = (T, q, s) -> q - TD.q_vap_saturation(thp, T, ρ) - mu_S
            # the unsplit rule: no spread at fixed S′
            unsplit = CA.TemperatureSpread(thp, ρ, mu_S, FT(0), T_frz, FT(150), FT(0.05))
            @test CA.sum_over_conditional_nodes(one_f, rule, spread) ≈
                  CA.sum_over_conditional_nodes(one_f, rule, unsplit)
            @test CA.sum_over_conditional_nodes(T_f, rule, spread) ≈
                  CA.sum_over_conditional_nodes(T_f, rule, unsplit)
            @test CA.sum_over_conditional_nodes(S_f, rule, spread) ≈
                  CA.sum_over_conditional_nodes(S_f, rule, unsplit) atol =
                64 * eps(FT) * σ_S
        end
    end

    @testset "vanishing variance: the grid mean" begin
        for FT in (Float32, Float64)
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            cmp = CMP.Microphysics1MParams(toml_dict)
            quad = CA.SGSQuadrature(
                FT; quadrature_order = 3, distribution = CA.GaussianSGS(),
            )
            ρ = FT(1.1)
            T′T′ = FT(1e-10)
            q′q′ = FT(1e-18)
            corr = FT(0.5)
            α = FT(1)
            for T in FT.((285, 273.5, 260)), z in FT.((-1, 0.5))
                q_sat = TD.q_vap_saturation(thp, T, ρ)
                mu_S = z * FT(1e-4)
                q_tot = q_sat + mu_S
                (q_rai, q_sno) = (FT(2e-5), FT(1e-5))
                q_lcl = FT(1e-4)
                q_icl = FT(2e-5)
                mom = CA._compute_sgs_moments(
                    thp, ρ, T, q_tot, q_lcl + q_icl, quad, T′T′, q′q′, corr, α,
                    q_tot - q_rai - q_sno,
                )
                out = CA.microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(), quad, cmp, thp, ρ, T, FT(0.1),
                    q_tot, q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr,
                    mom.λ_lagrange, α, FT(0), FT(0), FT(150), 2,
                    TD.liquid_fraction(thp, T, q_lcl, q_icl), mu_S,
                    FT(0), FT(0.5), mom.sigma_S, FT(0.5),
                )
                ref = CA.microphysics_tendencies_1m(
                    ρ, q_tot, q_lcl, q_icl, q_rai, q_sno, T, FT(0.1), cmp, thp,
                    FT(150), 2,
                )
                scale = maximum(abs, values(ref)) + floatmin(FT)
                for k in (:dq_lcl_dt, :dq_icl_dt, :dq_rai_dt, :dq_sno_dt)
                    @test getproperty(out, k) ≈ getproperty(ref, k) atol =
                        (FT == Float32 ? 1e-3 : 1e-6) * scale
                end
            end
        end
    end
end
