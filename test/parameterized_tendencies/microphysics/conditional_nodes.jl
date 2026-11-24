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
    variance vanishes,
  - the linearized width `σ_S` and the temperature regression `β_T` keep their
    Float32 precision and the Cauchy–Schwarz bound where the fluctuations
    follow the saturation curve.
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
                    T, σ_S, β_T, -λ / α, -mu_S, t_p, FT(150),
                )
                @test rule isa CA.ConditionalRule{8, FT}
                # the weights are differences of tail probabilities from two
                # approximations of Φ, accurate to ~10⁻⁸
                @test sum(n.w for n in rule.nodes) ≈ 1 atol = (FT == Float32 ? 1e-6 : 1e-8)
                @test all(n.w >= 0 for n in rule.nodes)
                @test rule.nodes[1].w > 0
                # the evaluator recovers S′ = q − q_sat − μ_S at each node, from
                # the point state and saturation `_point_value` hands it
                unsplit_pt = CA.TemperatureSpread(
                    thp, ρ, mu_S, FT(0), FT(273.16), FT(150), FT(0.05),
                )
                for n in rule.nodes
                    S′ = CA._point_value(
                        (T, q, s, q_sat) -> q - q_sat - mu_S, n, FT(0), unsplit_pt,
                    )
                    @test S′ ≈ n.S′ atol = 64 * eps(FT) * (FT(0.05) + σ_S)
                end
                # the node means of the linear condensates are the continuous
                # ones, to the accuracy of the Φ approximations in the tail
                # (`truncated_gaussian_condensate` uses `normal_cdf`)
                g = CA.conditional_rule_mean(CA.NodeCondensate(λ, α), rule)
                @test g ≈ q_c rtol = (FT == Float32 ? 3e-3 : 2e-3)
                g_eq = CA.conditional_rule_mean(CA.NodeCondensate(mu_S, one(FT)), rule)
                @test g_eq ≈ q_c_eq rtol = (FT == Float32 ? 3e-3 : 2e-3)
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

    @testset "rule: nearly coincident cloud edges collapse" begin
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
            mu_S = -σ_S / 2
            t_a = -mu_S
            t_p = CA.sgs_precip_shaft_threshold(FT(0.3), σ_S)
            rule_ref = CA.conditional_rule(
                T, σ_S, β_T, t_a, t_a, t_p, FT(150),
            )
            # an equilibrium edge within 10⁻³ σ_S of the actual edge gives the
            # rule of coincident edges, so no node is built from the rounding
            # noise of a sliver interval's moments
            for δ in FT[1e-4, -1e-4, 1e-7]
                rule = CA.conditional_rule(
                    T, σ_S, β_T, t_a, t_a + δ * σ_S, t_p, FT(150),
                )
                @test all(
                    n.S′ == m.S′ && n.w == m.w && n.T == m.T
                    for (n, m) in zip(rule.nodes, rule_ref.nodes)
                )
            end
            # an equilibrium edge well apart from the actual edge keeps its
            # own interval
            rule_apart = CA.conditional_rule(
                T, σ_S, β_T, t_a, t_a + σ_S / 2, t_p, FT(150),
            )
            @test any(
                n.w != m.w for (n, m) in zip(rule_apart.nodes, rule_ref.nodes)
            )
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
                    FT(0.5), FT(0.4), mom.sigma_S, FT(0.1), mom.λ_eq,
                )
                @test out.dq_lcl_dt isa FT
                # a warm cell without ice or snow makes none
                @test out.dq_icl_dt == 0 && out.dq_sno_dt == 0
                uniform(a_p, floor) = CA.microphysics_tendencies_1m(
                    BMT.Microphysics1Moment(), quad, cmp, thp, ρ, T, FT(0.1),
                    q_tot, q_lcl, q_icl, q_rai, q_sno, T′T′, q′q′, corr,
                    mom.λ_lagrange, α, FT(0.1), FT(0.1), FT(180), 2,
                    TD.liquid_fraction(thp, T, q_lcl, q_icl), mu_S,
                    FT(0), a_p, mom.sigma_S, floor, mom.λ_eq,
                )
                out_u = uniform(FT(0.4), FT(0.1))
                # without placement the shaft area and its floor do not enter
                @test out_u == uniform(FT(0.9), FT(0.5))
                # the shaft sits at the moist end, where the cloud is: its rain
                # evaporates less and collects more cloud than uniform rain
                @test out.dq_rai_dt > out_u.dq_rai_dt
                @test out.dq_lcl_dt < out_u.dq_lcl_dt
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
                T_frz + FT(0.2), σ_S, β_T, -λ, -mu_S, t_p, FT(150),
            )
            σ⊥ = CA.sgs_temperature_spread_at_fixed_excess(FT(1), β_T, σ_S)
            @test (β_T * σ_S)^2 + σ⊥^2 ≈ 1
            spread = CA.TemperatureSpread(thp, ρ, mu_S, σ⊥, T_frz, FT(150), FT(0.05))
            one_f = (T, q, s, q_sat) -> one(T)
            T_f = (T, q, s, q_sat) -> T
            S_f = (T, q, s, q_sat) -> q - q_sat - mu_S
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
                    FT(0), FT(0.5), mom.sigma_S, FT(0.5), mom.λ_eq,
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

    @testset "σ_S and β_T where q′ ≈ γ T′ (Float32)" begin
        # A DYCOMS RF01 cloud-layer state with `σ_q = γ σ_T` to four digits
        # and `corr = 1`, at which the two terms of the saturation-excess
        # variance cancel. The density sweep moves `γ σ_T` across `σ_q`, and
        # the last case sets `σ_q = γ σ_T` to rounding. The Cauchy–Schwarz
        # bound `|β_T| σ_S ≤ σ_T` keeps a node at `S′` within `σ_T |S′| / σ_S`
        # of the mean temperature.
        FT = Float32
        thp = TD.Parameters.ThermodynamicsParameters(CP.create_toml_dict(FT))
        T = FT(283.8034)
        T′T′ = FT(6.9575e-3)
        σ_T = sqrt(T′T′)
        q′q′ = FT(2.0623e-9)
        corr = FT(1)
        q_c = FT(3.711e-4)
        λ_eq = FT(1e-4)
        T_min = FT(150)
        ρ_mid = FT(1.13465)
        q′q′_mid = (FT(TD.∂q_vap_sat_∂T(thp, T, ρ_mid)) * σ_T)^2
        for (ρ, qq) in (
            (FT(1.1345), q′q′), (FT(1.1346), q′q′), (FT(1.1347), q′q′),
            (FT(1.1348), q′q′), (FT(1.1349), q′q′), (FT(1.1350), q′q′),
            (ρ_mid, q′q′_mid),
        )
            σ_S = CA._sgs_linear_sigma_S(thp, ρ, T, T′T′, qq, corr)
            β_T = CA.sgs_temperature_regression(thp, T, ρ, σ_S, T′T′, qq, corr)
            @test isfinite(σ_S) && σ_S >= CA.ϵ_numerics(FT)
            @test isfinite(β_T)
            @test abs(β_T) * σ_S <= σ_T * (1 + 2 * eps(FT))
            λ = CA._continuous_lagrange(q_c, σ_S)
            rule = CA.conditional_rule(T, σ_S, β_T, -λ, -λ_eq, FT(Inf), T_min)
            @test sum(n -> n.w, rule.nodes) ≈ 1
            for n in rule.nodes
                @test isfinite(n.T)
                n.w > 0 || continue
                @test abs(n.T - T) <=
                      σ_T * abs(n.S′) / σ_S * (1 + 4 * eps(FT)) + eps(T)
            end
        end
    end

    @testset "σ_S and β_T: Float32 against Float64" begin
        thp32 = TD.Parameters.ThermodynamicsParameters(CP.create_toml_dict(Float32))
        thp64 = TD.Parameters.ThermodynamicsParameters(CP.create_toml_dict(Float64))
        ϵ32 = Float64(CA.ϵ_numerics(Float32))
        # `δ = σ_q / (γ σ_T) − 1` and `m = 1 − corr` run from exact
        # cancellation (`δ = m = 0`) to anticorrelated fluctuations (`m = 2`).
        for T in (230.0, 260.0, 290.0, 310.0), ρ in (0.4, 1.2),
            σ_T in (1e-3, 0.1, 3.0),
            δ in (-0.5, -1e-3, -1e-6, 0.0, 1e-6, 1e-3, 1.0),
            m in (0.0, 1e-6, 1e-3, 0.5, 1.0, 2.0)

            σ_q = TD.∂q_vap_sat_∂T(thp64, T, ρ) * σ_T * (1 + δ)
            (ρ32, T32, T′T′32, q′q′32, corr32) =
                Float32.((ρ, T, σ_T^2, σ_q^2, 1 - m))
            (ρ64, T64, T′T′64, q′q′64, corr64) =
                Float64.((ρ32, T32, T′T′32, q′q′32, corr32))
            σ_S32 = CA._sgs_linear_sigma_S(thp32, ρ32, T32, T′T′32, q′q′32, corr32)
            σ_S64 = CA._sgs_linear_sigma_S(thp64, ρ64, T64, T′T′64, q′q′64, corr64)
            β_T32 = CA.sgs_temperature_regression(
                thp32, T32, ρ32, σ_S32, T′T′32, q′q′32, corr32,
            )
            β_T64 = CA.sgs_temperature_regression(
                thp64, T64, ρ64, σ_S64, T′T′64, q′q′64, corr64,
            )
            @test abs(β_T32) * σ_S32 <= sqrt(T′T′32) * (1 + 2 * eps(Float32))
            @test abs(β_T64) * σ_S64 <= sqrt(T′T′64) * (1 + 2 * eps(Float64))

            # `σ_S` is 1-Lipschitz in `γ σ_T`, so the Float32 error of `γ`
            # from Thermodynamics enters once, next to the rounding of the
            # two variance terms (measured below 0.7 `eps` of
            # `σ_q + γ σ_T` on this grid) and the Float32 floor.
            γ32 = TD.∂q_vap_sat_∂T(thp32, T32, ρ32)
            γ64 = TD.∂q_vap_sat_∂T(thp64, T64, ρ64)
            σ_T64 = sqrt(T′T′64)
            scale = sqrt(q′q′64) + γ64 * σ_T64
            tol = abs(γ32 - γ64) * σ_T64 + 4 * eps(Float32) * scale
            @test abs(σ_S32 - σ_S64) <= tol + ϵ32
            # `|∂β_T/∂(γ σ_T)| ≤ σ_T / σ_S²`, and the factor 2 covers the
            # rounding of `β_T` itself. This first-order bound applies where
            # `σ_S` is well above the perturbation.
            if σ_S64 > 1e-3 * scale
                @test abs(β_T32 - β_T64) <= 2 * tol * σ_T64 / σ_S64^2
            end
        end
    end
end
