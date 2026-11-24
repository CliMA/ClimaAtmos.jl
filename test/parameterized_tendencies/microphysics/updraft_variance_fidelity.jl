#=
Unit tests for the updraft rule of the variance fidelity
(`sgs_variance_fidelity_model: updraft`, `updraft_variance_fidelity`):

  - values, the clamps (`α_min` and the equilibrium bound `1/(1 − w_eq)`), the
    fallback to the equilibrium rule without cloudy updrafts, the weight from
    the updraft cloud cover and the lower area-limiter factor, and several
    updrafts,
  - config parsing and parameters,
  - no allocations,
  - dispatch: a no-op outside its models.
=#
using Test
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD

# One updraft of area `a` at `(T, ρ)` with cloud condensate `(q_lcl, q_icl)`
# and vapor `δ` below saturation (`δ = 0`: saturated), as the tuple of the
# per-updraft arguments of `updraft_variance_fidelity`.
function updraft(thp, a, q_lcl, q_icl; T = 290, ρ = 1.1, δ = 0)
    FT = typeof(TD.Parameters.R_d(thp))
    (T, ρ, q_lcl, q_icl) = FT.((T, ρ, q_lcl, q_icl))
    q_sat = TD.q_vap_saturation(thp, T, ρ, max(q_lcl, 0), max(q_icl, 0))
    q_tot = q_sat - FT(δ) + q_lcl + q_icl
    sgs = (; ρa = FT(a) * ρ, q_lcl, q_icl)
    return (; sgs, ρ, T, q_tot, q_liq = q_lcl, q_ice = q_icl)
end

rule(α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max) =
    CA.updraft_variance_fidelity(
        α, map(u -> u.sgs, ups), map(u -> u.ρ, ups), map(u -> u.T, ups),
        map(u -> u.q_tot, ups), map(u -> u.q_liq, ups), map(u -> u.q_ice, ups),
        σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
    )
rule(
    α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
    c_u_strat, φ_ref, q_c_env, T_mean,
) = CA.updraft_variance_fidelity(
    α, map(u -> u.sgs, ups), map(u -> u.ρ, ups), map(u -> u.T, ups),
    map(u -> u.q_tot, ups), map(u -> u.q_liq, ups), map(u -> u.q_ice, ups),
    σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
    c_u_strat, φ_ref, q_c_env, T_mean,
)

rule(
    α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
    c_u_strat, φ_ref, q_c_env, T_mean,
    T_warm_start, T_warm_end, inv_blend,
) = CA.updraft_variance_fidelity(
    α, map(u -> u.sgs, ups), map(u -> u.ρ, ups), map(u -> u.T, ups),
    map(u -> u.q_tot, ups), map(u -> u.q_liq, ups), map(u -> u.q_ice, ups),
    σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
    c_u_strat, φ_ref, q_c_env, T_mean,
    T_warm_start, T_warm_end, inv_blend,
)

function allocs_updraft(α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max)
    rule(α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max)
    return @allocated rule(α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max)
end
function allocs_updraft(
    α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
    c_u_strat, φ_ref, q_c_env, T_mean,
)
    rule(
        α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
        c_u_strat, φ_ref, q_c_env, T_mean,
    )
    return @allocated rule(
        α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
        c_u_strat, φ_ref, q_c_env, T_mean,
    )
end
function allocs_updraft(
    α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
    c_u_strat, φ_ref, q_c_env, T_mean,
    T_warm_start, T_warm_end, inv_blend,
)
    rule(
        α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
        c_u_strat, φ_ref, q_c_env, T_mean,
        T_warm_start, T_warm_end, inv_blend,
    )
    return @allocated rule(
        α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max,
        c_u_strat, φ_ref, q_c_env, T_mean,
        T_warm_start, T_warm_end, inv_blend,
    )
end

@testset "Updraft variance fidelity" begin

    @testset "rule: values, clamps, fallback, weight, several updrafts" begin
        for FT in (Float32, Float64)
            params = CA.ClimaAtmosParameters(FT)
            thp = CAP.thermodynamics_params(params)
            tcp = CAP.turbconv_params(params)
            a_min = CAP.min_area(tcp)
            σ_abs = CAP.cloud_fraction_sigma_abs(params)
            mm = CA.NonEquilibriumMicrophysics2M()
            c_u, α_min, α_max = FT(0.6), FT(0.05), FT(2)
            α_eq = FT(1.3)
            r(ups, σ_S; α_max = α_max) =
                rule(α_eq, ups, FT(σ_S), thp, mm, tcp, σ_abs, c_u, α_min, α_max)
            cover(u) = CA._updraft_cloud_fraction(
                thp, u.T, u.ρ, u.q_tot, u.q_liq, u.q_ice, u.sgs.q_lcl, u.sgs.q_icl,
                σ_abs, mm,
            )
            L(a) = CA.lower_area_limiter_factor(FT(a), tcp)
            blend(W, α_up) = (1 - W) * α_eq + W * α_up

            # one saturated updraft: cover 1, weight L(a), α σ_S = c_u q_c
            u = updraft(thp, 0.1, 1e-4, 2e-5)
            @test cover(u) ≈ 1
            @test L(0.1) ≈ 1 - a_min / FT(0.1)
            @test r((u,), 3e-4) ≈ blend(L(0.1), c_u * FT(1.2e-4) / FT(3e-4)) rtol =
                100 * eps(FT)
            @test r((u,), 3e-4) ≈ FT(0.24) rtol = FT(1e-3)
            @test r((u,), 3e-4) isa FT
            # clamps: α_min for little condensate, the equilibrium bound for a
            # narrow PDF, none for `w_eq = 1` (`α_max = Inf`)
            @test r((updraft(thp, 0.1, 2e-5, 0),), 3e-4) ≈ blend(L(0.1), α_min)
            @test r((updraft(thp, 0.1, 1e-3, 0),), 1e-5) ≈ blend(L(0.1), α_max)
            @test r((updraft(thp, 0.1, 1e-3, 0),), 0) ≈ blend(L(0.1), α_max)
            @test r((updraft(thp, 0.1, 1e-3, 0),), 2e-4; α_max = FT(Inf)) ≈
                  blend(L(0.1), FT(3)) rtol = 100 * eps(FT)
            # the equilibrium rule, exactly: no condensate, no area, or an
            # updraft at the minimum area
            @test r((updraft(thp, 0.1, 0, 0),), 3e-4) == α_eq
            @test r((updraft(thp, 0.1, -1e-6, 0),), 3e-4) == α_eq
            @test r((updraft(thp, 0, 1e-3, 0),), 3e-4) == α_eq
            @test r((updraft(thp, 0.5 * a_min, 1e-3, 0),), 3e-4) == α_eq
            # a subsaturated updraft with trace condensate (below its LCL)
            # has almost no cover, so it barely moves α
            sub = updraft(thp, 0.1, 1e-6, 0; δ = 1e-3)
            @test cover(sub) < FT(0.01)
            @test abs(r((sub,), 3e-4) - α_eq) < FT(0.01) * abs(α_eq - α_min)
            # partial cover: the condensate is the in-cloud one, q_c / f,
            # and the weight f L(a)
            part = updraft(thp, 0.1, 1e-5, 0; δ = 1e-5)
            f = cover(part)
            @test FT(0.1) < f < FT(0.99)
            @test r((part,), 1e-5) ≈ blend(
                f * L(0.1), clamp(c_u * FT(1e-5) / f / FT(1e-5), α_min, α_max),
            ) rtol = 100 * eps(FT)
            # the weight vanishes linearly at the minimum area
            near = updraft(thp, 1.01 * a_min, 3e-4, 0)
            @test r((near,), 2e-4) ≈ blend(L(1.01 * a_min), FT(0.9)) rtol =
                100 * eps(FT)
            @test L(1.01 * a_min) < FT(0.01)
            # two updrafts: area-weighted in-cloud condensate, and the
            # area-weighted mean of f L as the weight
            u2 = updraft(thp, 0.3, 2e-4, 0; ρ = 0.9)
            W2 = (FT(0.1) * L(0.1) + FT(0.3) * L(0.3)) / FT(0.4)
            @test r((updraft(thp, 0.1, 1e-4, 0), u2), 3e-4) ≈
                  blend(W2, c_u * FT(1.75e-4) / FT(3e-4)) rtol = 100 * eps(FT)
            # a clear updraft adds area but no cover, halving the weight here
            dry = updraft(thp, 0.1, 0, 0)
            @test r((updraft(thp, 0.1, 1e-3, 0), dry), 5e-4) ≈
                  blend(L(0.1) / 2, FT(1.2)) rtol = 100 * eps(FT)
        end
    end

    @testset "regime-selective coefficient" begin
        for FT in (Float32, Float64)
            params = CA.ClimaAtmosParameters(FT)
            thp = CAP.thermodynamics_params(params)
            tcp = CAP.turbconv_params(params)
            σ_abs = CAP.cloud_fraction_sigma_abs(params)
            mm = CA.NonEquilibriumMicrophysics2M()
            c_u_conv, c_u_strat = FT(2.5), FT(0.75)
            φ_ref = FT(0.35)
            α_min, α_max, α_eq = FT(0.05), FT(4), FT(1.3)
            σ_S = FT(5e-4)
            u = updraft(thp, 0.1, 2e-4, 0)
            aq = FT(0.1) * FT(2e-4)
            a = FT(0.1)

            # Equal coefficients reproduce the uniform rule bitwise identically
            base = rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs, c_u_conv, α_min, α_max,
            )
            for (q_c_env, T_mean) in
                ((FT(0), FT(290)), (FT(5e-4), FT(290)), (FT(5e-4), FT(250)))
                @test rule(
                    α_eq, (u,), σ_S, thp, mm, tcp, σ_abs,
                    c_u_conv, α_min, α_max, c_u_conv, φ_ref, q_c_env, T_mean,
                ) === base
            end

            # Cold environment (T <= 263.15 K): w_warm = 0 -> c_u,conv
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, φ_ref, aq, a, FT(1e-3), FT(250),
            ) === c_u_conv
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, FT(1e3), aq, a, FT(1e-3), FT(263.15),
            ) === c_u_conv
            @test rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs,
                c_u_conv, α_min, α_max, c_u_strat, φ_ref, FT(1e-3), FT(250),
            ) === base

            # Warm convective environment (φ_up >= φ_ref): w_st = 0 -> c_u,conv
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, φ_ref, aq, a, FT(0), FT(290),
            ) === c_u_conv
            @test rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs,
                c_u_conv, α_min, α_max, c_u_strat, φ_ref, FT(0), FT(290),
            ) === base

            # Warm stratiform environment: φ_up = φ_ref / 2 -> midpoint coeff
            q_c_mid = aq * (inv(φ_ref / 2) - 1) / (1 - a)
            c_mid = (c_u_conv + c_u_strat) / 2
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, φ_ref, aq, a, q_c_mid, FT(290),
            ) ≈ c_mid rtol = 100 * eps(FT)
            @test rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs,
                c_u_conv, α_min, α_max, c_u_strat, φ_ref, q_c_mid, FT(290),
            ) ≈ rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs, c_mid, α_min, α_max,
            ) rtol = 100 * eps(FT)

            # Warm environment with large φ_ref reduces weight to w_warm alone
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, FT(Inf), aq, a, FT(0), FT(290),
            ) === c_u_strat
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, FT(1e3), aq, a, FT(0), FT(290),
            ) ≈ c_u_strat rtol = FT(3e-3)
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, FT(Inf), aq, a, FT(0), FT(268.15),
            ) ≈ c_mid rtol = 100 * eps(FT)
            @test rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs,
                c_u_conv, α_min, α_max, c_u_strat, FT(Inf), FT(0), FT(290),
            ) === rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs, c_u_strat, α_min, α_max,
            )

            # Edge cases: zero condensate, negative q_c_env, a >= 1
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, φ_ref, FT(0), a, FT(0), FT(290),
            ) === c_u_strat
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, φ_ref, aq, a, FT(-1e-6), FT(290),
            ) === c_u_conv
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, φ_ref, aq, FT(1.1), FT(1e-3), FT(290),
            ) === c_u_conv

            # Default ramp bounds + linear blend match 19-arg call bitwise
            @test rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs,
                c_u_conv, α_min, α_max, c_u_strat, φ_ref, q_c_mid, FT(268.15),
                FT(263.15), FT(273.15), false,
            ) === rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs,
                c_u_conv, α_min, α_max, c_u_strat, φ_ref, q_c_mid, FT(268.15),
            )

            # Custom warm-ramp bounds (243.15 - 253.15 K)
            T0, T1 = FT(243.15), FT(253.15)
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, FT(Inf), aq, a, FT(0), FT(243.15),
                T0, T1, false,
            ) === c_u_conv
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, FT(Inf), aq, a, FT(0), FT(253.15),
                T0, T1, false,
            ) === c_u_strat
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, FT(Inf), aq, a, FT(0), FT(272),
                T0, T1, false,
            ) === c_u_strat
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, FT(Inf), aq, a, FT(0), FT(248.15),
                T0, T1, false,
            ) ≈ c_mid rtol = 100 * eps(FT)

            # Inverse (1 / c_u) blend: endpoints match, midpoint is harmonic
            c_harm = inv((inv(c_u_conv) + inv(c_u_strat)) / 2)
            @test c_harm < c_mid
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, φ_ref, aq, a, FT(0), FT(290),
                T0, T1, true,
            ) ≈ c_u_conv rtol = 10 * eps(FT)
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, FT(Inf), aq, a, FT(0), FT(290),
                T0, T1, true,
            ) ≈ c_u_strat rtol = 10 * eps(FT)
            @test CA._updraft_fidelity_coeff(
                c_u_conv, c_u_strat, φ_ref, aq, a, q_c_mid, FT(290),
                T0, T1, true,
            ) ≈ c_harm rtol = 100 * eps(FT)
            @test rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs,
                c_u_conv, α_min, α_max, c_u_strat, φ_ref, q_c_mid, FT(290),
                T0, T1, true,
            ) ≈ rule(
                α_eq, (u,), σ_S, thp, mm, tcp, σ_abs, c_harm, α_min, α_max,
            ) rtol = 100 * eps(FT)
        end
    end

    @testset "config and parameters" begin
        @test CA.get_sgs_variance_fidelity_model(Dict()) ==
              CA.EquilibriumVarianceFidelity()
        @test CA.get_sgs_variance_fidelity_model(
            Dict("sgs_variance_fidelity_model" => "updraft"),
        ) == CA.UpdraftVarianceFidelity()
        @test_throws ErrorException CA.get_sgs_variance_fidelity_model(
            Dict("sgs_variance_fidelity_model" => "other"),
        )
        @test CA.get_sgs_variance_fidelity_model(CA.default_config_dict()) ==
              CA.EquilibriumVarianceFidelity()
        @test CA.AtmosWater().sgs_variance_fidelity_model ==
              CA.EquilibriumVarianceFidelity()
        for FT in (Float32, Float64)
            params = CA.ClimaAtmosParameters(FT)
            @test CAP.sgs_fidelity_updraft_coeff(params) == FT(0.6)
            @test CAP.sgs_fidelity_updraft_coeff_stratiform(params) == FT(-1)
            @test CAP.sgs_fidelity_updraft_share_ref(params) == FT(0.35)
            @test CAP.sgs_fidelity_updraft_warm_ramp_start(params) == FT(263.15)
            @test CAP.sgs_fidelity_updraft_warm_ramp_end(params) == FT(273.15)
            @test CAP.sgs_fidelity_updraft_inv_blend(params) == FT(0)
            @test CAP.sgs_fidelity_updraft_min(params) == FT(0.05)
        end
        # `w_eq = 1` is allowed (no upper bound on α); `w_eq > 1` is not
        for (w_eq, ok) in ((1.0, true), (1.5, false))
            mktemp() do path, io
                write(
                    io,
                    """
                    [sgs_fidelity_equilibrium_weight]
                    value = $w_eq
                    type = "float"
                    """,
                )
                flush(io)
                config = CA.AtmosConfig(
                    Dict("toml" => [path]);
                    job_id = "updraft_fidelity_w_eq",
                )
                if ok
                    params = CA.ClimaAtmosParameters(config)
                    @test CAP.sgs_fidelity_equilibrium_weight(params) == w_eq
                else
                    @test_throws ArgumentError CA.ClimaAtmosParameters(config)
                end
            end
        end
        # `sgs_fidelity_updraft_coeff_stratiform`: -1 (equal to c_u) or >= 0
        for (val, ok) in ((-1.0, true), (0.0, true), (0.75, true), (-0.5, false))
            mktemp() do path, io
                write(
                    io,
                    """
                    [sgs_fidelity_updraft_coeff_stratiform]
                    value = $val
                    type = "float"
                    """,
                )
                flush(io)
                config = CA.AtmosConfig(
                    Dict("toml" => [path]);
                    job_id = "updraft_fidelity_c_u_strat",
                )
                if ok
                    params = CA.ClimaAtmosParameters(config)
                    @test CAP.sgs_fidelity_updraft_coeff_stratiform(params) ==
                          eltype(params)(val)
                else
                    @test_throws ArgumentError CA.ClimaAtmosParameters(config)
                end
            end
        end
        # `sgs_fidelity_updraft_share_ref`: strictly positive
        for (val, ok) in ((0.35, true), (1e3, true), (0.0, false), (-0.1, false))
            mktemp() do path, io
                write(
                    io,
                    """
                    [sgs_fidelity_updraft_share_ref]
                    value = $val
                    type = "float"
                    """,
                )
                flush(io)
                config = CA.AtmosConfig(
                    Dict("toml" => [path]);
                    job_id = "updraft_fidelity_phi_ref",
                )
                if ok
                    params = CA.ClimaAtmosParameters(config)
                    @test CAP.sgs_fidelity_updraft_share_ref(params) ==
                          eltype(params)(val)
                else
                    @test_throws ArgumentError CA.ClimaAtmosParameters(config)
                end
            end
        end
        # `sgs_fidelity_updraft_warm_ramp_start` < `_end`, both positive
        for (t0, t1, ok) in (
            (243.15, 253.15, true),
            (273.15, 273.15, false),
            (280.0, 270.0, false),
            (0.0, 250.0, false),
        )
            mktemp() do path, io
                write(
                    io,
                    """
                    [sgs_fidelity_updraft_warm_ramp_start]
                    value = $t0
                    type = "float"
                    [sgs_fidelity_updraft_warm_ramp_end]
                    value = $t1
                    type = "float"
                    """,
                )
                flush(io)
                config = CA.AtmosConfig(
                    Dict("toml" => [path]);
                    job_id = "updraft_fidelity_warm_ramp",
                )
                if ok
                    params = CA.ClimaAtmosParameters(config)
                    @test CAP.sgs_fidelity_updraft_warm_ramp_start(params) ==
                          eltype(params)(t0)
                    @test CAP.sgs_fidelity_updraft_warm_ramp_end(params) ==
                          eltype(params)(t1)
                else
                    @test_throws ArgumentError CA.ClimaAtmosParameters(config)
                end
            end
        end
        # `sgs_fidelity_updraft_inv_blend`: 0 or 1, and positive coeffs for 1
        for (val, c_strat, ok) in (
            (0.0, -1.0, true),
            (1.0, 0.75, true),
            (0.5, 0.75, false),
            (1.0, 0.0, false),
        )
            mktemp() do path, io
                write(
                    io,
                    """
                    [sgs_fidelity_updraft_inv_blend]
                    value = $val
                    type = "float"
                    [sgs_fidelity_updraft_coeff_stratiform]
                    value = $c_strat
                    type = "float"
                    """,
                )
                flush(io)
                config = CA.AtmosConfig(
                    Dict("toml" => [path]);
                    job_id = "updraft_fidelity_inv_blend",
                )
                if ok
                    params = CA.ClimaAtmosParameters(config)
                    @test CAP.sgs_fidelity_updraft_inv_blend(params) ==
                          eltype(params)(val)
                else
                    @test_throws ArgumentError CA.ClimaAtmosParameters(config)
                end
            end
        end
    end

    @testset "no allocations" begin
        for FT in (Float32, Float64)
            params = CA.ClimaAtmosParameters(FT)
            thp = CAP.thermodynamics_params(params)
            ups = (
                updraft(thp, 0.1, 1e-4, 0),
                updraft(thp, 0.2, 2e-4, 1e-5; δ = 1e-5),
            )
            @test allocs_updraft(
                FT(1), ups, FT(3e-4), thp, CA.NonEquilibriumMicrophysics2M(),
                CAP.turbconv_params(params), CAP.cloud_fraction_sigma_abs(params),
                FT(0.6), FT(0.05), FT(2),
            ) == 0
            @test allocs_updraft(
                FT(1), ups, FT(3e-4), thp, CA.NonEquilibriumMicrophysics2M(),
                CAP.turbconv_params(params), CAP.cloud_fraction_sigma_abs(params),
                FT(2.5), FT(0.05), FT(4), FT(0.75), FT(0.35), FT(2e-4), FT(290),
            ) == 0
            @test allocs_updraft(
                FT(1), ups, FT(3e-4), thp, CA.NonEquilibriumMicrophysics2M(),
                CAP.turbconv_params(params), CAP.cloud_fraction_sigma_abs(params),
                FT(2.5), FT(0.05), FT(4), FT(0.75), FT(0.6), FT(2e-4), FT(272),
                FT(243.15), FT(253.15), true,
            ) == 0
        end
    end

    @testset "dispatch: a no-op outside its models" begin
        # The rule acts only for `UpdraftVarianceFidelity` with
        # `PrognosticEDMFX` and non-equilibrium microphysics; any other
        # combination returns before touching its arguments.
        @test isnothing(
            CA.set_updraft_variance_fidelity!(
                CA.EquilibriumVarianceFidelity(), nothing, nothing, :Y, :p,
            ),
        )
        @test isnothing(
            CA.set_updraft_variance_fidelity!(
                CA.UpdraftVarianceFidelity(), nothing,
                CA.NonEquilibriumMicrophysics2M(), :Y, :p,
            ),
        )
        @test isnothing(
            CA.set_updraft_variance_fidelity!(
                CA.UpdraftVarianceFidelity(), nothing,
                CA.EquilibriumMicrophysics0M(), :Y, :p,
            ),
        )
    end
end
