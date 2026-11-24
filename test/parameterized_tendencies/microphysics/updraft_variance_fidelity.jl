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

function allocs_updraft(α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max)
    rule(α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max)
    return @allocated rule(α, ups, σ_S, thp, mm, tcp, σ_abs, c_u, α_min, α_max)
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
    end

    @testset "no allocations" begin
        FT = Float64
        params = CA.ClimaAtmosParameters(FT)
        thp = CAP.thermodynamics_params(params)
        ups = (updraft(thp, 0.1, 1e-4, 0), updraft(thp, 0.2, 2e-4, 1e-5; δ = 1e-5))
        @test allocs_updraft(
            FT(1), ups, FT(3e-4), thp, CA.NonEquilibriumMicrophysics2M(),
            CAP.turbconv_params(params), CAP.cloud_fraction_sigma_abs(params),
            FT(0.6), FT(0.05), FT(2),
        ) == 0
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
