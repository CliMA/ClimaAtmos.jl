#=
Unit tests for the SGS cloud-condensate distributions
(src/parameterized_tendencies/microphysics/sgs_condensate_distribution.jl):
  - ExcessCondensateDistribution reproduces the historical (λ, 1−λ) split
    bitwise, and the constructors without distributions default to it
  - UniformCondensateDistribution puts the subdomain mean of the species at
    every node while the other species keeps its share of the excess: at a dry
    node the uniform species is still there (evaporation/sublimation), at the
    mean node all combinations agree when λ_lagrange = q_c
  - the quadrature wrapper accepts both distributions; with zero variance all
    combinations give the same tendencies
  - the config getters
=#
using Test
using ClimaAtmos
import Thermodynamics as TD
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

const CA = ClimaAtmos

@testset "SGS condensate distribution" begin
    @testset "config getters" begin
        for (getter, key) in (
            (CA.get_sgs_ice_distribution, "sgs_ice_distribution"),
            (CA.get_sgs_liquid_distribution, "sgs_liquid_distribution"),
        )
            @test getter(Dict{String, Any}()) isa CA.ExcessCondensateDistribution
            @test getter(Dict{String, Any}(key => "excess")) isa
                  CA.ExcessCondensateDistribution
            @test getter(Dict{String, Any}(key => "uniform")) isa
                  CA.UniformCondensateDistribution
            # blended: ξ from the parameters (0.5 without them)
            bl = getter(Dict{String, Any}(key => "blended"))
            @test bl isa CA.BlendedCondensateDistribution && bl.uniform_fraction == 0.5
            params = CA.ClimaAtmosParameters(Float32)
            bl = getter(Dict{String, Any}(key => "blended"), params)
            @test bl isa CA.BlendedCondensateDistribution{Float32} &&
                  bl.uniform_fraction == 0.5f0
            @test_throws ErrorException getter(Dict{String, Any}(key => "nope"))
        end
        # the keys are independent
        pa = Dict{String, Any}("sgs_ice_distribution" => "uniform")
        @test CA.get_sgs_ice_distribution(pa) isa CA.UniformCondensateDistribution
        @test CA.get_sgs_liquid_distribution(pa) isa CA.ExcessCondensateDistribution
        w = CA.AtmosWater()
        @test w.sgs_ice_distribution isa CA.ExcessCondensateDistribution
        @test w.sgs_liquid_distribution isa CA.ExcessCondensateDistribution
    end

    for FT in (Float32, Float64)
        @testset "FT = $FT" begin
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            mp = CMP.Microphysics1MParams(toml_dict)
            exc = CA.ExcessCondensateDistribution()
            uni = CA.UniformCondensateDistribution()
            bl0 = CA.BlendedCondensateDistribution(FT(0))
            bl1 = CA.BlendedCondensateDistribution(FT(1))
            blh = CA.BlendedCondensateDistribution(FT(0.25))

            @testset "local condensate split" begin
                for λ in (FT(0), FT(0.3), FT(1)), se in (FT(0), FT(2e-4))
                    q_l, q_i = FT(4e-5), FT(5e-5)
                    ql, qi = CA.sgs_local_cloud_condensate(exc, exc, λ, se, q_l, q_i)
                    @test ql === λ * se
                    @test qi === (FT(1) - λ) * se
                    ql, qi = CA.sgs_local_cloud_condensate(exc, uni, λ, se, q_l, q_i)
                    @test ql === λ * se && qi === q_i
                    ql, qi = CA.sgs_local_cloud_condensate(uni, exc, λ, se, q_l, q_i)
                    @test ql === q_l && qi === (FT(1) - λ) * se
                    ql, qi = CA.sgs_local_cloud_condensate(uni, uni, λ, se, q_l, q_i)
                    @test ql === q_l && qi === q_i
                    # blended ends reproduce the two distributions; the interior is the
                    # linear blend and conserves the species mean at the mean node
                    ql, qi = CA.sgs_local_cloud_condensate(exc, bl0, λ, se, q_l, q_i)
                    @test qi == (FT(1) - λ) * se
                    ql, qi = CA.sgs_local_cloud_condensate(exc, bl1, λ, se, q_l, q_i)
                    @test qi == q_i
                    ql, qi = CA.sgs_local_cloud_condensate(blh, blh, λ, se, q_l, q_i)
                    @test ql ≈ FT(0.75) * λ * se + FT(0.25) * q_l
                    @test qi ≈ FT(0.75) * (FT(1) - λ) * se + FT(0.25) * q_i
                    @test qi isa FT && ql isa FT
                end
            end

            dt = FT(60)
            nsubs = 1
            ρ = FT(0.6)

            # ---- ice-only cell at −25 °C: λ = 0
            T = FT(248)
            q_icl = FT(3e-5)
            q_c = q_icl
            λ_i = FT(TD.liquid_fraction(thp, T, FT(0), q_icl))
            @test λ_i == 0
            q_sat = TD.q_vap_saturation(thp, T, ρ)
            q_tot = q_sat + q_c
            mu_S = q_tot - q_sat
            make(liq, ice) = CA.Microphysics1MEvaluator(
                BMT.Microphysics1Moment(), mp, thp, ρ, FT(0), FT(0),
                λ_i, liq, ice, FT(0), q_icl, q_c, mu_S, FT(1), dt, nsubs, (),
            )
            ev_e = make(exc, exc)
            ev_u = make(exc, uni)
            bulk(q_tot_hat, ql, qi) = BMT.bulk_microphysics_tendencies(
                BMT.LinearizedAverage(),
                BMT.Microphysics1Moment(), mp, thp, ρ, T,
                q_tot_hat, ql, qi, FT(0), FT(0), dt, nsubs,
            )

            @testset "constructors without distributions default to excess" begin
                ev_h = CA.Microphysics1MEvaluator(
                    BMT.Microphysics1Moment(), mp, thp, ρ, FT(0), FT(0),
                    λ_i, q_c, mu_S, FT(1), dt, nsubs, (),
                )
                @test ev_h.ice isa CA.ExcessCondensateDistribution
                @test ev_h.liq isa CA.ExcessCondensateDistribution
                @test ev_h.q_icl == 0 && ev_h.q_lcl == 0
                @test ev_h.λ === λ_i
                for q̂ in (q_tot, q_tot - FT(2) * q_c, q_tot + q_c)
                    @test ev_h(T, q̂) === ev_e(T, q̂)
                end
            end

            @testset "ice: mean node, both distributions hold q_icl" begin
                ref = bulk(q_tot, FT(0), q_icl)
                for ev in (ev_e, ev_u)
                    out = ev(T, q_tot)
                    @test out.dq_icl_dt ≈ ref.dq_icl_dt rtol = FT(1e-4)
                    @test out.dq_lcl_dt ≈ ref.dq_lcl_dt rtol = FT(1e-4)
                end
            end

            @testset "ice: dry node sublimates when uniform, absent when excess" begin
                q̂ = FT(0.9) * TD.q_vap_saturation(thp, T, ρ, TD.Ice())
                @test q_c + q̂ - q_tot < 0   # shifted_excess = 0
                out_e = ev_e(T, q̂)
                out_u = ev_u(T, q̂)
                @test out_e.dq_icl_dt ≈ bulk(q̂, FT(0), FT(0)).dq_icl_dt rtol = FT(1e-4)
                @test out_u.dq_icl_dt ≈ bulk(q̂, FT(0), q_icl).dq_icl_dt rtol = FT(1e-4)
                @test out_e.dq_icl_dt ≈ 0 atol = eps(FT)
                @test out_u.dq_icl_dt < 0
            end

            @testset "ice: wet node, uniform ice does not grow with the excess" begin
                q̂ = q_tot + FT(2) * q_c   # shifted_excess = 3 q_c
                @test ev_e(T, q̂).dq_icl_dt ≈ bulk(q̂, FT(0), FT(3) * q_c).dq_icl_dt rtol =
                    FT(1e-4)
                @test ev_u(T, q̂).dq_icl_dt ≈ bulk(q̂, FT(0), q_icl).dq_icl_dt rtol = FT(1e-4)
            end

            # ---- liquid-only warm cell: λ = 1
            T_w = FT(285)
            q_lcl = FT(2e-4)
            λ_w = FT(TD.liquid_fraction(thp, T_w, q_lcl, FT(0)))
            @test λ_w == 1
            q_sat_w = TD.q_vap_saturation(thp, T_w, ρ)
            q_tot_w = q_sat_w + q_lcl
            mu_w = q_tot_w - q_sat_w
            make_w(liq, ice) = CA.Microphysics1MEvaluator(
                BMT.Microphysics1Moment(), mp, thp, ρ, FT(0), FT(0),
                λ_w, liq, ice, q_lcl, FT(0), q_lcl, mu_w, FT(1), dt, nsubs, (),
            )
            bulk_w(q_tot_hat, ql, qi) = BMT.bulk_microphysics_tendencies(
                BMT.LinearizedAverage(),
                BMT.Microphysics1Moment(), mp, thp, ρ, T_w,
                q_tot_hat, ql, qi, FT(0), FT(0), dt, nsubs,
            )
            @testset "liquid: mean node, both distributions hold q_lcl" begin
                ref = bulk_w(q_tot_w, q_lcl, FT(0))
                for ev in (make_w(exc, exc), make_w(uni, exc), make_w(uni, uni))
                    @test ev(T_w, q_tot_w).dq_lcl_dt ≈ ref.dq_lcl_dt rtol = FT(1e-4)
                end
            end
            @testset "liquid: dry node evaporates when uniform, absent when excess" begin
                q̂ = FT(0.8) * q_sat_w   # shifted_excess = 0, subsaturated over liquid
                out_e = make_w(exc, exc)(T_w, q̂)
                out_u = make_w(uni, exc)(T_w, q̂)
                @test out_e.dq_lcl_dt ≈ bulk_w(q̂, FT(0), FT(0)).dq_lcl_dt rtol = FT(1e-4)
                @test out_u.dq_lcl_dt ≈ bulk_w(q̂, q_lcl, FT(0)).dq_lcl_dt rtol = FT(1e-4)
                @test out_e.dq_lcl_dt ≈ 0 atol = eps(FT)
                @test out_u.dq_lcl_dt < 0   # evaporation of the mean liquid
            end
            @testset "liquid: wet node, uniform liquid does not grow with the excess" begin
                q̂ = q_tot_w + FT(2) * q_lcl   # shifted_excess = 3 q_lcl
                @test make_w(exc, exc)(T_w, q̂).dq_lcl_dt ≈
                      bulk_w(q̂, FT(3) * q_lcl, FT(0)).dq_lcl_dt rtol = FT(1e-4)
                @test make_w(uni, exc)(T_w, q̂).dq_lcl_dt ≈ bulk_w(q̂, q_lcl, FT(0)).dq_lcl_dt rtol =
                    FT(1e-4)
                # the vapour excess condenses onto the mean liquid instead
                @test make_w(uni, exc)(T_w, q̂).dq_lcl_dt > 0
            end

            @testset "mixed cell: each species follows its own distribution" begin
                T_m = FT(263)
                q_l, q_i = FT(4e-5), FT(2e-5)
                λ_m = FT(TD.liquid_fraction(thp, T_m, q_l, q_i))
                @test λ_m ≈ q_l / (q_l + q_i)
                q_sat_m = TD.q_vap_saturation(thp, T_m, ρ)
                q_tot_m = q_sat_m + q_l + q_i
                mu_m = q_tot_m - q_sat_m
                mk(liq, ice) = CA.Microphysics1MEvaluator(
                    BMT.Microphysics1Moment(), mp, thp, ρ, FT(0), FT(0),
                    λ_m, liq, ice, q_l, q_i, q_l + q_i, mu_m, FT(1), dt, nsubs, (),
                )
                q̂ = q_tot_m + (q_l + q_i)   # shifted_excess = 2 q_c
                se = FT(2) * (q_l + q_i)
                bulk_m(ql, qi) = BMT.bulk_microphysics_tendencies(
                    BMT.LinearizedAverage(),
                    BMT.Microphysics1Moment(), mp, thp, ρ, T_m,
                    q̂, ql, qi, FT(0), FT(0), dt, nsubs,
                )
                cases = (
                    ((exc, exc), (λ_m * se, (FT(1) - λ_m) * se)),
                    ((exc, uni), (λ_m * se, q_i)),
                    ((uni, exc), (q_l, (FT(1) - λ_m) * se)),
                    ((uni, uni), (q_l, q_i)),
                )
                for ((liq, ice), (ql, qi)) in cases
                    out = mk(liq, ice)(T_m, q̂)
                    ref = bulk_m(ql, qi)
                    @test out.dq_lcl_dt ≈ ref.dq_lcl_dt rtol = FT(1e-4)
                    @test out.dq_icl_dt ≈ ref.dq_icl_dt rtol = FT(1e-4)
                end
            end

            @testset "quadrature wrapper: zero variance ⇒ all combinations agree" begin
                quad = CA.SGSQuadrature(FT; quadrature_order = 3)
                T_m = FT(263)
                q_l, q_i = FT(4e-5), FT(2e-5)
                λ_m = FT(TD.liquid_fraction(thp, T_m, q_l, q_i))
                q_tot_m = TD.q_vap_saturation(thp, T_m, ρ) + q_l + q_i
                mu_m = q_tot_m - TD.q_vap_saturation(thp, T_m, ρ)
                args = (
                    BMT.Microphysics1Moment(), quad, mp, thp, ρ, T_m, q_tot_m,
                    q_l, q_i, FT(0), FT(0),
                    FT(0), FT(0), FT(0),        # T′T′, q′q′, corr
                    q_l + q_i, FT(1), dt, nsubs,
                )
                out_e = CA.microphysics_tendencies_1m(args..., λ_m, mu_m, exc, exc)
                for (ice, liq) in ((uni, exc), (exc, uni), (uni, uni))
                    out = CA.microphysics_tendencies_1m(args..., λ_m, mu_m, ice, liq)
                    @test out.dq_icl_dt ≈ out_e.dq_icl_dt rtol = FT(1e-4)
                    @test out.dq_lcl_dt ≈ out_e.dq_lcl_dt rtol = FT(1e-4)
                end
                # the default (no distributions given) is the excess split
                out_d = CA.microphysics_tendencies_1m(args..., λ_m, mu_m)
                @test out_d.dq_icl_dt === out_e.dq_icl_dt
                @test out_d.dq_lcl_dt === out_e.dq_lcl_dt
                # with variance the combinations differ and stay finite
                argv = (
                    args[1:11]...,
                    FT(0),
                    (FT(0.3) * q_tot_m)^2,
                    FT(0),
                    q_l + q_i,
                    FT(1),
                    dt,
                    nsubs,
                )
                outv_e = CA.microphysics_tendencies_1m(argv..., λ_m, mu_m, exc, exc)
                outv_u = CA.microphysics_tendencies_1m(argv..., λ_m, mu_m, uni, uni)
                @test isfinite(outv_u.dq_icl_dt) && isfinite(outv_u.dq_lcl_dt)
                @test outv_u.dq_icl_dt != outv_e.dq_icl_dt
                @test outv_u.dq_lcl_dt != outv_e.dq_lcl_dt
            end
        end
    end
end
