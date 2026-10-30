#=
Unit tests for implicit_condensation.jl helpers.

Covers the pure helpers that drive the implicit cloud-liquid condensation
source: dispatch of `_cm_condensation_rate` on `cloud_liquid_formation`,
the sign of the rate across branches, non-negative clamping of the
condensate inputs, and the "no double counting" contract that routes
condensation either through CM's bulk tendency or through the implicit
source, never both.
=#

using Test
using ClimaAtmos

import Thermodynamics as TD
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.MicrophysicsNonEq as CMNe

import ClimaAtmos:
    _cm_condensation_rate, _split_cloud_liquid_formation,
    ImplicitCloudLiquidFormation

# `make_1m_params` builds a 1M parameter set with the given
# `cloud_liquid_formation` option, so we can test the dispatch of
# `_cm_condensation_rate` on `::Nothing` vs `::CloudLiquidFormation`.
function make_1m_params(toml_dict, cloud_liquid_formation)
    return CMP.Microphysics1MParams(toml_dict; cloud_liquid_formation)
end

@testset "Implicit cloud-liquid condensation" begin
    for FT in (Float32, Float64)
        @testset "FT = $FT" begin
            toml_dict = CP.create_toml_dict(FT)
            thp = TD.Parameters.ThermodynamicsParameters(toml_dict)
            mp_on = make_1m_params(toml_dict, CMP.CloudLiquidFormation())
            mp_off = make_1m_params(toml_dict, nothing)

            # Reference supersaturated warm state (expect condensation).
            T = FT(285)
            ρ = FT(1.1)
            q_sat_liq = TD.q_vap_saturation(thp, T, ρ, TD.Liquid())
            q_tot = FT(1.1) * q_sat_liq
            q_lcl = FT(1e-4)
            q_icl = FT(0)
            q_rai = FT(0)
            q_sno = FT(0)

            @testset "Dispatch on cloud_liquid_formation" begin
                # `nothing` → the CM dispatch returns zero at any state.
                @test iszero(
                    _cm_condensation_rate(
                        mp_off, thp, ρ, T, q_tot,
                        q_lcl, q_icl, q_rai, q_sno,
                    ),
                )
                # `CloudLiquidFormation` → the active kernel runs.
                S = _cm_condensation_rate(
                    mp_on, thp, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
                )
                @test S > zero(FT)
                @test isfinite(S)
            end

            @testset "Rate matches CM kernel exactly" begin
                τ = mp_on.process_params.cloud_liquid_formation.τ_relax
                T_hom = mp_on.process_params.cloud_liquid_formation.T_hom
                micro = (;
                    q_tot,
                    q_lcl = max(zero(FT), q_lcl),
                    q_icl = max(zero(FT), q_icl),
                    q_rai = max(zero(FT), q_rai),
                    q_sno = max(zero(FT), q_sno),
                )
                thermo = (; ρ, T)
                S_wrapper = _cm_condensation_rate(
                    mp_on, thp, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
                )
                S_kernel = CMNe._conv_q_vap_to_q_lcl_const(
                    τ, thp, micro, thermo; T_hom,
                )
                @test S_wrapper == S_kernel
            end

            @testset "Subsaturated with q_lcl > 0 is evaporation" begin
                T_sub = FT(285)
                q_sub_sat = TD.q_vap_saturation(thp, T_sub, ρ, TD.Liquid())
                q_tot_sub = FT(0.5) * q_sub_sat + FT(1e-4)  # vapor < q_sat
                S_evap = _cm_condensation_rate(
                    mp_on, thp, ρ, T_sub, q_tot_sub,
                    FT(1e-4), FT(0), FT(0), FT(0),
                )
                @test S_evap < zero(FT)
            end

            @testset "Subsaturated with q_lcl = 0 gives zero" begin
                T_dry = FT(285)
                S_dry = _cm_condensation_rate(
                    mp_on, thp, ρ, T_dry, FT(1e-5),
                    FT(0), FT(0), FT(0), FT(0),
                )
                @test iszero(S_dry)
            end

            @testset "Condensate inputs are clamped non-negative" begin
                # Negative tracers should give the same rate as their
                # clamped-to-zero counterparts (as if CM saw q = 0).
                S_neg = _cm_condensation_rate(
                    mp_on, thp, ρ, T, q_tot,
                    -FT(1e-9), -FT(1e-9), -FT(1e-9), -FT(1e-9),
                )
                S_zero = _cm_condensation_rate(
                    mp_on, thp, ρ, T, q_tot,
                    FT(0), FT(0), FT(0), FT(0),
                )
                @test S_neg == S_zero
            end
        end
    end

    @testset "No double counting: implicit and explicit are mutually exclusive" begin
        # The `_split_cloud_liquid_formation` function translates the
        # user-selected `cloud_liquid_formation` into the pair
        # `(bulk, implicit)` passed to `microphysics_1m_params` and
        # `microphysics_1m_implicit_params`. At most one of the two can be
        # non-`nothing` for any valid input, so CM's bulk tendency and the
        # implicit source never both run condensation.
        bulk_off, impl_off = _split_cloud_liquid_formation(nothing)
        @test isnothing(bulk_off) && isnothing(impl_off)

        bulk_exp, impl_exp =
            _split_cloud_liquid_formation(CMP.CloudLiquidFormation())
        @test bulk_exp isa CMP.CloudLiquidFormation && isnothing(impl_exp)

        bulk_imp, impl_imp =
            _split_cloud_liquid_formation(ImplicitCloudLiquidFormation())
        @test isnothing(bulk_imp) && impl_imp isa CMP.CloudLiquidFormation

        # Contract: for every choice, (bulk, implicit) never both set.
        for clf in (
            nothing,
            CMP.CloudLiquidFormation(),
            ImplicitCloudLiquidFormation(),
        )
            bulk, impl = _split_cloud_liquid_formation(clf)
            @test isnothing(bulk) || isnothing(impl)
        end
    end
end
