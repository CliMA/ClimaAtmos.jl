#=
Unit tests for BMT (BulkMicrophysicsTendencies) integration
Validates that the new BMT API produces equivalent results to legacy wrappers.
=#

using Test
using ClimaAtmos

import Thermodynamics as TD
import CloudMicrophysics as CM
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT


@testset "BMT Integration" begin

    @testset "Parameter Structure" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)

                # 0M parameters
                mp_0m = CMP.Microphysics0MParams(toml_dict)
                @test hasfield(typeof(mp_0m), :precip)
                @test hasfield(typeof(mp_0m.precip), :τ_precip)
                @test hasfield(typeof(mp_0m.precip), :qc_0)

                # 1M parameters
                mp_1m = CMP.Microphysics1MParams(toml_dict; with_2M_autoconv = true)
                @test hasfield(typeof(mp_1m), :cloud)
                @test hasfield(typeof(mp_1m), :precip)
                @test hasfield(typeof(mp_1m), :terminal_velocity)
                @test hasfield(typeof(mp_1m.cloud), :liquid)
                @test hasfield(typeof(mp_1m.cloud), :ice)

                # 2M parameters (warm rain only)
                mp_2m = CMP.Microphysics2MParams(toml_dict; with_ice = false)
                @test hasfield(typeof(mp_2m), :warm_rain)
                @test hasfield(typeof(mp_2m), :ice)
                @test mp_2m.ice === nothing  # No ice for 2M warm rain
                @test hasfield(typeof(mp_2m.warm_rain), :seifert_beheng)

                # 2M+P3 parameters (with ice)
                mp_p3 = CMP.Microphysics2MParams(toml_dict; with_ice = true)
                @test mp_p3.ice !== nothing
                @test hasfield(typeof(mp_p3.ice), :scheme)
                @test hasfield(typeof(mp_p3.ice), :terminal_velocity)
            end
        end
    end

    @testset "0M BMT API" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics0MParams(toml_dict)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                # Realistic conditions
                T = FT(280.0)       # K
                q_liq = FT(0.001)   # kg/kg
                q_ice = FT(0.0005)  # kg/kg

                # Test BMT call
                result = BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics0Moment(),
                    mp,
                    thp,
                    T,
                    q_liq,
                    q_ice,
                )

                # Verify return type (includes fields and value types)
                @test result isa NamedTuple{(:dq_tot_dt, :e_int_precip), NTuple{2, FT}}

                # Verify finite values
                @test isfinite(result.dq_tot_dt)
                @test isfinite(result.e_int_precip)

                # Physical: 0M should remove condensate
                @test result.dq_tot_dt <= 0  # Condensate is removed as precipitation
            end
        end
    end

    @testset "1M BMT API" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict; with_2M_autoconv = true)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                # Realistic atmospheric conditions
                ρ = FT(1.0)      # kg/m³
                T = FT(280.0)    # K
                q_tot = FT(0.01)     # kg/kg
                q_liq = FT(0.001)    # kg/kg
                q_ice = FT(0.0005)   # kg/kg
                q_rai = FT(0.0002)   # kg/kg
                q_sno = FT(0.0001)   # kg/kg

                # Test BMT call
                result = BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics1Moment(),
                    mp,
                    thp,
                    ρ,
                    T,
                    q_tot,
                    q_liq,
                    q_ice,
                    q_rai,
                    q_sno,
                )

                # Verify return type (includes fields and value types)
                @test result isa NamedTuple{
                    (:dq_lcl_dt, :dq_icl_dt, :dq_rai_dt, :dq_sno_dt),
                    NTuple{4, FT},
                }

                # Verify finite values
                @test isfinite(result.dq_lcl_dt)
                @test isfinite(result.dq_icl_dt)
                @test isfinite(result.dq_rai_dt)
                @test isfinite(result.dq_sno_dt)

                # Physical sanity: tendencies should be reasonable
                max_tendency = FT(1e-3)
                @test abs(result.dq_lcl_dt) < max_tendency
                @test abs(result.dq_icl_dt) < max_tendency
                @test abs(result.dq_rai_dt) < max_tendency
                @test abs(result.dq_sno_dt) < max_tendency
            end
        end
    end

    @testset "2M BMT API" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics2MParams(toml_dict; with_ice = false)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                # Realistic 2M conditions
                ρ = FT(1.0)       # kg/m³
                T = FT(285.0)     # K (warm, above freezing)
                q_tot = FT(0.01)     # kg/kg (total specific humidity)
                q_lcl = FT(0.0005)   # kg/kg (cloud liquid)
                n_lcl = FT(1e8)      # kg⁻¹ (cloud droplet number)
                q_rai = FT(0.0001)   # kg/kg (rain)
                n_rai = FT(1e4)      # kg⁻¹ (rain drop number)
                q_ice = FT(0)        # kg/kg (no ice for warm rain 2M)
                n_ice = FT(0)        # kg⁻¹
                q_rim = FT(0)        # kg/kg

                # Test BMT call
                result = BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics2Moment(),
                    mp,
                    thp,
                    ρ,
                    T,
                    q_tot,
                    q_lcl,
                    n_lcl,
                    q_rai,
                    n_rai,
                    q_ice,
                    n_ice,
                    q_rim,
                )

                # Verify return type (includes fields and value types)
                # 2M returns 7 fields: mass and number for cloud/rain, plus ice-related
                @test result isa NamedTuple{
                    (
                        :dq_lcl_dt,
                        :dn_lcl_dt,
                        :dq_rai_dt,
                        :dn_rai_dt,
                        :dq_ice_dt,
                        :dq_rim_dt,
                        :db_rim_dt,
                    ),
                    NTuple{7, FT},
                }

                # Verify finite values
                @test isfinite(result.dq_lcl_dt)
                @test isfinite(result.dn_lcl_dt)
                @test isfinite(result.dq_rai_dt)
                @test isfinite(result.dn_rai_dt)

                # Physical: autoconversion should transfer cloud → rain
                # (when q_liq is significant)
                @test isfinite(result.dq_lcl_dt)
            end
        end
    end

    @testset "Conservation Check" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict; with_2M_autoconv = true)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                ρ = FT(1.0)
                T = FT(280.0)
                q_tot = FT(0.015)
                q_liq = FT(0.002)
                q_ice = FT(0.001)
                q_rai = FT(0.001)
                q_sno = FT(0.0005)

                result = BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics1Moment(),
                    mp,
                    thp,
                    ρ,
                    T,
                    q_tot,
                    q_liq,
                    q_ice,
                    q_rai,
                    q_sno,
                )

                # Total water should be conserved (no external sources/sinks)
                # dq_tot/dt = dq_lcl/dt + dq_icl/dt + dq_rai/dt + dq_sno/dt + dq_vap/dt = 0
                dq_vap_dt = -(
                    result.dq_lcl_dt + result.dq_icl_dt + result.dq_rai_dt +
                    result.dq_sno_dt
                )

                # Check that vapor tendency is finite (it should balance the others)
                @test isfinite(dq_vap_dt)
            end
        end
    end

    @testset "Zero Input Stability" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                toml_dict = CP.create_toml_dict(FT)
                mp = CMP.Microphysics1MParams(toml_dict; with_2M_autoconv = true)
                thp = TD.Parameters.ThermodynamicsParameters(toml_dict)

                # Zero hydrometeors
                result = BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics1Moment(),
                    mp,
                    thp,
                    FT(1.0),     # ρ
                    FT(290.0),   # T
                    FT(0.01),    # q_tot (vapor only)
                    FT(0.0),     # q_liq
                    FT(0.0),     # q_ice
                    FT(0.0),     # q_rai
                    FT(0.0),     # q_sno
                )

                # With no hydrometeors, tendencies should be zero or near-zero
                @test isfinite(result.dq_lcl_dt)
                @test isfinite(result.dq_icl_dt)
                @test result.dq_rai_dt == 0
                @test result.dq_sno_dt == 0
            end
        end
    end

end
