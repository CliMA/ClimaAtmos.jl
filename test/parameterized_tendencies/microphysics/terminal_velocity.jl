#=
Terminal velocity modes: selection from the config keys, and each (mode, species) closure
against its CloudMicrophysics reference.
=#

using Test
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.MicrophysicsNonEq as CMNe
import ClimaCore.MatrixFields: @name

@testset "Terminal velocity modes" begin
    for FT in (Float32, Float64)
        @testset "FT = $FT" begin
            base = Dict(
                "microphysics_model" => "1M",
                "FLOAT_TYPE" => string(FT),
                "output_default_diagnostics" => false,
            )
            make_water(extra) = begin
                config =
                    CA.AtmosConfig(merge(base, extra); job_id = "terminal_velocity_$FT")
                params = CA.ClimaAtmosParameters(config)
                (; water = CA.AtmosWater(config, params, FT), params)
            end

            @testset "Mode selection" begin
                (; water) = make_water(Dict())
                @test water.terminal_velocity_liquid isa CA.FixedTerminalVelocity
                @test water.terminal_velocity_ice isa CA.FixedTerminalVelocity
                @test water.terminal_velocity_rain isa CA.FixedTerminalVelocity
                @test water.terminal_velocity_snow isa CA.FixedTerminalVelocity

                (; water) = make_water(
                    Dict(
                        "cloud_liquid_terminal_velocity" => "Stokes",
                        "cloud_ice_terminal_velocity" => "Chen2022",
                        "rain_terminal_velocity" => "Chen2022",
                        "snow_terminal_velocity" => "Blk1M",
                    ),
                )
                @test water.terminal_velocity_liquid isa CA.StokesTerminalVelocity
                @test water.terminal_velocity_ice isa CA.Chen2022TerminalVelocity
                @test water.terminal_velocity_rain isa CA.Chen2022TerminalVelocity
                @test water.terminal_velocity_snow isa CA.Blk1MTerminalVelocity

                (; water) = make_water(
                    Dict(
                        "rain_terminal_velocity" => "Fixed",
                        "snow_terminal_velocity" => "Chen2022",
                    ),
                )
                @test water.terminal_velocity_rain isa CA.FixedTerminalVelocity
                @test water.terminal_velocity_snow isa CA.Chen2022TerminalVelocity

                # closures a species does not support are rejected
                @test_throws ErrorException make_water(
                    Dict("cloud_liquid_terminal_velocity" => "Blk1M"),
                )
                @test_throws ErrorException make_water(
                    Dict("cloud_ice_terminal_velocity" => "Stokes"),
                )
            end

            @testset "Closures" begin
                (; water, params) = make_water(Dict())
                mm = water.microphysics_model
                cmc = CAP.microphysics_cloud_params(params)
                cmp = CAP.microphysics_1m_params(params)
                w(mode, name, ρ, q) =
                    CA.terminal_velocity(mm, mode, name, params, cmc, cmp, FT(ρ), FT(q))
                fixed = CA.FixedTerminalVelocity()
                blk = CA.Blk1MTerminalVelocity()
                chen = CA.Chen2022TerminalVelocity()
                stokes = CA.StokesTerminalVelocity()
                for ρ in (0.4, 1.2), q in (1e-6, 1e-3)
                    @test w(fixed, @name(q_lcl), ρ, q) ==
                          CAP.fixed_cloud_liquid_terminal_velocity(params)
                    @test w(fixed, @name(q_icl), ρ, q) ==
                          CAP.fixed_cloud_ice_terminal_velocity(params)
                    @test w(fixed, @name(q_rai), ρ, q) ==
                          CAP.fixed_rain_terminal_velocity(params)
                    @test w(fixed, @name(q_sno), ρ, q) ==
                          CAP.fixed_snow_terminal_velocity(params)
                    @test w(stokes, @name(q_lcl), ρ, q) ==
                          CMNe.terminal_velocity(cmc.liquid, cmc.stokes, FT(ρ), FT(q))
                    @test w(chen, @name(q_icl), ρ, q) ==
                          CMNe.terminal_velocity(
                        cmc.ice,
                        cmc.Ch2022.small_ice,
                        FT(ρ),
                        FT(q),
                    )
                    @test w(blk, @name(q_rai), ρ, q) == CM1.terminal_velocity(
                        cmp.precip.rain,
                        cmp.terminal_velocity.rain,
                        FT(ρ),
                        FT(q),
                    )
                    @test w(chen, @name(q_rai), ρ, q) ==
                          CM1.terminal_velocity(
                        cmp.precip.rain,
                        cmc.Ch2022.rain,
                        FT(ρ),
                        FT(q),
                    )
                    @test w(blk, @name(q_sno), ρ, q) == CM1.terminal_velocity(
                        cmp.precip.snow,
                        cmp.terminal_velocity.snow,
                        FT(ρ),
                        FT(q),
                    )
                    @test w(chen, @name(q_sno), ρ, q) == CM1.terminal_velocity(
                        cmp.precip.snow,
                        cmc.Ch2022.large_ice,
                        FT(ρ),
                        FT(q),
                    )
                end
            end
        end
    end
end
