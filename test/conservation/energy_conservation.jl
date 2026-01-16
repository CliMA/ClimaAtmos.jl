#=
Energy conservation tests for ClimaAtmos.jl

TODO: Implement tests for:
- Total energy conservation (kinetic + internal + potential)
- Enthalpy budget in moist simulations
- Energy conservation under different turbulence closures
- Radiative energy balance (TOA and surface)
- Energy conservation in adiabatic limit

Reference: In the absence of external forcing (radiation, surface fluxes),
total energy should be conserved to machine precision.
=#

using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

@testset "Energy conservation" begin
    @testset "Placeholder - total energy" begin
        @test_skip "Total energy conservation tests not yet implemented"
    end

    @testset "Placeholder - adiabatic limit" begin
        @test_skip "Adiabatic energy conservation tests not yet implemented"
    end

    @testset "Placeholder - radiative energy balance" begin
        @test_skip "Radiative energy balance tests not yet implemented"
    end
end
