#=
Unit tests for `imex_gamma` extraction from ODE algorithms.
=#

using Test
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS

@testset "Microphysics averaging window" begin

    @testset "imex_gamma extracts the DIRK diagonal" begin
        newton = CTS.NewtonsMethod(; max_iters = 1)
        γ_ars222_expected = 1 - 1 / sqrt(2.0)
        γ_ars343_expected = 0.4358665215084590
        γ_ars233_expected = (3 + sqrt(3.0)) / 6
        for (name, γ_expected) in (
            (CTS.ARS222(), γ_ars222_expected),
            (CTS.ARS343(), γ_ars343_expected),
            (CTS.ARS233(), γ_ars233_expected),
        )
            alg = CTS.IMEXAlgorithm(name, newton)
            γ = CA.imex_gamma(alg)
            @test γ !== nothing
            @test γ ≈ γ_expected rtol = 1e-6
        end

        # Fully explicit algorithm has no implicit stages → `nothing`.
        exp_alg = CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher())
        @test CA.imex_gamma(exp_alg) === nothing

        # `nothing` (no ODE scheme) → `nothing`.
        @test CA.imex_gamma(nothing) === nothing
    end
end
