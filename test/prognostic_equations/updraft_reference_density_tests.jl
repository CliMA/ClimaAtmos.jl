# The excess of the clear-air updraft reference density over the environment
# density (`updraft_reference_density_excess`) removes the fraction
# `f_ev (1 − cf⁰)` of the environment cloud condensate and its latent heat at
# fixed pressure and total water: it vanishes in clear and overcast
# environments, is positive otherwise, and matches the linearized cooling and
# gas-constant change for a small removed amount.
using Test
import ClimaAtmos as CA
const TD = CA.TD

@testset "Updraft reference density excess" begin
    for FT in (Float32, Float64)
        thp = CA.Parameters.thermodynamics_params(CA.ClimaAtmosParameters(FT))
        T, p, q_tot = FT(285), FT(9e4), FT(0.012)
        q_lcl, q_rai, q_icl = FT(4e-4), FT(1e-4), FT(0)
        Δρ(f_ev, cf, q_lcl, q_icl; T = T) = CA.updraft_reference_density_excess(
            thp, f_ev, cf, T, p, q_tot, q_lcl + q_rai, q_icl, q_lcl, q_icl,
        )
        ρ = TD.air_density(thp, T, p, q_tot, q_lcl + q_rai, q_icl)
        # Zero without weight, in an overcast environment, and without cloud
        # condensate (rain stays in place).
        @test Δρ(FT(0), FT(0.1), q_lcl, q_icl) == 0
        @test Δρ(FT(1), FT(1), q_lcl, q_icl) == 0
        @test Δρ(FT(1), FT(0.1), FT(0), FT(0)) == 0
        # Positive otherwise, and largest for a clear environment.
        @test Δρ(FT(1), FT(0.5), q_lcl, q_icl) > 0
        @test Δρ(FT(1), FT(0), q_lcl, q_icl) > Δρ(FT(1), FT(0.5), q_lcl, q_icl)
        # Small removed amount: δρ/ρ = −δT/T − δR_m/R_m to first order.
        q_c = FT(1e-5)
        δT = -q_c * TD.latent_heat_vapor(thp, T) /
             TD.cp_m(thp, q_tot, q_c + q_rai, FT(0))
        R_m = TD.gas_constant_air(thp, q_tot, q_c + q_rai, FT(0))
        δR_m = TD.gas_constant_air(thp, q_tot, q_rai, FT(0)) - R_m
        ρ_c = TD.air_density(thp, T, p, q_tot, q_c + q_rai, FT(0))
        @test Δρ(FT(1), FT(0), q_c, FT(0)) / ρ_c ≈ -δT / T - δR_m / R_m rtol =
            FT(2e-2)
        # Cloud ice releases the latent heat of sublimation, more than the same
        # mass of liquid releases on evaporation.
        T_cold = FT(260)
        @test Δρ(FT(1), FT(0), FT(0), q_lcl; T = T_cold) >
              Δρ(FT(1), FT(0), q_lcl, FT(0); T = T_cold) > 0
        @test Δρ(FT(1), FT(0.5), q_lcl, q_icl) isa FT
        @test Δρ(FT(1), FT(0), q_lcl, q_icl) / ρ < FT(1e-2)
    end
end
