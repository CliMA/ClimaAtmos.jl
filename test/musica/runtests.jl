# Musica supports Julia 1.11 and 1.12, so its test has its own environment.
using Test
import ClimaAtmos as CA
import Musica

@testset "GasPhaseChem prints MUSICA version" begin
    @test_logs (:info, r"MUSICA version: .+") CA.chemistry_tendency!(
        nothing,
        nothing,
        nothing,
        0.0,
        CA.GasPhaseChem(),
    )
end
