#=
Hyperdiffusion: the microphysics species of the one-moment, two-moment, and P3
schemes are in the set that the generic ∇⁴ tracer tendency skips, so each takes
its share of the total-water tendency or receives nothing. A species missing
from the set would receive the full-strength ∇⁴ tendency on top of its share.

This asserts on the name set rather than on the tendency because two-moment and
P3 microphysics are disabled in `set_precomputed_quantities!`, so a state
carrying `ρn_ice`, `ρq_rim`, or `ρb_rim` cannot be built. Assert on the
tendency once that restriction lifts.
=#
using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore.MatrixFields

@testset "Microphysics species are excluded from the generic ∇⁴ tendency" begin
    state = CA.Setups.physical_state(; T = 300.0, p = 1e5)
    for model in (
        CA.NonEquilibriumMicrophysics1M(),
        CA.NonEquilibriumMicrophysics2M(),
        CA.NonEquilibriumMicrophysics2MP3(),
    )
        species = keys(CA.Setups.precip_variables(1.0, state, model))
        @test !isempty(species)
        for name in species
            @test MatrixFields.FieldName(name) in CA.hyperdiffusion_excluded_gs_names
        end
    end
    # The cloud masses are excluded as well; they take their share of the
    # total-water tendency.
    for name in (:ρq_lcl, :ρq_icl)
        @test MatrixFields.FieldName(name) in CA.hyperdiffusion_excluded_gs_names
    end
    # A passive tracer is not excluded.
    @test !(MatrixFields.FieldName(:ρq_gas_A) in CA.hyperdiffusion_excluded_gs_names)
end
