#=
Regression test for `correct_implicit_advection_tendency!` (the post-Newton
upwind correction to the central-differenced implicit advection of `ρe_tot`
and `ρq_tot`).

Identity: on the same state,
    corrected + central == full_upwind
i.e. `vertical_transport(:none) + (vtt_upwind − vtt_central) == vtt_upwind`,
which is trivially true algebraically but this test guards against future
divergence between the correction function and the implicit advection
function.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

include("../test_helpers.jl")

@testset "correct_implicit_advection_tendency! identity" begin
    # Small moist column so `ρq_tot` is present.
    config = CA.AtmosConfig(
        Dict(
            "config" => "column",
            "initial_condition" => "PrecipitatingColumn",
            "microphysics_model" => "1M",
            "energy_q_tot_upwinding" => "first_order",
            "output_default_diagnostics" => false,
        );
        job_id = "correct_implicit_advection_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    (; dt) = p
    (; ᶠu³, ᶜh_tot) = p.precomputed
    upwinding = p.atmos.numerics.energy_q_tot_upwinding

    # Perturb `ᶠu³` so both the central and upwind fluxes are nontrivial (the
    # initial condition has zero vertical velocity, which would make both
    # sides of the identity zero).
    parent(ᶠu³.components.data.:1) .+= randn(size(parent(ᶠu³.components.data.:1)))

    # Corrected tendency: (upwind − central).
    Yₜ_corr = similar(Y)
    CA.correct_implicit_advection_tendency!(Yₜ_corr, Y, p, t)

    # Central and full-upwind implicit vertical fluxes, materialized directly
    # from `vertical_transport`.
    central_ρe = similar(Y.c.ρe_tot)
    upwind_ρe = similar(Y.c.ρe_tot)
    vtt_c = CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    vtt_up = CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, upwinding)
    @. central_ρe = vtt_c
    @. upwind_ρe = vtt_up

    @test parent(Yₜ_corr.c.ρe_tot) .+ parent(central_ρe) ≈ parent(upwind_ρe)

    if !(p.atmos.microphysics_model isa CA.DryModel)
        ᶜq_tot = @. CA.specific(Y.c.ρq_tot, Y.c.ρ)
        central_ρq = similar(Y.c.ρq_tot)
        upwind_ρq = similar(Y.c.ρq_tot)
        vtt_c = CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜq_tot, dt, Val(:none))
        vtt_up = CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜq_tot, dt, upwinding)
        @. central_ρq = vtt_c
        @. upwind_ρq = vtt_up
        @test parent(Yₜ_corr.c.ρq_tot) .+ parent(central_ρq) ≈ parent(upwind_ρq)
    end
end
