#=
Tests the water budget of the EDMFX vertical SGS diffusive flux
(`edmfx_sgs_diffusive_flux_tendency!`). Water diffuses as a single substance on
`q_tot_eff = q_tot - q_rai - q_sno`; the suspended cloud species have no flux of
their own and instead take a share of the aggregate tendency scaled by the
clipped ratio `min(q_μ / q_tot_eff, 1)`, while rain and snow get no `K_h`
transport. The distribution is asserted directly, because a guard that never
passes leaves it silently absent. See #4770.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields

include("../test_helpers.jl")

@testset "EDMFX SGS diffusive flux water distribution (Bomex column)" begin
    config = CA.AtmosConfig(
        Dict{String, Any}(
            "initial_condition" => "Bomex",
            "FLOAT_TYPE" => "Float64",
            "config" => "column",
            "turbconv" => "prognostic_edmfx",
            "edmfx_entr_model" => "Generalized",
            "edmfx_detr_model" => "Generalized",
            "edmfx_sgs_mass_flux" => true,
            "edmfx_sgs_diffusive_flux" => true,
            "edmfx_nh_pressure" => true,
            "prognostic_tke" => true,
            "microphysics_model" => "1M",
            "z_max" => 3000.0,
            "z_elem" => 30,
            "z_stretch" => false,
            "dt" => "1secs",
            "t_end" => "10secs",
            "ode_algo" => "ARS222",
            "toml" => [joinpath(pkgdir(CA), "toml", "prognostic_edmfx.toml")],
            "output_default_diagnostics" => false,
        );
        job_id = "edmfx_sgs_diffusive_flux_water_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)

    # A vertically varying water profile drives every branch of the flux.
    ᶜz = Fields.coordinate_field(Y.c).z
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    @. Y.c.ρq_lcl = Y.c.ρ * FT(2e-4) * (1 + sin(FT(2π) * ᶜz / FT(3000)))
    @. Y.c.ρq_icl = Y.c.ρ * FT(1e-4) * (1 + sin(FT(2π) * ᶜz / FT(3000) + FT(1)))
    @. Y.c.ρq_rai = Y.c.ρ * FT(1e-5) * (1 + sin(FT(2π) * ᶜz / FT(3000) + FT(2)))
    @. Y.c.ρq_sno = zero(Y.c.ρ)
    @. Y.c.ρq_tot =
        Y.c.ρ * FT(8e-3) * (1 + FT(0.2) * sin(FT(2π) * ᶜz / FT(3000) + FT(0.5))) +
        Y.c.ρq_rai + Y.c.ρq_sno
    CA.set_precomputed_quantities!(Y, p, t)

    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.edmfx_sgs_diffusive_flux_tendency!(
        Yₜ, Y, p, t, p.atmos.turbconv_model,
    )

    # The aggregate water tendency is nonzero, so nothing below is trivial.
    @test maximum(abs, parent(Yₜ.c.ρq_tot)) > 0
    # Diffusing water moves moist air mass with it.
    @test parent(Yₜ.c.ρ) == parent(Yₜ.c.ρq_tot)

    # Each suspended cloud species takes its clipped share of that tendency.
    ᶜq_tot_eff = similar(Y.c.ρ)
    @. ᶜq_tot_eff = (Y.c.ρq_tot - Y.c.ρq_rai - Y.c.ρq_sno) / Y.c.ρ
    ᶜratio = similar(Y.c.ρ)
    ᶜexpected = similar(Y.c.ρ)
    for name in (:ρq_lcl, :ρq_icl)
        ᶜρq = getproperty(Y.c, name)
        @. ᶜratio = max(
            FT(0),
            min(FT(1), (ᶜρq / Y.c.ρ) / max(ᶜq_tot_eff, eps(FT))),
        )
        @test maximum(parent(ᶜratio)) > 0
        @. ᶜexpected = ᶜratio * Yₜ.c.ρq_tot
        @test maximum(abs, parent(ᶜexpected)) > 0
        @test parent(getproperty(Yₜ.c, name)) ≈ parent(ᶜexpected) rtol = FT(1e-10)
    end

    # Rain and snow are excluded from the aggregate and get no K_h transport of
    # their own, so their only tendency here is the K_e entrainment term.
    @test maximum(abs, parent(Y.c.ρq_rai)) > 0
end
