#=
Tests for the total-enthalpy diffusion of the Smagorinsky-Lilly closure: the
horizontal and vertical energy tendencies equal the sum of their
dry-static-energy, vapor, liquid, and ice constituent fluxes. See #4704.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields

include(joinpath(@__DIR__, "..", "..", "test_helpers.jl"))

@testset "Smagorinsky-Lilly energy flux constituents" begin
    config = CA.AtmosConfig(
        Dict{String, Any}(
            "initial_condition" => "DryDensityCurrentProfile",
            "FLOAT_TYPE" => "Float64",
            "config" => "box",
            "smagorinsky_lilly" => "UVW",
            "hyperdiff" => nothing,
            "x_max" => 6400.0, "x_elem" => 2, "y_max" => 6400.0, "y_elem" => 2,
            "z_max" => 3000.0, "z_elem" => 10, "z_stretch" => false,
            "dt" => "1secs", "t_end" => "10secs",
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "smagorinsky_energy_flux_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)

    # Independent variation in each enthalpy constituent and the diffusivities.
    coords = Fields.coordinate_field(Y.c)
    ᶜphase = similar(Y.c.ρ)
    @. ᶜphase = FT(2π) * (coords.x / FT(6400) + coords.z / FT(3000))
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜD_h, ᶜνₜ_v) = p.precomputed
    @. ᶜT += FT(2) * sin(ᶜphase)
    @. ᶜq_tot_nonneg = FT(2e-3) * (1 + FT(0.5) * sin(ᶜphase + FT(0.5)))
    @. ᶜq_liq = FT(2e-4) * (1 + sin(ᶜphase + FT(1)))
    @. ᶜq_ice = FT(2e-4) * (1 + sin(ᶜphase + FT(2)))
    @. ᶜD_h = FT(10) * (1 + FT(0.5) * sin(ᶜphase))
    @. ᶜνₜ_v = FT(5) * (1 + FT(0.5) * cos(ᶜphase))

    thermo_params = CA.CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core
    ᶜρ = Y.c.ρ
    ᶜq_vap = similar(Y.c.ρ)
    @. ᶜq_vap = CA.TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)

    # The horizontal ρe_tot tendency is the sum of the four constituents.
    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.horizontal_smagorinsky_lilly_tendency!(
        Yₜ, Y, p, t, p.atmos.smagorinsky_lilly,
    )
    ᶜflux = similar(Y.c.ρ)
    ᶜtotal = similar(Y.c.ρ)
    ᶜtotal .= 0
    @. ᶜflux = CA.wdivₕ(
        ᶜρ * ᶜD_h *
        CA.gradₕ(CA.TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)),
    )
    @test maximum(abs, parent(ᶜflux)) > 0
    @. ᶜtotal += ᶜflux
    @. ᶜflux = CA.wdivₕ(
        ᶜρ * ᶜD_h * (CA.TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ) *
        CA.gradₕ(ᶜq_vap),
    )
    @test maximum(abs, parent(ᶜflux)) > 0
    @. ᶜtotal += ᶜflux
    @. ᶜflux = CA.wdivₕ(
        ᶜρ * ᶜD_h * (CA.TD.enthalpy_liquid(thermo_params, ᶜT) + ᶜΦ) *
        CA.gradₕ(ᶜq_liq),
    )
    @test maximum(abs, parent(ᶜflux)) > 0
    @. ᶜtotal += ᶜflux
    @. ᶜflux = CA.wdivₕ(
        ᶜρ * ᶜD_h * (CA.TD.enthalpy_ice(thermo_params, ᶜT) + ᶜΦ) *
        CA.gradₕ(ᶜq_ice),
    )
    @test maximum(abs, parent(ᶜflux)) > 0
    @. ᶜtotal += ᶜflux
    @test parent(Yₜ.c.ρe_tot) ≈ parent(ᶜtotal) rtol = FT(1e-10)

    # The vertical ρe_tot tendency is the sum of the four constituents.
    Yₜ .= zero(eltype(Yₜ))
    CA.vertical_smagorinsky_lilly_tendency!(
        Yₜ, Y, p, t, p.atmos.smagorinsky_lilly,
    )
    Pr_t = CA.CAP.Prandtl_number_0(CA.CAP.turbconv_params(p.params))
    ᶜtotal .= 0
    @. ᶜflux =
        -CA.ᶜdiffdivᵥ(
            -(
                CA.ᶠinterp(ᶜρ) * CA.ᶠinterp(ᶜνₜ_v) / Pr_t *
                CA.ᶠgradᵥ(CA.TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ))
            ),
        )
    @test maximum(abs, parent(ᶜflux)) > 0
    @. ᶜtotal += ᶜflux
    @. ᶜflux =
        -CA.ᶜdiffdivᵥ(
            -(
                CA.ᶠinterp(ᶜρ) * CA.ᶠinterp(ᶜνₜ_v) / Pr_t *
                CA.ᶠinterp(CA.TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ) *
                CA.ᶠgradᵥ(ᶜq_vap)
            ),
        )
    @test maximum(abs, parent(ᶜflux)) > 0
    @. ᶜtotal += ᶜflux
    @. ᶜflux =
        -CA.ᶜdiffdivᵥ(
            -(
                CA.ᶠinterp(ᶜρ) * CA.ᶠinterp(ᶜνₜ_v) / Pr_t *
                CA.ᶠinterp(CA.TD.enthalpy_liquid(thermo_params, ᶜT) + ᶜΦ) *
                CA.ᶠgradᵥ(ᶜq_liq)
            ),
        )
    @test maximum(abs, parent(ᶜflux)) > 0
    @. ᶜtotal += ᶜflux
    @. ᶜflux =
        -CA.ᶜdiffdivᵥ(
            -(
                CA.ᶠinterp(ᶜρ) * CA.ᶠinterp(ᶜνₜ_v) / Pr_t *
                CA.ᶠinterp(CA.TD.enthalpy_ice(thermo_params, ᶜT) + ᶜΦ) *
                CA.ᶠgradᵥ(ᶜq_ice)
            ),
        )
    @test maximum(abs, parent(ᶜflux)) > 0
    @. ᶜtotal += ᶜflux
    @test parent(Yₜ.c.ρe_tot) ≈ parent(ᶜtotal) rtol = FT(1e-10)
end
