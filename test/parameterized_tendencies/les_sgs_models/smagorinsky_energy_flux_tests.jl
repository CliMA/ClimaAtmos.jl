#=
Tests the total-enthalpy diffusion of the Smagorinsky-Lilly closure. The
tendencies diffuse the dry-static-energy + water-enthalpy decomposition rather
than `h_tot` directly. At uniform temperature and uniform kinetic energy the two
formulations differ by exactly the flux carried by the dry-air mass gradient,

    ∇h_tot - (∇s_d + Σ_μ (h_μ + Φ) ∇q_μ) = -s_d ∇q_tot,   μ ∈ {vap, liq, ice}

so they agree when `q_tot` is uniform and differ by that one term when it is
not. Both cases are checked in the horizontal and the vertical. See #4704.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields

include(joinpath(@__DIR__, "..", "..", "test_helpers.jl"))

@testset "Smagorinsky-Lilly total-enthalpy flux" begin
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

    thermo_params = CA.CAP.thermodynamics_params(p.params)
    Pr_t = CA.CAP.Prandtl_number_0(CA.CAP.turbconv_params(p.params))
    (; ᶜΦ) = p.core
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜD_h, ᶜνₜ_v, ᶜh_tot) = p.precomputed
    ᶜρ = Y.c.ρ

    ᶜphase = similar(ᶜρ)
    coords = Fields.coordinate_field(Y.c)
    @. ᶜphase = FT(2π) * (coords.x / FT(6400) + coords.z / FT(3000))

    # Size of one term relative to the tendency it belongs to.
    fraction(ᶜterm, ᶜtendency) =
        maximum(abs, parent(ᶜterm)) / maximum(abs, parent(ᶜtendency))

    Yₜ = similar(Y)
    ᶜq_vap = similar(ᶜρ)
    ᶜdecomposed = similar(ᶜρ)
    ᶜdirect = similar(ᶜρ)
    ᶜdry_mass = similar(ᶜρ)

    for uniform_q_tot in (true, false)
        # A uniform temperature makes every constituent enthalpy spatially
        # constant, which isolates the dry-air mass term. The diffusivities vary
        # so the identity is not checked at constant diffusivity.
        @. ᶜT = FT(285)
        @. ᶜq_liq = FT(2e-4) * (1 + sin(ᶜphase + FT(1)))
        @. ᶜq_ice = FT(2e-4) * (1 + sin(ᶜphase + FT(2)))
        @. ᶜD_h = FT(10) * (1 + FT(0.5) * sin(ᶜphase))
        @. ᶜνₜ_v = FT(5) * (1 + FT(0.5) * cos(ᶜphase))
        if uniform_q_tot
            @. ᶜq_tot_nonneg = FT(6e-3)
        else
            @. ᶜq_tot_nonneg = FT(6e-3) * (1 + FT(0.3) * sin(ᶜphase + FT(0.5)))
        end
        @. ᶜq_vap = CA.TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)

        # `h_tot` consistent with the state above, at uniform kinetic energy.
        @. ᶜh_tot =
            (1 - ᶜq_tot_nonneg) * CA.TD.enthalpy_dry(thermo_params, ᶜT) +
            ᶜq_vap * CA.TD.enthalpy_vapor(thermo_params, ᶜT) +
            ᶜq_liq * CA.TD.enthalpy_liquid(thermo_params, ᶜT) +
            ᶜq_ice * CA.TD.enthalpy_ice(thermo_params, ᶜT) +
            ᶜΦ

        ᶜs_d = @. CA.TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)

        ## Horizontal
        Yₜ .= zero(eltype(Yₜ))
        CA.horizontal_smagorinsky_lilly_tendency!(
            Yₜ, Y, p, t, p.atmos.smagorinsky_lilly,
        )
        @. ᶜdecomposed = Yₜ.c.ρe_tot
        @. ᶜdirect = CA.wdivₕ(ᶜρ * ᶜD_h * CA.gradₕ(ᶜh_tot))
        @. ᶜdry_mass = -CA.wdivₕ(ᶜρ * ᶜD_h * ᶜs_d * CA.gradₕ(ᶜq_tot_nonneg))

        @test maximum(abs, parent(ᶜdecomposed)) > 0
        # A uniform `q_tot` leaves no dry-mass term. It is not bitwise zero,
        # because the spectral gradient of a constant field is not.
        rel_dry_mass = fraction(ᶜdry_mass, ᶜdecomposed)
        @test uniform_q_tot ? rel_dry_mass < FT(1e-12) :
              rel_dry_mass > FT(1e-6)
        @test parent(ᶜdirect) ≈ parent(ᶜdecomposed) .+ parent(ᶜdry_mass) rtol =
            FT(1e-10)

        ## Vertical
        Yₜ .= zero(eltype(Yₜ))
        CA.vertical_smagorinsky_lilly_tendency!(
            Yₜ, Y, p, t, p.atmos.smagorinsky_lilly,
        )
        @. ᶜdecomposed = Yₜ.c.ρe_tot
        ᶠρD = @. CA.ᶠinterp(ᶜρ) * CA.ᶠinterp(ᶜνₜ_v) / Pr_t
        @. ᶜdirect = -CA.ᶜdiffdivᵥ(-(ᶠρD * CA.ᶠgradᵥ(ᶜh_tot)))
        @. ᶜdry_mass = -CA.ᶜdiffdivᵥ(
            ᶠρD * CA.ᶠinterp(ᶜs_d) * CA.ᶠgradᵥ(ᶜq_tot_nonneg),
        )

        @test maximum(abs, parent(ᶜdecomposed)) > 0
        rel_dry_mass = fraction(ᶜdry_mass, ᶜdecomposed)
        @test uniform_q_tot ? rel_dry_mass < FT(1e-12) :
              rel_dry_mass > FT(1e-6)
        @test parent(ᶜdirect) ≈ parent(ᶜdecomposed) .+ parent(ᶜdry_mass) rtol =
            FT(1e-10)
    end
end
