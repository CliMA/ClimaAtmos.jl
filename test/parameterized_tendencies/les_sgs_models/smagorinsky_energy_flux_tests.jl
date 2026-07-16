#=
Tests the total-enthalpy diffusion of the Smagorinsky-Lilly closure. The
tendencies diffuse the single-gradient decomposition

    F_h = -ρ D [∇s_d + (h_eff + Φ) ∇q_tot_eff]

rather than `∇h_tot` directly, matching the boundary-layer and EDMFX diffusion.
Two properties are checked in the horizontal and the vertical: the tendency is
the dry-static-energy flux plus the aggregate water flux and nothing else, and
it differs from diffusing `h_tot` by the spurious transport the decomposition
removes. Uniform `q_tot_eff` is checked as the limit in which the water flux
vanishes while the `h_tot` flux does not; that case is measured against the
flux scale of the varying case, since both are otherwise roundoff-sized. See
#4704.
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
            "initial_condition" => "Bomex",
            "FLOAT_TYPE" => "Float64",
            "config" => "box",
            "smagorinsky_lilly" => "UVW",
            "microphysics_model" => "1M",
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
    ᶜh_eff_plus_Φ = similar(ᶜρ)
    ᶜq_tot_eff = similar(ᶜρ)
    ᶜtendency = similar(ᶜρ)
    ᶜdse_flux = similar(ᶜρ)
    ᶜwater_flux = similar(ᶜρ)
    ᶜdirect = similar(ᶜρ)

    h_scale = Ref(zero(FT))
    v_scale = Ref(zero(FT))
    for uniform_q_tot in (false, true)
        # Temperature and the diffusivities vary, so the dry-static-energy
        # flux is nonzero in both directions and nothing is checked at
        # constant D.
        @. ᶜT = FT(285) + FT(2) * sin(ᶜphase)
        @. Y.c.ρq_lcl = ᶜρ * FT(2e-4) * (1 + sin(ᶜphase + FT(1)))
        @. Y.c.ρq_icl = ᶜρ * FT(2e-4) * (1 + sin(ᶜphase + FT(2)))
        @. Y.c.ρq_rai = ᶜρ * FT(1e-5) * (1 + sin(ᶜphase + FT(3)))
        @. Y.c.ρq_sno = zero(ᶜρ)
        @. ᶜD_h = FT(10) * (1 + FT(0.5) * sin(ᶜphase))
        @. ᶜνₜ_v = FT(5) * (1 + FT(0.5) * cos(ᶜphase))
        if uniform_q_tot
            # q_tot_eff = q_tot - q_rai - q_sno is what has to be uniform.
            @. Y.c.ρq_tot = ᶜρ * FT(6e-3) + Y.c.ρq_rai + Y.c.ρq_sno
        else
            @. Y.c.ρq_tot =
                ᶜρ * FT(6e-3) * (1 + FT(0.3) * sin(ᶜphase + FT(0.5))) +
                Y.c.ρq_rai + Y.c.ρq_sno
        end

        # Keep the precomputed partition consistent with the state above.
        @. ᶜq_tot_nonneg = Y.c.ρq_tot / ᶜρ
        @. ᶜq_liq = Y.c.ρq_lcl / ᶜρ
        @. ᶜq_ice = Y.c.ρq_icl / ᶜρ
        @. ᶜq_vap = CA.TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)

        # `h_tot` consistent with that state, at uniform kinetic energy.
        @. ᶜh_tot =
            (1 - ᶜq_tot_nonneg) * CA.TD.enthalpy_dry(thermo_params, ᶜT) +
            ᶜq_vap * CA.TD.enthalpy_vapor(thermo_params, ᶜT) +
            ᶜq_liq * CA.TD.enthalpy_liquid(thermo_params, ᶜT) +
            ᶜq_ice * CA.TD.enthalpy_ice(thermo_params, ᶜT) +
            ᶜΦ

        # Reference coefficients, assembled from the thermodynamics rather than
        # through the helpers the source uses.
        ᶜs_d = @. CA.TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)
        @. ᶜq_tot_eff = (Y.c.ρq_tot - Y.c.ρq_rai - Y.c.ρq_sno) / ᶜρ
        @. ᶜh_eff_plus_Φ =
            (
                CA.TD.enthalpy_vapor(thermo_params, ᶜT) * max(FT(0), ᶜq_vap) +
                CA.TD.enthalpy_liquid(thermo_params, ᶜT) *
                max(FT(0), Y.c.ρq_lcl / ᶜρ) +
                CA.TD.enthalpy_ice(thermo_params, ᶜT) *
                max(FT(0), Y.c.ρq_icl / ᶜρ)
            ) / max(
                max(FT(0), ᶜq_vap) +
                max(FT(0), Y.c.ρq_lcl / ᶜρ) +
                max(FT(0), Y.c.ρq_icl / ᶜρ),
                eps(FT),
            ) + ᶜΦ

        ## Horizontal
        Yₜ .= zero(eltype(Yₜ))
        CA.horizontal_smagorinsky_lilly_tendency!(
            Yₜ, Y, p, t, p.atmos.smagorinsky_lilly,
        )
        @. ᶜtendency = Yₜ.c.ρe_tot
        @. ᶜdse_flux = CA.wdivₕ(ᶜρ * ᶜD_h * CA.gradₕ(ᶜs_d))
        @. ᶜwater_flux =
            CA.wdivₕ(ᶜρ * ᶜD_h * ᶜh_eff_plus_Φ * CA.gradₕ(ᶜq_tot_eff))
        @. ᶜdirect = CA.wdivₕ(ᶜρ * ᶜD_h * CA.gradₕ(ᶜh_tot))

        @test maximum(abs, parent(ᶜtendency)) > 0
        @test maximum(abs, parent(ᶜdse_flux)) > 0
        # The tendency is the two fluxes and nothing else.
        @test parent(ᶜtendency) ≈ parent(ᶜdse_flux) .+ parent(ᶜwater_flux) rtol =
            FT(1e-10)
        # A uniform q_tot_eff leaves no water flux. It is not bitwise zero,
        # because the spectral gradient of a constant field is not, so it is
        # measured against the flux scale of the varying case rather than
        # against a tendency that is itself roundoff-sized.
        if uniform_q_tot
            @test maximum(abs, parent(ᶜwater_flux)) < FT(1e-10) * h_scale[]
        else
            h_scale[] = maximum(abs, parent(ᶜtendency))
            @test fraction(ᶜwater_flux, ᶜtendency) > FT(1e-6)
        end
        # Diffusing h_tot directly is a different tendency in both cases: at
        # uniform q_tot_eff the phase partition still varies, and h_tot follows
        # that variation while the aggregate form does not.
        @test maximum(abs, parent(@. ᶜdirect - ᶜtendency)) >
              FT(1e-6) * h_scale[]

        ## Vertical
        Yₜ .= zero(eltype(Yₜ))
        CA.vertical_smagorinsky_lilly_tendency!(
            Yₜ, Y, p, t, p.atmos.smagorinsky_lilly,
        )
        @. ᶜtendency = Yₜ.c.ρe_tot
        ᶠρD = @. CA.ᶠinterp(ᶜρ) * CA.ᶠinterp(ᶜνₜ_v) / Pr_t
        @. ᶜdse_flux = -CA.ᶜdiffdivᵥ(-(ᶠρD * CA.ᶠgradᵥ(ᶜs_d)))
        @. ᶜwater_flux =
            -CA.ᶜdiffdivᵥ(
                -(ᶠρD * CA.ᶠinterp(ᶜh_eff_plus_Φ) * CA.ᶠgradᵥ(ᶜq_tot_eff)),
            )
        @. ᶜdirect = -CA.ᶜdiffdivᵥ(-(ᶠρD * CA.ᶠgradᵥ(ᶜh_tot)))

        @test maximum(abs, parent(ᶜtendency)) > 0
        @test maximum(abs, parent(ᶜdse_flux)) > 0
        @test parent(ᶜtendency) ≈ parent(ᶜdse_flux) .+ parent(ᶜwater_flux) rtol =
            FT(1e-10)
        if uniform_q_tot
            @test maximum(abs, parent(ᶜwater_flux)) < FT(1e-10) * v_scale[]
        else
            v_scale[] = maximum(abs, parent(ᶜtendency))
            @test fraction(ᶜwater_flux, ᶜtendency) > FT(1e-6)
        end
        @test maximum(abs, parent(@. ᶜdirect - ᶜtendency)) >
              FT(1e-6) * v_scale[]
    end
end
