using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Geometry, Fields, Operators

include("test_helpers.jl")

# The consolidated diffusion operators and lazy flux helpers reproduce the
# per-function forms they replace. See #4669.
@testset "Consolidated diffusion operators and flux helpers" begin
    @testset for FT in (Float32, Float64)
        (; cent_space, face_space) = get_cartesian_spaces(; FT)
        ᶜcoord = Fields.coordinate_field(cent_space)
        ᶠcoord = Fields.coordinate_field(face_space)

        # ᶜdiffdivᵥ reproduces DivergenceF2C with a zero-flux C3 boundary, for
        # both an integer-zero and a float-zero boundary value.
        ᶠflux =
            @. Geometry.Covariant3Vector(sinpi(ᶠcoord.z) * cospi(ᶠcoord.x) + FT(0.5))
        divc_int = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.Covariant3Vector(0)),
            top = Operators.SetValue(Geometry.Covariant3Vector(0)),
        )
        divc_ft = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.Covariant3Vector(FT(0))),
            top = Operators.SetValue(Geometry.Covariant3Vector(FT(0))),
        )
        ref = @. CA.ᶜdiffdivᵥ(ᶠflux)
        @test parent(ref) == parent(@. divc_int(ᶠflux))
        @test parent(ref) == parent(@. divc_ft(ᶠflux))

        # ᶠdiffdivᵥ_u₃ reproduces DivergenceC2F with a zero-divergence boundary,
        # for both integer-zero and float-zero, on a Contravariant3 vector flux
        # and on the production rank-2 momentum flux (ρ * τ, a UVW⊗UVW tensor).
        divf_int = Operators.DivergenceC2F(
            bottom = Operators.SetDivergence(0),
            top = Operators.SetDivergence(0),
        )
        divf_ft = Operators.DivergenceC2F(
            bottom = Operators.SetDivergence(FT(0)),
            top = Operators.SetDivergence(FT(0)),
        )
        ᶜvflux = @. Geometry.Contravariant3Vector(
            sinpi(ᶜcoord.z) * cospi(ᶜcoord.x) + FT(0.3),
        )
        ref_v = @. CA.ᶠdiffdivᵥ_u₃(ᶜvflux)
        @test parent(ref_v) == parent(@. divf_int(ᶜvflux))
        @test parent(ref_v) == parent(@. divf_ft(ᶜvflux))
        ᶜtflux = @. Geometry.outer(
            Geometry.UVWVector(sinpi(ᶜcoord.z), FT(0.2), cospi(ᶜcoord.x)),
            Geometry.UVWVector(FT(0.1), cospi(ᶜcoord.z), sinpi(ᶜcoord.x)),
        )
        # As in the u₃ call sites, the divergence is projected with C3 rather
        # than materialized bare, so the scalar boundary value is accepted.
        ref_t = @. CA.C3(CA.ᶠdiffdivᵥ_u₃(ᶜtflux))
        @test parent(ref_t) == parent(@. CA.C3(divf_int(ᶜtflux)))
        @test parent(ref_t) == parent(@. CA.C3(divf_ft(ᶜtflux)))

        # ᶜdiffusive_flux_divergenceᵥ pins the flux sign and composition; a sign
        # flip fails here.
        ᶠcoef = @. sinpi(ᶠcoord.z) + FT(1.5)
        ᶜχ = @. cospi(ᶜcoord.z) * sinpi(ᶜcoord.x)
        ref_div = @. CA.ᶜdiffdivᵥ(-(ᶠcoef * CA.ᶠgradᵥ(ᶜχ)))
        flux_div = similar(ref_div)
        flux_div .= CA.ᶜdiffusive_flux_divergenceᵥ(ᶠcoef, ᶜχ)
        @test parent(flux_div) == parent(ref_div)

        # ᶠtotal_enthalpy_gradientᵥ matches the hand-expanded four-term sum.
        thermo_params = CA.ClimaAtmosParameters(FT).thermodynamics_params
        ᶜT = @. FT(280) + FT(10) * sinpi(ᶜcoord.z)
        ᶜΦ = @. FT(9.8) * FT(1000) * ᶜcoord.z
        ᶜq_vap = @. FT(0.01) * (1 + FT(0.1) * cospi(ᶜcoord.z))
        ᶜq_liq = @. FT(0.001) * sinpi(ᶜcoord.x)
        ᶜq_ice = @. FT(0.0005) * cospi(ᶜcoord.x)
        ref_h = @. CA.ᶠgradᵥ(CA.TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)) +
           CA.ᶠinterp(CA.TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ) *
           CA.ᶠgradᵥ(ᶜq_vap) +
           CA.ᶠinterp(CA.TD.enthalpy_liquid(thermo_params, ᶜT) + ᶜΦ) *
           CA.ᶠgradᵥ(ᶜq_liq) +
           CA.ᶠinterp(CA.TD.enthalpy_ice(thermo_params, ᶜT) + ᶜΦ) *
           CA.ᶠgradᵥ(ᶜq_ice)
        grad_h = similar(ref_h)
        grad_h .= CA.ᶠtotal_enthalpy_gradientᵥ(
            thermo_params, ᶜT, ᶜΦ, ᶜq_vap, ᶜq_liq, ᶜq_ice,
        )
        @test parent(grad_h) == parent(ref_h)
    end
end
