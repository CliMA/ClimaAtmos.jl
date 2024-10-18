#####
##### Vertical diffusion boundary layer parameterization
#####

import StaticArrays
import ClimaCore.Geometry: ⊗
import ClimaCore.Utilities: half
import Thermodynamics as TD
import SurfaceFluxes as SF
import ClimaCore.Spaces as Spaces
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators

vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t) =
    vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, p.atmos.vert_diff)

vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function vertical_diffusion_boundary_layer_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::Union{VerticalDiffusion, FriersonDiffusion},
)
    FT = eltype(Y)
    (; ᶜu, ᶜh_tot, ᶜspecific, ᶜK_u, ᶜK_h, sfc_conditions) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    if diffuse_momentum(p.atmos.vert_diff)
        ᶠstrain_rate = p.scratch.ᶠtemp_UVWxUVW
        compute_strain_rate_face!(ᶠstrain_rate, ᶜu)
        @. Yₜ.c.uₕ -= C12(
            ᶜdivᵥ(-2 * ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_u) * ᶠstrain_rate) / Y.c.ρ,
        )

        # apply boundary condition for momentum flux
        ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ),
        )
        @. Yₜ.c.uₕ -= ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ))) / Y.c.ρ
    end

    ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0))),
        bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot),
    )
    @. Yₜ.c.ρe_tot -=
        ᶜdivᵥ_ρe_tot(-(ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜh_tot)))

    ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
    ρ_flux_χ = p.scratch.sfc_temp_C3
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        if χ_name == :q_tot
            @. ρ_flux_χ = sfc_conditions.ρ_flux_q_tot
        else
            @. ρ_flux_χ = C3(FT(0))
        end
        ᶜdivᵥ_ρχ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(ρ_flux_χ),
        )
        @. ᶜρχₜ_diffusion =
            ᶜdivᵥ_ρχ(-(ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜχ)))
        @. ᶜρχₜ -= ᶜρχₜ_diffusion
        if !(χ_name in (:q_rai, :q_sno))
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion
        end
    end
end
