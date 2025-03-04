#####
##### Surface flux tendencies
#####

import ClimaCore.Geometry: ⊗
import ClimaCore.Operators as Operators

function surface_flux_tendency!(Yₜ, Y, p, t)

    p.atmos.disable_surface_flux_tendency && return

    FT = eltype(Y)
    (; ᶜh_tot, ᶜspecific, sfc_conditions) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    if !disable_momentum_vertical_diffusion(p.atmos.vert_diff)
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
    @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(-(FT(0) * ᶠgradᵥ(ᶜh_tot)))

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
        @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρχ(-(FT(0) * ᶠgradᵥ(ᶜχ)))
        @. ᶜρχₜ -= ᶜρχₜ_diffusion
        if !(χ_name in (:q_rai, :q_sno))
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion
        end
    end
end
