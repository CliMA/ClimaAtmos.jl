#####
##### Surface flux tendencies
#####

import ClimaCore.Geometry: ⊗
import ClimaCore.Operators as Operators

"""
    Boundary tendency for momentum, applied through the divergence operator
    with zeros in the interior and user specified boundary values. The
    resulting tendency is only non-zero at the boundaries.
"""
function boundary_tendency_momentum(ᶜρ, ᶜuₕ, ρ_flux_uₕ)
    FT = eltype(ᶜρ)
    ᶠgradᵥ = Operators.GradientC2F()
    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        bottom = Operators.SetValue(ρ_flux_uₕ),
    )
    return @. lazy(ᶜdivᵥ_uₕ((0 * ᶠgradᵥ(ᶜuₕ))) / ᶜρ)
end

"""
    Boundary tendency for scalars, applied through the divergence operator
    with zeros in the interior and user specified boundary values. The
    resulting tendency is only non-zero at the boundaries.
"""
function boundary_tendency_scalar(ᶜχ, ρ_flux_χ)
    FT = eltype(ᶜχ)
    ᶠgradᵥ = Operators.GradientC2F()
    ᶜdivᵥ_χ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0))),
        bottom = Operators.SetValue(ρ_flux_χ),
    )
    return @. lazy(ᶜdivᵥ_χ(0 * ᶠgradᵥ(ᶜχ)))
end

"""
   Apply surface flux tendency as an explicit boundary source.
   When `disable_surface_flux_tendency` is true, no surface flux tendency is applied
   no matter what the surface conditions are.
"""
function surface_flux_tendency!(Yₜ, Y, p, t)

    p.atmos.disable_surface_flux_tendency && return

    FT = eltype(Y)
    (; ᶜh_tot, ᶜspecific, sfc_conditions) = p.precomputed

    if !disable_momentum_vertical_diffusion(p.atmos.vert_diff)
        btt =
            boundary_tendency_momentum(Y.c.ρ, Y.c.uₕ, sfc_conditions.ρ_flux_uₕ)
        @. Yₜ.c.uₕ -= btt
    end

    btt = boundary_tendency_scalar(ᶜh_tot, sfc_conditions.ρ_flux_h_tot)
    @. Yₜ.c.ρe_tot -= btt
    ρ_flux_χ = p.scratch.sfc_temp_C3
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        if χ_name == :q_tot
            @. ρ_flux_χ = sfc_conditions.ρ_flux_q_tot
        else
            @. ρ_flux_χ = C3(FT(0))
        end
        btt = boundary_tendency_scalar(ᶜχ, ρ_flux_χ)
        @. ᶜρχₜ -= btt
        @. Yₜ.c.ρ -= btt
    end
end
