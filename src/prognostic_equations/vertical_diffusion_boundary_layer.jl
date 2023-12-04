#####
##### Vertical diffusion boundary layer parameterization
#####

import StaticArrays
import ClimaCore.Geometry: ⊗
import ClimaCore.Utilities: half
import LinearAlgebra: norm
import Thermodynamics as TD
import SurfaceFluxes as SF
import ClimaCore.Spaces as Spaces
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators

function vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    Fields.bycolumn(axes(Y.c.uₕ)) do colidx
        (; vert_diff) = p.atmos
        vertical_diffusion_boundary_layer_tendency!(
            Yₜ,
            Y,
            p,
            t,
            colidx,
            vert_diff,
        )
    end
end

vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) =
    nothing

function vertical_diffusion_boundary_layer_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    ::Union{VerticalDiffusion, FriersonDiffusion},
)
    FT = eltype(Y)
    (; ᶜu, ᶜh_tot, ᶜspecific, ᶜK_u, ᶜK_h, sfc_conditions) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    if diffuse_momentum(p.atmos.vert_diff)
        ᶠstrain_rate = p.scratch.ᶠtemp_UVWxUVW
        compute_strain_rate_face!(ᶠstrain_rate[colidx], ᶜu[colidx])
        @. Yₜ.c.uₕ[colidx] -= C12(
            ᶜdivᵥ(
                -2 *
                ᶠinterp(Y.c.ρ[colidx]) *
                ᶠinterp(ᶜK_u[colidx]) *
                ᶠstrain_rate[colidx],
            ) / Y.c.ρ[colidx],
        )

        # apply boundary condition for momentum flux
        ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
        )
        @. Yₜ.c.uₕ[colidx] -=
            ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx]
    end

    ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0))),
        bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot[colidx]),
    )
    @. Yₜ.c.ρe_tot[colidx] -= ᶜdivᵥ_ρe_tot(
        -(
            ᶠinterp(Y.c.ρ[colidx]) *
            ᶠinterp(ᶜK_h[colidx]) *
            ᶠgradᵥ(ᶜh_tot[colidx])
        ),
    )

    ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
    ρ_flux_χ = p.scratch.sfc_temp_C3
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        if χ_name == :q_tot
            @. ρ_flux_χ[colidx] = sfc_conditions.ρ_flux_q_tot[colidx]
        else
            @. ρ_flux_χ[colidx] = C3(FT(0))
        end
        ᶜdivᵥ_ρχ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(ρ_flux_χ[colidx]),
        )
        @. ᶜρχₜ_diffusion[colidx] = ᶜdivᵥ_ρχ(
            -(
                ᶠinterp(Y.c.ρ[colidx]) *
                ᶠinterp(ᶜK_h[colidx]) *
                ᶠgradᵥ(ᶜχ[colidx])
            ),
        )
        @. ᶜρχₜ[colidx] -= ᶜρχₜ_diffusion[colidx]
        @. Yₜ.c.ρ[colidx] -= ᶜρχₜ_diffusion[colidx]
    end
end
