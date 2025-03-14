#####
##### Vertical diffusion boundary layer parameterization
#####

import ClimaCore.Geometry: ⊗
import ClimaCore.Operators as Operators

vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t) =
    vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, p.atmos.vert_diff)

vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function vertical_diffusion_boundary_layer_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::Union{VerticalDiffusion, DecayWithHeightDiffusion},
)
    FT = eltype(Y)
    α_vert_diff_tracer = CAP.α_vert_diff_tracer(p.params)
    (; ᶜu, ᶜh_tot, ᶜspecific, ᶜK_u, ᶜK_h) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    if !disable_momentum_vertical_diffusion(p.atmos.vert_diff)
        ᶠstrain_rate = p.scratch.ᶠtemp_UVWxUVW
        bc_strain_rate = compute_strain_rate_face(ᶜu)
        @. ᶠstrain_rate = bc_strain_rate
        @. Yₜ.c.uₕ -= C12(
            ᶜdivᵥ(-2 * ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_u) * ᶠstrain_rate) / Y.c.ρ,
        )
    end

    ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0))),
        bottom = Operators.SetValue(C3(FT(0))),
    )
    @. Yₜ.c.ρe_tot -=
        ᶜdivᵥ_ρe_tot(-(ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜh_tot)))

    ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
    ᶜK_h_scaled = p.scratch.ᶜtemp_scalar_2
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        if χ_name in (:q_rai, :q_sno)
            @. ᶜK_h_scaled = α_vert_diff_tracer * ᶜK_h
        else
            @. ᶜK_h_scaled = ᶜK_h
        end
        ᶜdivᵥ_ρχ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        @. ᶜρχₜ_diffusion =
            ᶜdivᵥ_ρχ(-(ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜχ)))
        @. ᶜρχₜ -= ᶜρχₜ_diffusion
        @. Yₜ.c.ρ -= ᶜρχₜ_diffusion
    end
end
