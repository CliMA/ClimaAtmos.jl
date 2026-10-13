#####
##### AMD Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import ClimaCore: Geometry
import LinearAlgebra: norm_sqr

horizontal_amd_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_amd_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

"""
    amd_scalar_diffusivity(c_amd, ∂̂u_uvw, Δ, ∇χ)

Return the AMD diffusivity for a scalar at a point,

    Dₜ = max(0, -c (∂̂ₖuᵢ)(∂̂ₖχ) ∂ᵢχ / (∂ₗχ)(∂ₗχ)),

from the filter-scaled velocity gradient `∂̂u_uvw`, the filter width `Δ` of the
direction the gradient `∇χ` (a `UVW` vector) is taken along, so that
`∂̂χ = Δ ∇χ`, and the coefficient `c = c_amd`. The denominator is the squared
physical norm of the gradient.
"""
function amd_scalar_diffusivity(c_amd, ∂̂u_uvw, Δ, ∇χ)
    FT = eltype(∇χ)
    M = ∂̂u_uvw * (Δ * ∇χ) ⊗ ∇χ
    tr_M = M.components.data.:1 + M.components.data.:5 + M.components.data.:9
    return max(FT(0), -c_amd * tr_M / max(eps(FT), norm_sqr(∇χ)))
end

"""
    horizontal_amd_tendency!(Yₜ, Y, p, t, les::AnisotropicMinimumDissipation)

Add the horizontal Anisotropic Minimum Dissipation (AMD) subgrid-scale flux divergences to
`Yₜ` in place; return `nothing`.

The AMD closure of [Akbar2016](@cite) builds the eddy viscosity and diffusivity from
gradients scaled by the anisotropic filter widths, `∂̂ᵢ = Δᵢ ∂ᵢ` with `Δᵢ` the filter width
in direction `i` (horizontal node spacing and vertical layer thickness here):

```math
νₜ = \\max(0, -c \\, (∂̂ₖuᵢ)(∂̂ₖuⱼ) Sᵢⱼ / (∂ₗuₘ)(∂ₗuₘ)), \\quad
Dₜ = \\max(0, -c \\, (∂̂ₖuᵢ)(∂̂ₖχ) ∂ᵢχ / (∂ₗχ)(∂ₗχ)),
```

with `c = les.c_amd` [-] and a separate `Dₜ` for each diffused scalar
(`amd_scalar_diffusivity`). Momentum receives `-∇ₕ·(ρ τ)/ρ` with the SGS momentum flux
tensor `τ = -2 νₜ S`. Total energy receives the divergence of the split enthalpy flux
`-ρ [Dₜ(s_d) ∇ₕs_d + Dₜ(q_tot_eff) (h_eff + Φ) ∇ₕq_tot_eff]` (see `ᶜh_eff_plus_Φ!`), each
part with the diffusivity of the scalar whose gradient it acts on and the water part absent
in dry configurations. Each grid-scale tracer `χ` receives `+∇ₕ·(ρ Dₜ ∇ₕχ)`, and the
`ρq_tot` diffusion is also added to `Yₜ.c.ρ` so that moisture diffusion conserves mass.
Reads `ᶜu`, `ᶠu³`, and `ᶜT` from `p.precomputed` and uses several `p.scratch` fields.

This tendency is always applied explicitly, from `remaining_tendency!`. To remove it in
debugging workflows, comment out the call in `remaining_tendency.jl`. See also
`vertical_amd_tendency!`.
"""
function horizontal_amd_tendency!(Yₜ, Y, p, t, les::AnisotropicMinimumDissipation)
    (; atmos, precomputed, scratch) = p
    FT = eltype(Y)
    c_amd = les.c_amd
    (; ᶜu, ᶠu³) = precomputed
    (; ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW, ᶜtemp_strain, ᶠtemp_strain) = scratch
    (; ᶜtemp_scalar, ᶜtemp_scalar_2, ᶜtemp_scalar_3) = scratch
    (; ᶠtemp_scalar, ᶠtemp_scalar_2) = scratch
    (; ᶜtemp_UVW, ᶠtemp_UVW) = scratch

    ∇ᵥuvw_boundary = Geometry.outer(Geometry.WVector(0), UVW(0, 0, 0))
    ᶠgradᵥ_uvw = Operators.GradientC2F(
        bottom = Operators.SetGradient(∇ᵥuvw_boundary),
        top = Operators.SetGradient(∇ᵥuvw_boundary),
    )
    axis_uvw = (Geometry.UVWAxis(),)

    # Compute UVW velocities
    ᶜu_uvw = @. ᶜtemp_UVW = UVW(ᶜu)
    ᶠu_uvw = @. ᶠtemp_UVW = UVW(ᶠinterp(Y.c.uₕ)) + UVW(ᶠu³)

    # filter scales
    h_space = Spaces.horizontal_space(axes(Y.c))
    Δ_h = Spaces.node_horizontal_length_scale(h_space)
    ᶜΔ_z = Fields.Δz_field(Y.c)

    # Gradients
    ## cell centers
    ∇ᶜu_uvw = @. ᶜtemp_UVWxUVW = Geometry.project(axis_uvw, ᶜgradᵥ(ᶠu_uvw))  # vertical component
    @. ∇ᶜu_uvw += Geometry.project(axis_uvw, gradₕ(ᶜu_uvw))  # horizontal component
    ## cell faces
    ∇ᶠu_uvw = @. ᶠtemp_UVWxUVW = Geometry.project(axis_uvw, ᶠgradᵥ_uvw(ᶜu_uvw))  # vertical component
    @. ∇ᶠu_uvw += Geometry.project(axis_uvw, gradₕ(ᶠu_uvw))  # horizontal component

    # Strain rate tensor
    ᶜS = @. ᶜtemp_strain = (∇ᶜu_uvw + adjoint(∇ᶜu_uvw)) / 2
    ᶠS = @. ᶠtemp_strain = (∇ᶠu_uvw + adjoint(∇ᶠu_uvw)) / 2

    # The viscosity denominator (∂ₗuₘ)(∂ₗuₘ), evaluated here because
    # `∇ᶜu_uvw` and the scaled derivative below share `ᶜtemp_UVWxUVW`.
    ᶜ∂ₗuₘ∂ₗuₘ = @. ᶜtemp_scalar_3 = norm_sqr(∇ᶜu_uvw)

    # Scaled Derivatives ∂̂ᵢ = Δ₍ᵢ₎∂ᵢ
    ᶜ∂̂u_uvw = @.ᶜtemp_UVWxUVW = Δ_h * Geometry.project(axis_uvw, gradₕ(ᶜu_uvw))
    @. ᶜ∂̂u_uvw += ᶜΔ_z * Geometry.project(axis_uvw, ᶜgradᵥ(ᶠu_uvw))

    ᶜ∂ₖuᵢ∂ₖuⱼ = @. lazy(ᶜ∂̂u_uvw * adjoint(ᶜ∂̂u_uvw))

    # AMD eddy viscosity
    ᶜνₜ = @. ᶜtemp_scalar = max(
        FT(0),
        -c_amd *
        (
            (ᶜ∂ₖuᵢ∂ₖuⱼ * ᶜS).components.data.:1 +
            (ᶜ∂ₖuᵢ∂ₖuⱼ * ᶜS).components.data.:5 +
            (ᶜ∂ₖuᵢ∂ₖuⱼ * ᶜS).components.data.:9
        ) / max.(eps(FT), ᶜ∂ₗuₘ∂ₗuₘ),
    )
    ᶠνₜ = @. ᶠtemp_scalar = ᶠinterp(ᶜνₜ)

    # Subgrid-scale momentum flux tensor, `τ = -2 νₜ ∘ S`
    ᶜτ_amd = @. lazy(-2 * ᶜνₜ * ᶜS)
    ᶠτ_amd = @. lazy(-2 * ᶠνₜ * ᶠS)

    ## Momentum tendencies
    ᶠρ = @. ᶠtemp_scalar_2 = ᶠinterp(Y.c.ρ)
    @. Yₜ.c.uₕ -= C12(wdivₕ(Y.c.ρ * ᶜτ_amd) / Y.c.ρ)
    @. Yₜ.f.u₃ -= C3(wdivₕ(ᶠρ * ᶠτ_amd) / ᶠρ)

    ## Total energy tendency: the dry-static-energy part of the enthalpy flux,
    ## then the enthalpy carried by the diffusing water, each with the AMD
    ## diffusivity of its own gradient
    ᶜs_d = ᶜdry_static_energy(p)
    ∇s_d = @. lazy(Geometry.project(axis_uvw, gradₕ(ᶜs_d)))
    ᶜD_amd = @. ᶜtemp_scalar = amd_scalar_diffusivity(c_amd, ᶜ∂̂u_uvw, Δ_h, ∇s_d)
    @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD_amd * gradₕ(ᶜs_d))
    if !(atmos.microphysics_model isa DryModel)
        ᶜh_eff_plus_Φ = ᶜh_eff_plus_Φ!(ᶜtemp_scalar_2, Y, p)
        ᶜq_tot_eff = ᶜdiffusing_water(Y, p)
        ∇q_tot_eff = @. lazy(Geometry.project(axis_uvw, gradₕ(ᶜq_tot_eff)))
        @. ᶜD_amd = amd_scalar_diffusivity(c_amd, ᶜ∂̂u_uvw, Δ_h, ∇q_tot_eff)
        @. Yₜ.c.ρe_tot +=
            wdivₕ(Y.c.ρ * ᶜD_amd * ᶜh_eff_plus_Φ * gradₕ(ᶜq_tot_eff))
    end

    # Tracer diffusion and associated mass changes
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        ∇ᶜχ = @. lazy(Geometry.project(axis_uvw, gradₕ(ᶜχ)))
        @. ᶜD_amd = amd_scalar_diffusivity(c_amd, ᶜ∂̂u_uvw, Δ_h, ∇ᶜχ)
        ᶜρχₜ_diffusion = @. lazy(wdivₕ(Y.c.ρ * ᶜD_amd * gradₕ(ᶜχ)))
        @. ᶜρχₜ += ᶜρχₜ_diffusion
        # Rain and snow does not affect the mass
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ += ᶜρχₜ_diffusion
        end
    end
end

import UnrolledUtilities as UU


"""
    vertical_amd_tendency!(Yₜ, Y, p, t, les::AnisotropicMinimumDissipation)

Add the vertical Anisotropic Minimum Dissipation (AMD) subgrid-scale flux divergences to
`Yₜ` in place; return `nothing`.

Computes the same [Akbar2016](@cite) eddy viscosity `νₜ` and scalar diffusivities `Dₜ` as
`horizontal_amd_tendency!` (see that docstring for the formulas), with the scalar
diffusivities evaluated on cell faces from vertical gradients. Momentum receives
`-∇ᵥ·(ρ τ)/ρ` with the SGS momentum flux tensor `τ = -2 νₜ S`. Total energy receives the
divergence of the split enthalpy flux
`-ρ [Dₜ(s_d) ∇ᵥs_d + Dₜ(q_tot_eff) (h_eff + Φ) ∇ᵥq_tot_eff]`, whose water part is absent
in dry configurations, and each grid-scale tracer
receives the vertical diffusive-flux divergence `ᶜdiffusive_flux_divergenceᵥ` with face
diffusivity `ᶠρ Dₜ` (subtracted, since it is a flux divergence); the `ρq_tot` diffusion is
also applied to `Yₜ.c.ρ` so that moisture diffusion conserves mass. Reads `ᶜu`, `ᶠu³`, and
`ᶜT` from `p.precomputed` and uses several `p.scratch` fields.

This tendency is always applied explicitly, from `remaining_tendency!`; it is not part of
the implicit solver. To remove it in debugging workflows, comment out the call in
`remaining_tendency.jl`.
"""
function vertical_amd_tendency!(Yₜ, Y, p, t, les::AnisotropicMinimumDissipation)
    FT = eltype(Y)

    c_amd = les.c_amd

    ### AMD ###

    (; ᶜu, ᶠu³) = p.precomputed
    (; ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW, ᶜtemp_strain, ᶠtemp_strain) = p.scratch
    (; ᶜtemp_scalar, ᶜtemp_scalar_2, ᶜtemp_scalar_3) = p.scratch
    (; ᶠtemp_scalar, ᶜtemp_UVW, ᶠtemp_UVW) = p.scratch

    ∇ᵥuvw_boundary = Geometry.outer(Geometry.WVector(0), UVW(0, 0, 0))
    ᶠgradᵥ_uvw = Operators.GradientC2F(
        bottom = Operators.SetGradient(∇ᵥuvw_boundary),
        top = Operators.SetGradient(∇ᵥuvw_boundary),
    )
    ᶠgradᵥ_scalar = Operators.GradientC2F(
        bottom = Operators.SetGradient(UVW(0, 0, 0)),
        top = Operators.SetGradient(UVW(0, 0, 0)),
    )

    axis_uvw = (Geometry.UVWAxis(),)

    # Compute UVW velocities
    ᶜu_uvw = @. ᶜtemp_UVW = UVW(ᶜu)
    ᶠu_uvw = @. ᶠtemp_UVW = UVW(ᶠinterp(Y.c.uₕ)) + UVW(ᶠu³)

    # filter scales
    h_space = Spaces.horizontal_space(axes(Y.c))
    Δ_h = Spaces.node_horizontal_length_scale(h_space)
    ᶜΔ_z = Fields.Δz_field(Y.c)
    ᶠΔ_z = Fields.Δz_field(Y.f)

    # Gradients
    ## cell centers
    ∇ᶜu_uvw = @. ᶜtemp_UVWxUVW = Geometry.project(axis_uvw, ᶜgradᵥ(ᶠu_uvw))  # vertical component
    @. ∇ᶜu_uvw += Geometry.project(axis_uvw, gradₕ(ᶜu_uvw))  # horizontal component
    ## cell faces
    ∇ᶠu_uvw = @. ᶠtemp_UVWxUVW = Geometry.project(axis_uvw, ᶠgradᵥ_uvw(ᶜu_uvw))  # vertical component
    @. ∇ᶠu_uvw += Geometry.project(axis_uvw, gradₕ(ᶠu_uvw))  # horizontal component

    # Strain rate tensor
    ᶜS = @. ᶜtemp_strain = (∇ᶜu_uvw + adjoint(∇ᶜu_uvw)) / 2
    ᶠS = @. ᶠtemp_strain = (∇ᶠu_uvw + adjoint(∇ᶠu_uvw)) / 2

    # The viscosity denominator (∂ₗuₘ)(∂ₗuₘ), evaluated here because
    # `∇ᶜu_uvw` and the scaled derivative below share `ᶜtemp_UVWxUVW`.
    ᶜ∂ₗuₘ∂ₗuₘ = @. ᶜtemp_scalar_3 = norm_sqr(∇ᶜu_uvw)

    # Scaled Derivatives ∂̂ᵢ = Δ₍ᵢ₎∂ᵢ
    ᶜ∂̂u_uvw = @.ᶜtemp_UVWxUVW = Δ_h * Geometry.project(axis_uvw, gradₕ(ᶜu_uvw))
    @. ᶜ∂̂u_uvw += ᶜΔ_z * Geometry.project(axis_uvw, ᶜgradᵥ(ᶠu_uvw))

    ᶠ∂̂u_uvw = @.ᶠtemp_UVWxUVW = Δ_h * Geometry.project(axis_uvw, gradₕ(ᶠu_uvw))
    @. ᶠ∂̂u_uvw += ᶠΔ_z * Geometry.project(axis_uvw, ᶠgradᵥ_uvw(ᶜu_uvw))

    ᶜ∂ₖuᵢ∂ₖuⱼ = @. lazy(ᶜ∂̂u_uvw * adjoint(ᶜ∂̂u_uvw))

    # AMD eddy viscosity
    ᶜνₜ = @. ᶜtemp_scalar = max(
        FT(0),
        -c_amd *
        (
            (ᶜ∂ₖuᵢ∂ₖuⱼ * ᶜS).components.data.:1 +
            (ᶜ∂ₖuᵢ∂ₖuⱼ * ᶜS).components.data.:5 +
            (ᶜ∂ₖuᵢ∂ₖuⱼ * ᶜS).components.data.:9
        ) / max.(eps(FT), ᶜ∂ₗuₘ∂ₗuₘ),
    )
    ᶠνₜ = @. ᶠtemp_scalar = ᶠinterp(ᶜνₜ)

    # Subgrid-scale momentum flux tensor, `τ = -2 νₜ ∘ S`
    ᶜτ_amd = @. lazy(-2 * ᶜνₜ * ᶜS)
    ᶠτ_amd = @. lazy(-2 * ᶠνₜ * ᶠS)

    # Apply to tendencies
    ## Horizontal momentum tendency
    ᶠρ = @. lazy(ᶠinterp(Y.c.ρ))
    @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(ᶠρ * ᶠτ_amd) / Y.c.ρ)
    ## Vertical momentum tendency
    @. Yₜ.f.u₃ -= C3(ᶠdiffdivᵥ_u₃(Y.c.ρ * ᶜτ_amd) / ᶠρ)

    ## Total energy tendency: the dry-static-energy part of the enthalpy flux,
    ## then the enthalpy carried by the diffusing water, each with the AMD
    ## diffusivity of its own gradient
    ᶜs_d = ᶜdry_static_energy(p)
    ∇s_d = @. lazy(Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜs_d)))
    ᶠD_amd = @. ᶠtemp_scalar = amd_scalar_diffusivity(c_amd, ᶠ∂̂u_uvw, ᶠΔ_z, ∇s_d)
    ᶠρD = @. lazy(ᶠρ * ᶠD_amd)
    ᶜ∇ᵥρD∇s_dₜ = ᶜdiffusive_flux_divergenceᵥ(ᶠρD, ᶜs_d)
    @. Yₜ.c.ρe_tot -= ᶜ∇ᵥρD∇s_dₜ
    if !(p.atmos.microphysics_model isa DryModel)
        ᶜh_eff_plus_Φ = ᶜh_eff_plus_Φ!(ᶜtemp_scalar_2, Y, p)
        ᶜq_tot_eff = ᶜdiffusing_water(Y, p)
        ∇q_tot_eff = @. lazy(Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜq_tot_eff)))
        @. ᶠD_amd = amd_scalar_diffusivity(c_amd, ᶠ∂̂u_uvw, ᶠΔ_z, ∇q_tot_eff)
        ᶠρD_h_eff = @. lazy(ᶠρ * ᶠD_amd * ᶠinterp(ᶜh_eff_plus_Φ))
        ᶜ∇ᵥρDh_eff∇q_totₜ = ᶜdiffusive_flux_divergenceᵥ(ᶠρD_h_eff, ᶜq_tot_eff)
        @. Yₜ.c.ρe_tot -= ᶜ∇ᵥρDh_eff∇q_totₜ
    end

    ## Tracer diffusion and associated mass changes
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        ∇ᶜχ = @. lazy(Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜχ)))
        @. ᶠD_amd = amd_scalar_diffusivity(c_amd, ᶠ∂̂u_uvw, ᶠΔ_z, ∇ᶜχ)
        ᶠρD_tracer = @. lazy(ᶠρ * ᶠD_amd)
        ᶜ∇ᵥρD∇χₜ = ᶜdiffusive_flux_divergenceᵥ(ᶠρD_tracer, ᶜχ)
        @. ᶜρχₜ -= ᶜ∇ᵥρD∇χₜ
        # Rain and snow does not affect the mass
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ -= ᶜ∇ᵥρD∇χₜ
        end
    end
end
