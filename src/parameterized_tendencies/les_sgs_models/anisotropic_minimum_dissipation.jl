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
tensor `τ = -2 νₜ S`. Total energy receives the divergence of the enthalpy flux
`-ρ [Dₜ(T) (c_p ∇ₕT + ∇ₕΦ) + Dₜ(q_tot) ((h_v + Φ) ∇ₕq_tot + (h_l - h_v) ∇ₕq_liq + (h_i - h_v) ∇ₕq_ice)]`,
with `c_p` the heat capacity of the diffusing mixture (`add_sensible_heat_diffusionₕ!`),
where `q_liq` and `q_ice` include rain and snow; its water terms are absent in dry
configurations. Total water diffuses with `Dₜ(q_tot)` (mirrored to `Yₜ.c.ρ` for mass
conservation), each cloud and precipitation species diffuses down its own gradient with
the same diffusivity (`add_moisture_diffusionₕ!` with `precipitation = true`), and each
passive grid-scale tracer `χ` receives `+∇ₕ·(ρ Dₜ(χ) ∇ₕχ)`. Reads
`ᶜu`, `ᶠu³`, and `ᶜT` from `p.precomputed` and uses several `p.scratch` fields.

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
    ᶜS = @. ᶜtemp_strain = symmetric_part(∇ᶜu_uvw)
    ᶠS = @. ᶠtemp_strain = symmetric_part(∇ᶠu_uvw)

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

    ## Total energy tendency: sensible heat and geopotential with D(∇T), then
    ## the water enthalpy flux and the water species, precipitation included,
    ## with D(∇q_tot)
    ᶜT = p.precomputed.ᶜT
    ᶜq_water = ᶜdiffusing_water_phases(Y, p; precipitation = true)
    ∇T = @. lazy(Geometry.project(axis_uvw, gradₕ(ᶜT)))
    ᶜD_amd = @. ᶜtemp_scalar = amd_scalar_diffusivity(c_amd, ᶜ∂̂u_uvw, Δ_h, ∇T)
    add_sensible_heat_diffusionₕ!(
        Yₜ.c.ρe_tot,
        (@. lazy(Y.c.ρ * ᶜD_amd)),
        ᶜT,
        ᶜq_water,
        p,
        ᶜtemp_scalar_3,
    )
    if !(atmos.microphysics_model isa DryModel)
        ᶜq_tot = ᶜq_water[1]
        ∇q_tot = @. lazy(Geometry.project(axis_uvw, gradₕ(ᶜq_tot)))
        @. ᶜD_amd = amd_scalar_diffusivity(c_amd, ᶜ∂̂u_uvw, Δ_h, ∇q_tot)
        add_moisture_diffusionₕ!(
            Yₜ,
            Y,
            p,
            (@. lazy(Y.c.ρ * ᶜD_amd)),
            ᶜtemp_scalar_3,
            ᶜtemp_scalar_2;
            precipitation = true,
        )
    end

    # Passive (non-microphysics) grid-scale tracers
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ρχ_name in microphysics_tracer_names(Y) && return
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        ∇ᶜχ = @. lazy(Geometry.project(axis_uvw, gradₕ(ᶜχ)))
        @. ᶜD_amd = amd_scalar_diffusivity(c_amd, ᶜ∂̂u_uvw, Δ_h, ∇ᶜχ)
        @. ᶜρχₜ += wdivₕ(Y.c.ρ * ᶜD_amd * gradₕ(ᶜχ))
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
divergence of the enthalpy flux
`-ρ [Dₜ(T) (c_p ∇ᵥT + ∇ᵥΦ) + Dₜ(q_tot) ((h_v + Φ) ∇ᵥq_tot + (h_l - h_v) ∇ᵥq_liq + (h_i - h_v) ∇ᵥq_ice)]`,
with `c_p` the heat capacity of the diffusing mixture (`add_sensible_heat_diffusionᵥ!`),
where `q_liq` and `q_ice` include rain and snow; its water terms are absent in dry
configurations. Total water diffuses with `Dₜ(q_tot)` (mirrored to `Yₜ.c.ρ` for mass
conservation), each cloud and precipitation species diffuses down its own gradient with
the same diffusivity (`add_moisture_diffusionᵥ!` with `precipitation = true`), and each
passive grid-scale tracer receives the vertical diffusive-flux divergence
`-∇ᵥ·(-ρ Dₜ(χ) ∇ᵥχ)` with face diffusivity `ᶠρ Dₜ`.
Reads `ᶜu`, `ᶠu³`, and `ᶜT` from `p.precomputed` and uses several `p.scratch` fields.

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
    ᶜS = @. ᶜtemp_strain = symmetric_part(∇ᶜu_uvw)
    ᶠS = @. ᶠtemp_strain = symmetric_part(∇ᶠu_uvw)

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

    ## Total energy tendency: sensible heat and geopotential with D(∇T), then
    ## the water enthalpy flux and the water species, precipitation included,
    ## with D(∇q_tot)
    ᶜT = p.precomputed.ᶜT
    ᶜq_water = ᶜdiffusing_water_phases(Y, p; precipitation = true)
    ∇T = @. lazy(Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜT)))
    ᶠD_amd = @. ᶠtemp_scalar = amd_scalar_diffusivity(c_amd, ᶠ∂̂u_uvw, ᶠΔ_z, ∇T)
    ᶠρD = @. lazy(ᶠρ * ᶠD_amd)
    add_sensible_heat_diffusionᵥ!(Yₜ.c.ρe_tot, ᶠρD, ᶜT, ᶜq_water, p, ᶜtemp_scalar_3)
    if !(p.atmos.microphysics_model isa DryModel)
        ᶜq_tot = ᶜq_water[1]
        ∇q_tot = @. lazy(Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜq_tot)))
        @. ᶠD_amd = amd_scalar_diffusivity(c_amd, ᶠ∂̂u_uvw, ᶠΔ_z, ∇q_tot)
        add_moisture_diffusionᵥ!(
            Yₜ,
            Y,
            p,
            ᶠρD,
            ᶜtemp_scalar_3,
            ᶜtemp_scalar_2;
            precipitation = true,
        )
    end

    ## Passive (non-microphysics) grid-scale tracers
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ρχ_name in microphysics_tracer_names(Y) && return
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        ∇ᶜχ = @. lazy(Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜχ)))
        @. ᶠD_amd = amd_scalar_diffusivity(c_amd, ᶠ∂̂u_uvw, ᶠΔ_z, ∇ᶜχ)
        ᶠρD_tracer = @. lazy(ᶠρ * ᶠD_amd)
        @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠρD_tracer * ᶠgradᵥ(ᶜχ)))
    end
end
