#####
##### AMD Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import ClimaCore: Geometry, Spaces
import LinearAlgebra: tr

"""
    set_amd_precomputed_quantities!(Y, p)

Compute and cache the common strain rate calculations for the Anisotropic-Minimum-Dissipation
method. This includes:
- Scaled velocity gradients (ᶜ∂̂u_uvw, ᶠ∂̂u_uvw)
- Strain rate tensors (ᶜS, ᶠS)
- Gradient products (ᶜ∂ₖuᵢ∂ₖuⱼ, ᶠ∂ₖuᵢ∂ₖuⱼ, ᶜ∂ₗuₘ∂ₗuₘ)
- Filter scales (ᶜδ², ᶠδ²)
- Eddy viscosity (ᶜνₜ, ᶠνₜ)

These quantities are used by both horizontal_amd_tendency! and vertical_amd_tendency!.
"""
function set_amd_precomputed_quantities!(Y, p, les::AnisotropicMinimumDissipation)
    (; precomputed, scratch, params) = p
    FT = eltype(Y)
    c_amd = les.c_amd
    (; ᶜu, ᶠu³) = precomputed
    (; ᶜtemp_UVW, ᶠtemp_UVW) = scratch
    (;
        ᶜ∂̂u_uvw,
        ᶠ∂̂u_uvw,
        ᶜS,
        ᶠS,
        ᶜ∂ₖuᵢ∂ₖuⱼ,
        ᶠ∂ₖuᵢ∂ₖuⱼ,
        ᶜ∂ₗuₘ∂ₗuₘ,
        ᶜδ²,
        ᶠδ²,
        ᶜνₜ,
        ᶠνₜ,
    ) = precomputed

    ∇ᵥuvw_boundary = Geometry.outer(Geometry.WVector(0), UVW(0, 0, 0))
    ᶠgradᵥ_uvw = Operators.GradientC2F(
        bottom = Operators.SetGradient(∇ᵥuvw_boundary),
        top = Operators.SetGradient(∇ᵥuvw_boundary),
    )
    axis_uvw = (Geometry.UVWAxis(),)

    # Compute UVW velocities
    ᶜu_uvw = @. ᶜtemp_UVW = UVW(ᶜu)
    ᶠu_uvw = @. ᶠtemp_UVW = UVW(ᶠinterp(Y.c.uₕ)) + UVW(ᶠu³)

    # Extract UV and W components separately
    ᶜu_uv = @. Geometry.UVWVector(Geometry.UVVector(ᶜu_uvw)) +
            Geometry.UVWVector(Geometry.WVector(zero(ᶜu_uvw)))
    ᶜw = @. Geometry.UVWVector(Geometry.WVector(ᶜu_uvw))
    ᶠu_uv = @. Geometry.UVWVector(Geometry.UVVector(ᶠu_uvw)) +
            Geometry.UVWVector(Geometry.WVector(zero(ᶠu_uvw)))
    ᶠw = @. Geometry.UVWVector(Geometry.WVector(ᶠu_uvw))

    # filter scales
    h_space = Spaces.horizontal_space(axes(Y.c))
    Δ_h = Spaces.node_horizontal_length_scale(h_space)
    ᶜΔ_z = Fields.Δz_field(Y.c)
    ᶠΔ_z = Fields.Δz_field(Y.f)

    # Scaled gradients
    @. ᶜ∂̂u_uvw = Geometry.project(axis_uvw, gradₕ(ᶜu_uv))
    @. ᶜ∂̂u_uvw += ᶜΔ_z / Δ_h * Geometry.project(axis_uvw, gradₕ(ᶜw))
    @. ᶜ∂̂u_uvw += Δ_h / ᶜΔ_z * Geometry.project(axis_uvw, ᶜgradᵥ(ᶠu_uv))
    @. ᶜ∂̂u_uvw += Geometry.project(axis_uvw, ᶜgradᵥ(ᶠw))

    @. ᶠ∂̂u_uvw = Geometry.project(axis_uvw, gradₕ(ᶠu_uv))
    @. ᶠ∂̂u_uvw += ᶠΔ_z / Δ_h * Geometry.project(axis_uvw, gradₕ(ᶠw))
    @. ᶠ∂̂u_uvw += Δ_h / ᶠΔ_z * Geometry.project(axis_uvw, ᶠgradᵥ_uvw(ᶜu_uv))
    @. ᶠ∂̂u_uvw += Geometry.project(axis_uvw, ᶠgradᵥ_uvw(ᶜw))

    # Strain rate tensor
    @. ᶜS = (ᶜ∂̂u_uvw + adjoint(ᶜ∂̂u_uvw)) / 2
    @. ᶠS = (ᶠ∂̂u_uvw + adjoint(ᶠ∂̂u_uvw)) / 2

    # Gradient products
    @. ᶜ∂ₖuᵢ∂ₖuⱼ = ᶜ∂̂u_uvw * adjoint(ᶜ∂̂u_uvw)
    @. ᶠ∂ₖuᵢ∂ₖuⱼ = ᶠ∂̂u_uvw * adjoint(ᶠ∂̂u_uvw)
    @. ᶜ∂ₗuₘ∂ₗuₘ = CA.norm_sqr(ᶜ∂̂u_uvw)

    # Filter scales squared
    @. ᶜδ² = 3 * (Δ_h^2 * ᶜΔ_z^2) / (2 * ᶜΔ_z^2 + Δ_h^2)
    @. ᶠδ² = 3 * (Δ_h^2 * ᶠΔ_z^2) / (2 * ᶠΔ_z^2 + Δ_h^2)

    # AMD eddy viscosity
    @. ᶜνₜ = max(
        FT(0),
        -c_amd^2 * ᶜδ² *
        (
            (ᶜ∂ₖuᵢ∂ₖuⱼ * ᶜS).components.data.:1 +
            (ᶜ∂ₖuᵢ∂ₖuⱼ * ᶜS).components.data.:5 +
            (ᶜ∂ₖuᵢ∂ₖuⱼ * ᶜS).components.data.:9
        ) / max.(eps(FT), ᶜ∂ₗuₘ∂ₗuₘ),
    )
    @. ᶠνₜ = ᶠinterp(ᶜνₜ)

    return nothing
end
set_amd_precomputed_quantities!(Y, p, ::Nothing) = nothing

horizontal_amd_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_amd_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

"""
    horizontal_amd_tendency!(Yₜ,Y, p, t, ::AnisotropicMinimumDissipation)

Anisotropic Minimum Dissipation (AMD) Subgrid-Scale Model

This module implements the Anisotropic Minimum Dissipation (AMD) subgrid-scale (SGS) model
for Large-Eddy Simulation (LES) as described by Abkar et al. (2016). The AMD model provides
the minimal eddy dissipation necessary to dissipate the energy of sub-filter scales, ensuring
numerical stability and physical accuracy in LES.

### Mathematical Formulation

The AMD model computes anisotropic eddy viscosity νₜ and eddy diffusivity Dₜ using scaled
gradient operators that account for anisotropic filter scales:

**Scaled Gradient Operator:**
```
∂̂ᵢ = Δᵢ ∂ᵢ
```
where Δᵢ is the filter width in the i-th direction, accounting for anisotropic grid spacing.

**Eddy Viscosity:**
```
νₜ = max(0, -(∂̂ₖũᵢ)(∂̂ₖũⱼ)S̃ᵢⱼ / (∂ₗũₘ)(∂ₗũₘ))
```

**Eddy Diffusivity:**
```
Dₜ = max(0, -(∂̂ₖũᵢ)(∂̂ₖθ̃)∂ᵢθ̃ / (∂ₗθ̃)(∂ₗθ̃))
```

To remove this tendency in debugging workflows, comment (or delete) the call to this function
in `remaining_tendencies.jl`

"""
function horizontal_amd_tendency!(Yₜ, Y, p, t, les::AnisotropicMinimumDissipation)
    (; atmos, precomputed, scratch, params) = p
    FT = eltype(Y)
    c_amd = les.c_amd
    grav = CAP.grav(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜts) = precomputed
    (; ᶜtemp_scalar, ᶠtemp_scalar_2) = scratch

    # Use cached precomputed quantities
    (;
        ᶜ∂̂u_uvw,
        ᶠ∂̂u_uvw,
        ᶜS,
        ᶠS,
        ᶜ∂ₖuᵢ∂ₖuⱼ,
        ᶜ∂ₗuₘ∂ₗuₘ,
        ᶜδ²,
        ᶜνₜ,
        ᶠνₜ,
    ) = precomputed

    axis_uvw = (Geometry.UVWAxis(),)

    # Filter scale (needed for energy and tracer diffusion)
    h_space = Spaces.horizontal_space(axes(Y.c))
    Δ_h = Spaces.node_horizontal_length_scale(h_space)

    # Subgrid-scale momentum flux tensor, `τ = -2 νₜ ∘ S`
    (; ᶜτ_amd, ᶠτ_amd) = precomputed
    @. ᶜτ_amd = -2 * ᶜνₜ * ᶜS
    @. ᶠτ_amd = -2 * ᶠνₜ * ᶠS

    ## Momentum tendencies
    ᶠρ = @. ᶠtemp_scalar_2 = ᶠinterp(Y.c.ρ)
    @. Yₜ.c.uₕ -= C12(wdivₕ(Y.c.ρ * ᶜτ_amd) / Y.c.ρ)
    @. Yₜ.f.u₃ -= C3(wdivₕ(ᶠρ * ᶠτ_amd) / ᶠρ)

    ## Total energy tendency
    ᶜe_tot = @. lazy(specific(Y.c.ρe_tot, Y.c.ρ))
    ᶜh_tot = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, ᶜe_tot))
    ∇h_tot = @. lazy(Geometry.project(axis_uvw, gradₕ(ᶜh_tot)))
    ∂̂h_tot = @. lazy(Δ_h * ∇h_tot)
    
    ᶜD_amd = @. ᶜtemp_scalar = max(
        FT(0),
        -c_amd^2 * ᶜδ² *
        (
            (ᶜ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:1 +
            (ᶜ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:5 +
            (ᶜ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:9
        ) /
        max(eps(FT), CA.norm_sqr(∂̂h_tot)),
    )
    @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD_amd * gradₕ(ᶜh_tot))

    # Tracer diffusion and associated mass changes
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        ∇ᶜχ = @. lazy(Geometry.project(axis_uvw, gradₕ(ᶜχ)))
        ∂̂ᶜχ = @. lazy(Δ_h * ∇ᶜχ)
        @. ᶜD_amd = max(
            FT(0),
            -c_amd^2 * ᶜδ² *
            (
                (ᶜ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:1 +
                (ᶜ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:5 +
                (ᶜ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:9
            ) /
            max(eps(FT), CA.norm_sqr(∂̂ᶜχ)),
        )
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
    vertical_amd_tendency!(Yₜ,Y,p,t, ::AnisotropicMinimumDissipation)

This function implements the vertical component of the AMD subgrid-scale model as specified
by Abkar et al. (2016). It computes eddy viscosity and diffusivity based on the minimum
dissipation principle and applies them to vertical momentum, energy, and tracer transport.

**Scaled Gradient Operator:**
```
∂̂ᵢ = Δᵢ ∂ᵢ
```
where Δᵢ is the filter width in direction i.

**Eddy Viscosity:**
```
νₜ = max(0, -(∂̂ₖũᵢ)(∂̂ₖũⱼ)S̃ᵢⱼ / (∂ₗũₘ)(∂ₗũₘ))
```
**Eddy Diffusivity:**
```
Dₜ = max(0, -(∂̂ₖũᵢ)(∂̂ₖθ̃)∂ᵢθ̃ / (∂ₗθ̃)(∂ₗθ̃))
```
The scaled gradients ∂̂∇u account for anisotropic filter scales in the vertical direction.

To remove this tendency in debugging workflows, comment (or delete) the call to this function
in `remaining_tendencies.jl`

"""
function vertical_amd_tendency!(Yₜ, Y, p, t, les::AnisotropicMinimumDissipation)
    FT = eltype(Y)
    (; sfc_temp_C3) = p.scratch
    (; ᶜts, sfc_conditions) = p.precomputed
    (; ρ_flux_uₕ, ρ_flux_h_tot) = sfc_conditions
    thermo_params = CAP.thermodynamics_params(p.params)

    c_amd = les.c_amd

    # Define operators
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ
    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        bottom = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
    )
    ᶠdivᵥ = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(FT(0)),
        top = Operators.SetDivergence(FT(0)),
    )
    ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(;
        top = Operators.SetValue(C3(FT(0))),
        bottom = Operators.SetValue(C3(FT(0))),
    )

    ### AMD ###

    (; ᶜts) = p.precomputed
    (; ᶠtemp_scalar) = p.scratch

    # Use cached precomputed quantities
    (;
        ᶜ∂̂u_uvw,
        ᶠ∂̂u_uvw,
        ᶜS,
        ᶠS,
        ᶜδ²,
        ᶠδ²,
        ᶜνₜ,
        ᶠνₜ,
    ) = p.precomputed

    axis_uvw = (Geometry.UVWAxis(),)

    # Filter scale (needed for energy and tracer diffusion)
    h_space = Spaces.horizontal_space(axes(Y.c))
    Δ_h = Spaces.node_horizontal_length_scale(h_space)
    ᶠΔ_z = Fields.Δz_field(Y.f)

    ᶠgradᵥ_scalar = Operators.GradientC2F(
        bottom = Operators.SetGradient(UVW(0, 0, 0)),
        top = Operators.SetGradient(UVW(0, 0, 0)),
    )

    # Subgrid-scale momentum flux tensor, `τ = -2 νₜ ∘ S`
    (; ᶜτ_amd, ᶠτ_amd) = p.precomputed
    @. ᶜτ_amd = -2 * ᶜνₜ * ᶜS
    @. ᶠτ_amd = -2 * ᶠνₜ * ᶠS

    # Apply to tendencies
    ## Horizontal momentum tendency
    ᶠρ = @. lazy(ᶠinterp(Y.c.ρ))
    @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(ᶠρ * ᶠτ_amd) / Y.c.ρ)
    ## Apply boundary condition for momentum flux
    @. Yₜ.c.uₕ -= ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ))) / Y.c.ρ
    ## Vertical momentum tendency
    @. Yₜ.f.u₃ -= C3(ᶠdivᵥ(Y.c.ρ * ᶜτ_amd) / ᶠρ)

    ## Total energy tendency
    ᶜe_tot = @. lazy(specific(Y.c.ρe_tot, Y.c.ρ))
    ᶜh_tot = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, ᶜe_tot))
    # TODO: Fix @lazy broadcast (components access)
    ∇h_tot = @. lazy(Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜh_tot)))
    ∂̂h_tot = @. lazy(ᶠΔ_z * ∇h_tot)
    ᶠD_amd = @. ᶠtemp_scalar = max(
        FT(0),
        -c_amd^2 * ᶠδ² *
        (
            (ᶠ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:1 +
            (ᶠ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:5 +
            (ᶠ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:9
        ) /
        max(eps(FT), CA.norm_sqr(∂̂h_tot)),
    )
    @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(-(ᶠρ * ᶠD_amd * ᶠgradᵥ(ᶜh_tot)))

    ## Tracer diffusion and associated mass changes
    ᶜdivᵥ_ρχ = Operators.DivergenceF2C(;
        top = Operators.SetValue(C3(FT(0))),
        bottom = Operators.SetValue(C3(FT(0))),
    )

    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        ∇ᶜχ = @. lazy(Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜχ)))
        ∂̂ᶜχ = @. lazy(ᶠΔ_z * ∇ᶜχ)
        @. ᶠD_amd = max(
            FT(0),
            -c_amd^2 * ᶠδ² *
            (
                (ᶠ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:1 +
                (ᶠ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:5 +
                (ᶠ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:9
            ) /
            max(eps(FT), CA.norm_sqr(∂̂ᶜχ)),
        )
        ᶜ∇ᵥρD∇χₜ =
            @. lazy(ᶜdivᵥ_ρχ(-(ᶠρ * ᶠD_amd * ᶠgradᵥ(specific(ᶜρχ, Y.c.ρ)))))
        @. ᶜρχₜ -= ᶜ∇ᵥρD∇χₜ
        # Rain and snow does not affect the mass
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ -= ᶜ∇ᵥρD∇χₜ
        end
    end
end
