#####
##### AMD Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import ClimaCore: Geometry
import LinearAlgebra: tr

"""
    set_amd_precomputed_quantities!(Y, p)

Placeholder for precomputed quantities in the Anisotropic-Minimum-Dissipation method.
Returns `nothing`. This function is included for simple extensions in debugging workflows.
"""
function set_amd_precomputed_quantities!(Y, p)
    nothing
end

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
    (; ᶜu, ᶠu³) = precomputed
    (; ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW, ᶜtemp_strain, ᶠtemp_strain) = scratch
    (; ᶜtemp_scalar, ᶠtemp_scalar, ᶠtemp_scalar_2, ᶜtemp_UVW, ᶠtemp_UVW) =
        scratch

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

    # Scaled Derivatives ∂̂ᵢ = Δ₍ᵢ₎∂ᵢ
    ᶜ∂̂u_uvw = @.ᶜtemp_UVWxUVW = Δ_h * Geometry.project(axis_uvw, gradₕ(ᶜu_uvw))
    @. ᶜ∂̂u_uvw += ᶜΔ_z * Geometry.project(axis_uvw, ᶜgradᵥ(ᶠu_uvw))

    ᶠ∂̂u_uvw = @.ᶠtemp_UVWxUVW = Δ_h * Geometry.project(axis_uvw, gradₕ(ᶠu_uvw))
    @. ᶠ∂̂u_uvw += ᶠΔ_z * Geometry.project(axis_uvw, ᶠgradᵥ_uvw(ᶜu_uvw))

    ᶜ∂ₖuᵢ∂ₖuⱼ = @. lazy(ᶜ∂̂u_uvw * adjoint(ᶜ∂̂u_uvw))
    ᶠ∂ₖuᵢ∂ₖuⱼ = @. lazy(ᶠ∂̂u_uvw * adjoint(ᶠ∂̂u_uvw))
    ᶜ∂ₗuₘ∂ₗuₘ = @. lazy(CA.norm_sqr(∇ᶜu_uvw))

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

    ## Total energy tendency
    (; ᶜh_tot) = precomputed
    ∇h_tot = @. lazy(Geometry.project(axis_uvw, gradₕ(ᶜh_tot)))
    ∂̂h_tot = @. lazy(Δ_h * ∇h_tot)
    ᶜD_amd = @. ᶜtemp_scalar = max(
        FT(0),
        -c_amd *
        (
            (ᶜ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:1 +
            (ᶜ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:5 +
            (ᶜ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:9
        ) /
        max(eps(FT), CA.norm_sqr(gradₕ(ᶜh_tot))),
    )
    @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD_amd * gradₕ(ᶜh_tot))

    # Tracer diffusion and associated mass changes
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        ∇ᶜχ = @. lazy(Geometry.project(axis_uvw, gradₕ(ᶜχ)))
        ∂̂ᶜχ = @. lazy(Δ_h * ∇ᶜχ)
        @. ᶜD_amd = max(
            FT(0),
            -c_amd *
            (
                (ᶜ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:1 +
                (ᶜ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:5 +
                (ᶜ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:9
            ) /
            max(eps(FT), CA.norm_sqr(gradₕ(ᶜχ))),
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

    (; ᶜu, ᶠu³) = p.precomputed
    (; ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW, ᶜtemp_strain, ᶠtemp_strain) = p.scratch
    (; ᶜtemp_scalar, ᶠtemp_scalar, ᶜtemp_UVW, ᶠtemp_UVW) =
        p.scratch

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

    # Do we need scratch variables at all?
    # Scaled Derivatives ∂̂ᵢ = Δ₍ᵢ₎∂ᵢ
    ᶜ∂̂u_uvw = @.ᶜtemp_UVWxUVW = Δ_h * Geometry.project(axis_uvw, gradₕ(ᶜu_uvw))
    @. ᶜ∂̂u_uvw += ᶜΔ_z * Geometry.project(axis_uvw, ᶜgradᵥ(ᶠu_uvw))

    ᶠ∂̂u_uvw = @.ᶠtemp_UVWxUVW = Δ_h * Geometry.project(axis_uvw, gradₕ(ᶠu_uvw))
    @. ᶠ∂̂u_uvw += ᶠΔ_z * Geometry.project(axis_uvw, ᶠgradᵥ_uvw(ᶜu_uvw))

    ᶜ∂ₖuᵢ∂ₖuⱼ = @. lazy(ᶜ∂̂u_uvw * adjoint(ᶜ∂̂u_uvw))
    ᶠ∂ₖuᵢ∂ₖuⱼ = @. lazy(ᶠ∂̂u_uvw * adjoint(ᶠ∂̂u_uvw))
    ᶜ∂ₗuₘ∂ₗuₘ = @. lazy(CA.norm_sqr(∇ᶜu_uvw))

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
    ## Apply boundary condition for momentum flux
    @. Yₜ.c.uₕ -= ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ))) / Y.c.ρ
    ## Vertical momentum tendency
    @. Yₜ.f.u₃ -= C3(ᶠdivᵥ(Y.c.ρ * ᶜτ_amd) / ᶠρ)

    ## Total energy tendency
    (; ᶜh_tot) = p.precomputed
    # TODO: Fix @lazy broadcast (components access)
    ∇h_tot = @. lazy(Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜh_tot)))
    ∂̂h_tot = @. lazy(ᶠΔ_z * ∇h_tot)
    ᶠD_amd = @. ᶠtemp_scalar = max(
        FT(0),
        -c_amd *
        (
            (ᶠ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:1 +
            (ᶠ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:5 +
            (ᶠ∂̂u_uvw * ∂̂h_tot ⊗ ∇h_tot).components.data.:9
        ) /
        max(eps(FT), CA.norm_sqr(∇h_tot)),
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
            -c_amd *
            (
                (ᶠ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:1 +
                (ᶠ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:5 +
                (ᶠ∂̂u_uvw * ∂̂ᶜχ ⊗ ∇ᶜχ).components.data.:9
            ) /
            max(eps(FT), CA.norm_sqr(∇ᶜχ)),
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
