#####
##### Smagorinsky Lilly Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import ClimaCore: Geometry

"""
    set_smagorinsky_lilly_precomputed_quantities!(Y, p)

Compute the Smagorinsky-Lilly diffusivity tensors, `ᶜτ_smag`, `ᶠτ_smag`, `ᶜD_smag`, and `ᶠD_smag`. 
Store in the precomputed quantities `p.precomputed`.

The subgrid-scale momentum flux tensor is defined by `τ = -2 νₜ ∘ S`, where `νₜ` is the Smagorinsky-Lilly eddy viscosity 
and `S` is the strain rate tensor. 

The turbulent diffusivity is defined as `D = νₜ / Pr_t`, where `Pr_t` is the turbulent Prandtl number for neutral 
stratification. 

These quantities are computed for both cell centers and faces, with prefixes `ᶜ` and `ᶠ`, respectively.


# Arguments
- `Y`: The model state.
- `p`: The model parameters, e.g. `AtmosCache`.
"""
function set_smagorinsky_lilly_precomputed_quantities!(Y, p)

    (; atmos, precomputed, scratch, params) = p
    c_smag = CAP.c_smag(params)
    Pr_t = CAP.Prandtl_number_0(CAP.turbconv_params(params))
    (; ᶜu, ᶠu³, ᶜts, ᶜτ_smag, ᶠτ_smag, ᶜD_smag, ᶠD_smag) = precomputed
    FT = eltype(Y)
    grav = CAP.grav(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW, ᶜtemp_strain, ᶠtemp_strain) = scratch
    (; ᶜtemp_scalar, ᶜtemp_scalar_2, ᶠtemp_scalar, ᶜtemp_UVW, ᶠtemp_UVW) =
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

    # Stratification correction
    ᶜθ_v = @. ᶜtemp_scalar = TD.virtual_pottemp(thermo_params, ᶜts)
    ᶜ∇ᵥθ = @. ᶜtemp_scalar_2 =
        Geometry.WVector(ᶜgradᵥ(ᶠinterp(ᶜθ_v))).components.data.:1
    ᶜN² = @. ᶜtemp_scalar = grav / ᶜθ_v * ᶜ∇ᵥθ
    ᶜS_norm = @. ᶜtemp_scalar_2 = √(2 * CA.norm_sqr(ᶜS))

    ᶜRi = @. ᶜtemp_scalar = ᶜN² / (ᶜS_norm^2 + eps(FT))  # Ri = N² / |S|²
    ᶜfb = @. ᶜtemp_scalar = ifelse(ᶜRi ≤ 0, 1, max(0, 1 - ᶜRi / Pr_t)^(1 / 4))

    # filter scale
    h_space = Spaces.horizontal_space(axes(Y.c))
    Δ_xy = Spaces.node_horizontal_length_scale(h_space)^2 # Δ_x * Δ_y
    ᶜΔ_z = Fields.Δz_field(Y.c)
    ᶜΔ = @. ᶜtemp_scalar = ∛(Δ_xy * ᶜΔ_z) * ᶜfb

    # Smagorinsky-Lilly eddy viscosity
    ᶜνₜ = @. ᶜtemp_scalar = c_smag^2 * ᶜΔ^2 * ᶜS_norm
    ᶠνₜ = @. ᶠtemp_scalar = ᶠinterp(ᶜνₜ)

    # Subgrid-scale momentum flux tensor, `τ = -2 νₜ ∘ S`
    @. ᶜτ_smag = -2 * ᶜνₜ * ᶜS
    @. ᶠτ_smag = -2 * ᶠνₜ * ᶠS

    # Turbulent diffusivity
    @. ᶜD_smag = ᶜνₜ / Pr_t
    @. ᶠD_smag = ᶠνₜ / Pr_t

    nothing
end

horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::SmagorinskyLilly)
    (; ᶜτ_smag, ᶠτ_smag, ᶜD_smag, ᶜh_tot) = p.precomputed

    ## Momentum tendencies
    ᶠρ = @. p.scratch.ᶠtemp_scalar = ᶠinterp(Y.c.ρ)
    @. Yₜ.c.uₕ -= C12(wdivₕ(Y.c.ρ * ᶜτ_smag) / Y.c.ρ)
    @. Yₜ.f.u₃ -= C3(wdivₕ(ᶠρ * ᶠτ_smag) / ᶠρ)

    ## Total energy tendency
    @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD_smag * gradₕ(ᶜh_tot))

    ## Tracer diffusion and associated mass changes
    for (ᶜρχₜ, Ycρq, χ_name) in matching_ρ(Y, Yₜ.c)
        χ_name == :e_tot && continue
        ᶜρχₜ_diffusion = @. p.scratch.ᶜtemp_scalar =
            wdivₕ(Y.c.ρ * ᶜD_smag * gradₕ(Ycρq / Y.c.ρ))
        @. ᶜρχₜ += ᶜρχₜ_diffusion
        # Rain and snow does not affect the mass
        if χ_name ∉ (:q_rai, :q_sno)
            @. Yₜ.c.ρ += ᶜρχₜ_diffusion
        end
    end

end

function vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::SmagorinskyLilly)
    FT = eltype(Y)
    (; sfc_temp_C3, ᶠtemp_scalar, ᶜtemp_scalar) = p.scratch
    (; ᶜτ_smag, ᶠτ_smag, ᶠD_smag, ᶜh_tot, sfc_conditions) = p.precomputed
    (; ρ_flux_uₕ, ρ_flux_h_tot) = sfc_conditions

    # Define operators
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ
    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        bottom = Operators.SetValue(ρ_flux_uₕ),
    )
    ᶠdivᵥ = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(FT(0)),
        top = Operators.SetDivergence(FT(0)),
    )
    top = Operators.SetValue(C3(FT(0)))
    ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(;
        top,
        bottom = Operators.SetValue(ρ_flux_h_tot),
    )

    # Apply to tendencies
    ## Horizontal momentum tendency
    ᶠρ = @. ᶠtemp_scalar = ᶠinterp(Y.c.ρ)
    @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(ᶠρ * ᶠτ_smag) / Y.c.ρ)
    ## Apply boundary condition for momentum flux
    @. Yₜ.c.uₕ -= ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ))) / Y.c.ρ
    ## Vertical momentum tendency
    @. Yₜ.f.u₃ -= C3(ᶠdivᵥ(Y.c.ρ * ᶜτ_smag) / ᶠρ)

    ## Total energy tendency
    @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(-(ᶠρ * ᶠD_smag * ᶠgradᵥ(ᶜh_tot)))

    ## Tracer diffusion and associated mass changes
    sfc_zero = @. sfc_temp_C3 = C3(FT(0))
    for (ᶜρχₜ, Ycρq, χ_name) in matching_ρ(Y, Yₜ.c)
        χ_name == :e_tot && continue

        bottom = Operators.SetValue(
            χ_name == :q_tot ? sfc_conditions.ρ_flux_q_tot : sfc_zero,
        )
        ᶜdivᵥ_ρχ = Operators.DivergenceF2C(; top, bottom)
        ᶜ∇ᵥρD∇χₜ =
            @. ᶜtemp_scalar = ᶜdivᵥ_ρχ(-(ᶠρ * ᶠD_smag * ᶠgradᵥ(Ycρq / Y.c.ρ)))
        @. ᶜρχₜ -= ᶜ∇ᵥρD∇χₜ
        # Rain and snow does not affect the mass
        if χ_name ∉ (:q_rai, :q_sno)
            @. Yₜ.c.ρ -= ᶜ∇ᵥρD∇χₜ
        end
    end
end
