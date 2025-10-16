#####
##### Smagorinsky Lilly Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import ClimaCore: Geometry

"""
    set_smagorinsky_lilly_precomputed_quantities!(Y, p)

Compute the Smagorinsky-Lilly horizontal and vertical length scales `ᶜL_h` and `ᶜL_v`.
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
    FT = eltype(Y)
    (; ᶠu, ᶜts, ᶜL_h, ᶜL_v) = p.precomputed
    c_smag = CAP.c_smag(p.params)
    # grav = CAP.grav(p.params)
    # Pr_t = CAP.Prandtl_number_0(CAP.turbconv_params(p.params))
    # thermo_params = CAP.thermodynamics_params(p.params)
    # (; ᶜtemp_scalar, ᶜtemp_scalar_2) = p.scratch

    # Strain rate tensor
    # ᶜS = compute_strain_rate_center(ᶠu)

    # Stratification correction
    # ᶜθ_v = @. lazy(TD.virtual_pottemp(thermo_params, ᶜts))
    # ᶜ∇ᵥθ = @. ᶜtemp_scalar_2 = Geometry.WVector(ᶜgradᵥ(ᶠinterp(ᶜθ_v))).components.data.:1
    # ᶜN² = @. ᶜtemp_scalar = grav / ᶜθ_v * ᶜ∇ᵥθ
    # ᶜS_norm = @. ᶜtemp_scalar_2 = √(2 * norm_sqr(ᶜS))

    # ᶜRi = @. ᶜtemp_scalar = ᶜN² / (ᶜS_norm^2 + eps(FT))  # Ri = N² / |S|²
    # ᶜfb = @. ᶜtemp_scalar = ifelse(ᶜRi ≤ 0, 1, max(0, 1 - ᶜRi / Pr_t)^(1 / 4))

    # filter scale
    h_space = Spaces.horizontal_space(axes(Y.c))
    Δ_h = Spaces.node_horizontal_length_scale(h_space)
    ᶜΔ_z = Fields.Δz_field(Y.c)

    @. ᶜL_v = c_smag * ᶜΔ_z #* ᶜfb
    @. ᶜL_h = c_smag * Δ_h

    nothing
end

horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::SmagorinskyLilly)
    (; ᶜu, ᶠu, ᶜts, ᶜL_h) = p.precomputed
    (; ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW, ᶜtemp_scalar, ᶠtemp_scalar) = p.scratch
    thermo_params = CAP.thermodynamics_params(p.params)
    Pr_t = CAP.Prandtl_number_0(CAP.turbconv_params(p.params))

    ## Momentum tendencies
    ᶜS = compute_strain_rate_center(ᶠu)
    ᶠS = compute_strain_rate_face(ᶜu)
    ᶜS_h = @. lazy(Geometry.project((Geometry.UVAxis(),), ᶜS, (Geometry.UVAxis(),)))
    ᶜS_norm = @. lazy(√(2 * norm_sqr(ᶜS_h)))

    # Smagorinsky eddy viscosity
    ᶜνₜ_h = @. lazy(ᶜL_h^2 * ᶜS_norm)
    ᶠνₜ_h = @. lazy(ᶠinterp(ᶜνₜ_h))

    # Turbulent diffusivity
    ᶜD_smag = @. ᶜtemp_scalar = ᶜνₜ_h / Pr_t

    # Subgrid-scale momentum flux tensor, `τ = -2 νₜ ∘ S`
    ᶜτ_smag = @. ᶜtemp_UVWxUVW = -2 * ᶜνₜ_h * ᶜS  # TODO: Lazify once we can mix lazy horizontal & vertical operations
    ᶠτ_smag = @. ᶠtemp_UVWxUVW = -2 * ᶠνₜ_h * ᶠS

    ᶠρ = ᶠtemp_scalar .= face_density(Y)
    @. Yₜ.c.uₕ -= C12(wdivₕ(Y.c.ρ * ᶜτ_smag) / Y.c.ρ)
    @. Yₜ.f.u₃ -= C3(wdivₕ(ᶠρ * ᶠτ_smag) / ᶠρ)

    ## Total energy tendency
    ᶜh_tot = @. lazy(
        TD.total_specific_enthalpy(thermo_params, ᶜts, specific(Y.c.ρe_tot, Y.c.ρ)),
    )
    @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD_smag * gradₕ(ᶜh_tot))

    ## Tracer diffusion and associated mass changes
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        ᶜρχₜ_diffusion = @. lazy(wdivₕ(Y.c.ρ * ᶜD_smag * gradₕ(ᶜχ)))
        @. ᶜρχₜ += ᶜρχₜ_diffusion
        # Rain and snow does not affect the mass
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ += ᶜρχₜ_diffusion
        end
    end

end

function vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::SmagorinskyLilly)
    FT = eltype(Y)
    (; ᶜu, ᶠu, ᶜts, ᶜL_v) = p.precomputed
    (; ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW, ᶠtemp_scalar, ᶠtemp_scalar_2) = p.scratch
    thermo_params = CAP.thermodynamics_params(p.params)
    Pr_t = CAP.Prandtl_number_0(CAP.turbconv_params(p.params))

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

    ## Momentum tendencies
    ᶜS = compute_strain_rate_center(ᶠu)
    ᶠS = compute_strain_rate_face(ᶜu)
    ᶜS_v = @. lazy(Geometry.project((Geometry.WAxis(),), ᶜS, (Geometry.WAxis(),)))
    ᶜS_norm = @. lazy(√(2 * norm_sqr(ᶜS_v)))

    # Smagorinsky eddy viscosity
    ᶜνₜ_v = @. lazy(ᶜL_v^2 * ᶜS_norm)
    ᶠνₜ_v = @. lazy(ᶠinterp(ᶜνₜ_v))

    # Subgrid-scale momentum flux tensor, `τ = -2 νₜ ∘ S`
    ᶜτ_smag = @. ᶜtemp_UVWxUVW = -2 * ᶜνₜ_v * ᶜS
    ᶠτ_smag = @. ᶠtemp_UVWxUVW = -2 * ᶠνₜ_v * ᶠS

    # Turbulent diffusivity
    ᶠD_smag = @. ᶠtemp_scalar_2 = ᶠνₜ_v / Pr_t

    # Apply to tendencies
    ## Horizontal momentum tendency
    ᶠρ = ᶠtemp_scalar .= face_density(Y)
    @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(ᶠρ * ᶠτ_smag) / Y.c.ρ)
    ## Apply boundary condition for momentum flux
    @. Yₜ.c.uₕ -= ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ))) / Y.c.ρ
    ## Vertical momentum tendency
    @. Yₜ.f.u₃ -= C3(ᶠdivᵥ(Y.c.ρ * ᶜτ_smag) / ᶠρ)

    ## Total energy tendency
    ᶜh_tot = @. lazy(
        TD.total_specific_enthalpy(thermo_params, ᶜts, specific(Y.c.ρe_tot, Y.c.ρ)),
    )
    @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(-(ᶠρ * ᶠD_smag * ᶠgradᵥ(ᶜh_tot)))

    ## Tracer diffusion and associated mass changes
    ᶜdivᵥ_ρχ = Operators.DivergenceF2C(;
        top = Operators.SetValue(C3(FT(0))),
        bottom = Operators.SetValue(C3(FT(0))),
    )

    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜ∇ᵥρD∇χₜ = @. lazy(ᶜdivᵥ_ρχ(-(ᶠρ * ᶠD_smag * ᶠgradᵥ(specific(ᶜρχ, Y.c.ρ)))))
        @. ᶜρχₜ -= ᶜ∇ᵥρD∇χₜ
        # Rain and snow does not affect the mass
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ -= ᶜ∇ᵥρD∇χₜ
        end
    end
end
