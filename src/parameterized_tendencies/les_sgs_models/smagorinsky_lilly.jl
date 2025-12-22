#####
##### Smagorinsky Lilly Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import ClimaCore: Geometry

"""
    lilly_stratification_correction(p, ᶜS)

Return a lazy representation of the Lilly stratification correction factor 
    based on the local Richardson number.

# Arguments
- `p`: The model parameters, e.g. `AtmosCache`.
- `ᶜS`: The cell-centered strain rate tensor.
"""
function lilly_stratification_correction(p, ᶜS)
    (; ᶜts) = p.precomputed
    (; ᶜtemp_scalar) = p.scratch
    grav = CAP.grav(p.params)
    Pr_t = CAP.Prandtl_number_0(CAP.turbconv_params(p.params))
    thermo_params = CAP.thermodynamics_params(p.params)
    FT = eltype(Pr_t)
    # Stratification correction
    ᶜθ_v = @. lazy(TD.virtual_pottemp(thermo_params, ᶜts))
    ᶜ∇ᵥθ = @. ᶜtemp_scalar = Geometry.WVector(ᶜgradᵥ(ᶠinterp(ᶜθ_v))).components.data.:1
    ᶜN² = @. lazy(grav / ᶜθ_v * ᶜ∇ᵥθ)
    ᶜS_norm = strain_rate_norm(ᶜS, Geometry.WAxis())

    ᶜRi = @. lazy(ᶜN² / (ᶜS_norm^2 + eps(FT)))  # Ri = N² / |S|²
    ᶜfb = @. lazy(ifelse(ᶜRi ≤ 0, FT(1), max(0, 1 - ᶜRi / Pr_t)^(1 // 4)))
end

"""
    set_smagorinsky_lilly_precomputed_quantities!(Y, p)

Compute the Smagorinsky-Lilly horizontal and vertical quantities needed for 
    subgrid-scale diffusive tendencies

The subgrid-scale momentum flux tensor is defined by `τ = -2 νₜ ∘ S`, 
where `νₜ` is the Smagorinsky-Lilly eddy viscosity and `S` is the strain rate tensor. 

The turbulent diffusivity is defined as `D = νₜ / Pr_t`, 
where `Pr_t` is the turbulent Prandtl number for neutral stratification.

This method precomputes and stores in `p.precomputed` the following quantities:
- strain on centers and faces: `ᶜS`, `ᶠS`
- horizontal and vertical strain rate norm, eddy viscosities, and diffusivities, on centers:
    - `ᶜS_norm_h`, `ᶜS_norm_v`, `ᶜνₜ_h`, `ᶜνₜ_v`, `ᶜD_h`, `ᶜD_v`

# Arguments
- `Y`: The model state.
- `p`: The model parameters, e.g. `AtmosCache`.
- `model`: The Smagorinsky model type
"""
function set_smagorinsky_lilly_precomputed_quantities!(Y, p, model)
    (; ᶜu, ᶠu, ᶜS, ᶠS, ᶜL_h, ᶜL_v, ᶜS_norm_h, ᶜS_norm_v, ᶜνₜ_h, ᶜνₜ_v, ᶜD_h, ᶜD_v) =
        p.precomputed
    (; ᶜtemp_scalar) = p.scratch
    c_smag = CAP.c_smag(p.params)

    # Precompute 3D strain rate tensor
    compute_strain_rate_center_full!(ᶜS, ᶜu, ᶠu)
    compute_strain_rate_face_full!(ᶠS, ᶜu, ᶠu)

    # filter scale
    h_space = Spaces.horizontal_space(axes(Y.c))
    Δx = Δy = Spaces.node_horizontal_length_scale(h_space)
    ᶜΔz = Fields.Δz_field(Y.c)
    ax_xy = is_smagorinsky_UVW_coupled(model) ? Geometry.UVWAxis() : Geometry.UVAxis()
    ax_z = is_smagorinsky_UVW_coupled(model) ? Geometry.UVWAxis() : Geometry.WAxis()

    # ᶜfb = lilly_stratification_correction(p, ᶜS)
    ᶜfb = eltype(c_smag)(1)
    if is_smagorinsky_UVW_coupled(model)
        ᶜL_h = ᶜL_v = @. lazy(c_smag * cbrt(Δx * Δy * ᶜΔz) * ᶜfb)
    else
        ᶜL_h = @. lazy(c_smag * Δx)
        ᶜL_v = @. lazy(c_smag * ᶜΔz * ᶜfb)
    end

    # Cache strain rate norms for diagnostics
    ᶜS_norm_h .= strain_rate_norm(ᶜS, ax_xy)
    ᶜS_norm_v .= strain_rate_norm(ᶜS, ax_z)

    # Smagorinsky eddy viscosity
    @. ᶜνₜ_h = ᶜL_h^2 * ᶜS_norm_h
    @. ᶜνₜ_v = ᶜL_v^2 * ᶜS_norm_v

    # Turbulent diffusivity
    Pr_t = CAP.Prandtl_number_0(CAP.turbconv_params(p.params))
    @. ᶜD_h = ᶜνₜ_h / Pr_t
    @. ᶜD_v = ᶜνₜ_v / Pr_t

    nothing
end
set_smagorinsky_lilly_precomputed_quantities!(Y, p, ::Nothing) = nothing

horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, model::SmagorinskyLilly)
    is_smagorinsky_horizontal(model) || return nothing
    (; ᶜts, ᶜS, ᶠS, ᶜνₜ_h, ᶜD_h) = p.precomputed
    (; ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW, ᶜtemp_scalar, ᶠtemp_scalar) = p.scratch
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜρ = Y.c.ρ
    ᶠρ = @. ᶠtemp_scalar = ᶠinterp(ᶜρ)

    # Subgrid-scale momentum flux tensor, `τ = -2 νₜ ∘ S`
    ᶠνₜ_h = @. lazy(ᶠinterp(ᶜνₜ_h))
    ᶜτ_smag = @. ᶜtemp_UVWxUVW = -2 * ᶜνₜ_h * ᶜS  # TODO: Lazify once we can mix lazy horizontal & vertical operations
    ᶠτ_smag = @. ᶠtemp_UVWxUVW = -2 * ᶠνₜ_h * ᶠS

    # Apply to tendencies
    ## Horizontal momentum tendency
    @. Yₜ.c.uₕ -= C12(wdivₕ(ᶜρ * ᶜτ_smag) / ᶜρ)
    ## Vertical momentum tendency
    @. Yₜ.f.u₃ -= C3(wdivₕ(ᶠρ * ᶠτ_smag) / ᶠρ)

    ## Total energy tendency
    ᶜe_tot = @. lazy(specific(Y.c.ρe_tot, ᶜρ))
    ᶜh_tot = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, ᶜe_tot))
    @. Yₜ.c.ρe_tot += wdivₕ(ᶜρ * ᶜD_h * gradₕ(ᶜh_tot))

    ## Tracer diffusion and associated mass changes
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, ᶜρ))
        ᶜ∇ₕρD∇χₜ = @. lazy(wdivₕ(ᶜρ * ᶜD_h * gradₕ(ᶜχ)))
        @. ᶜρχₜ += ᶜ∇ₕρD∇χₜ
        # Rain and snow does not affect the mass
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ += ᶜ∇ₕρD∇χₜ
        end
    end
end

function vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, model::SmagorinskyLilly)
    is_smagorinsky_vertical(model) || return nothing
    FT = eltype(Y)
    (; ᶜts, ᶜS, ᶠS, ᶜνₜ_v) = p.precomputed
    (; ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW, ᶠtemp_scalar, ᶠtemp_scalar_2) = p.scratch
    Pr_t = CAP.Prandtl_number_0(CAP.turbconv_params(p.params))
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜρ = Y.c.ρ
    ᶠρ = @. ᶠtemp_scalar = ᶠinterp(ᶜρ)

    # Define operators
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ
    divᵥ_uₕ_bc = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0)))
    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(; bottom = divᵥ_uₕ_bc, top = divᵥ_uₕ_bc)
    divᵥ_bc = Operators.SetDivergence(FT(0))
    ᶠdivᵥ = Operators.DivergenceC2F(; bottom = divᵥ_bc, top = divᵥ_bc)
    divᵥ_ρχ_bc = Operators.SetValue(C3(FT(0)))
    ᶜdivᵥ_ρχ = Operators.DivergenceF2C(; bottom = divᵥ_ρχ_bc, top = divᵥ_ρχ_bc)

    # Subgrid-scale momentum flux tensor, `τ = -2 νₜ ∘ S`
    ᶠνₜ_v = @. lazy(ᶠinterp(ᶜνₜ_v))
    ᶜτ_smag = @. ᶜtemp_UVWxUVW = -2 * ᶜνₜ_v * ᶜS
    ᶠτ_smag = @. ᶠtemp_UVWxUVW = -2 * ᶠνₜ_v * ᶠS

    # Turbulent diffusivity
    ᶠD_smag = @. lazy(ᶠνₜ_v / Pr_t)

    # Apply to tendencies
    ## Horizontal momentum tendency
    @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(ᶠρ * ᶠτ_smag) / ᶜρ)
    ## Apply boundary condition for momentum flux
    @. Yₜ.c.uₕ -= ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ))) / ᶜρ
    ## Vertical momentum tendency
    @. Yₜ.f.u₃ -= C3(ᶠdivᵥ(ᶜρ * ᶜτ_smag) / ᶠρ)

    ## Total energy tendency
    ᶜe_tot = @. lazy(specific(Y.c.ρe_tot, ᶜρ))
    ᶜh_tot = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, ᶜe_tot))
    @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρχ(-(ᶠρ * ᶠD_smag * ᶠgradᵥ(ᶜh_tot)))

    ## Tracer diffusion and associated mass changes
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, ᶜρ))
        ᶜ∇ᵥρD∇χₜ = @. lazy(ᶜdivᵥ_ρχ(-(ᶠρ * ᶠD_smag * ᶠgradᵥ(ᶜχ))))
        @. ᶜρχₜ -= ᶜ∇ᵥρD∇χₜ
        # Rain and snow does not affect the mass
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ -= ᶜ∇ᵥρD∇χₜ
        end
    end
end
