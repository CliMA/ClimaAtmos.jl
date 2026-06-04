#####
##### Smagorinsky Lilly Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore: Geometry

"""
    lilly_stratification_correction(Y, p, ᶜS)

Return a lazy broadcast of the Lilly stratification correction factor
`fb = (1 - Ri/Pr_t)^(1/4)` for `0 < Ri < Pr_t`, and `fb = 1` for `Ri ≤ 0`, where
`Ri = N²/|S|²` is the local gradient Richardson number [-].

The buoyancy frequency `N²` is computed from the virtual potential temperature, and `|S|`
is the vertical (`WAxis`) strain rate norm. The vertical gradient of `θ_v` is materialized
into `p.scratch.ᶜtemp_scalar`, so the returned broadcast must be consumed before that
scratch field is reused.

# Arguments

  - `Y`: The model state.
  - `p`: The simulation cache (`AtmosCache`); reads `ᶜT`, `ᶜq_tot_nonneg`, `ᶜq_liq`, and
    `ᶜq_ice` from `p.precomputed`.
  - `ᶜS`: The cell-center strain rate tensor [1/s].

Called from `set_smagorinsky_lilly_precomputed_quantities!`.
"""
function lilly_stratification_correction(Y, p, ᶜS)
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; ᶜtemp_scalar) = p.scratch
    grav = CAP.grav(p.params)
    Pr_t = CAP.Prandtl_number_0(CAP.turbconv_params(p.params))
    thermo_params = CAP.thermodynamics_params(p.params)
    FT = eltype(Pr_t)
    # Stratification correction
    ᶜθ_v = @. lazy(
        TD.virtual_pottemp(
            thermo_params,
            ᶜT,
            Y.c.ρ,
            ᶜq_tot_nonneg,
            ᶜq_liq,
            ᶜq_ice,
        ),
    )
    ᶜ∇ᵥθ = @. ᶜtemp_scalar = Geometry.WVector(ᶜgradᵥ(ᶠinterp(ᶜθ_v))).components.data.:1
    ᶜN² = @. lazy(grav / ᶜθ_v * ᶜ∇ᵥθ)
    ᶜS_norm = strain_rate_norm(ᶜS, Geometry.WAxis())

    ᶜRi = @. lazy(ᶜN² / (ᶜS_norm^2 + eps(FT)))  # Ri = N² / |S|²
    ᶜfb = @. lazy(ifelse(ᶜRi ≤ 0, FT(1), sqrt(sqrt(max(0, 1 - ᶜRi / Pr_t)))))
end

"""
    set_smagorinsky_lilly_precomputed_quantities!(Y, p, model)

Compute the Smagorinsky-Lilly quantities needed by the subgrid-scale diffusive tendencies
and store them in `p.precomputed`; return `nothing`.

The eddy viscosity is `νₜ = L² |S|` and the turbulent diffusivity is `D = νₜ / Pr_t`,
where `Pr_t` is the turbulent Prandtl number at neutral stratification. For `axes = :UVW`
a single isotropic mixing length `L = c_smag ∛(Δx Δy Δz) fb` is used; otherwise the
horizontal length is `c_smag Δx` and the vertical length is `c_smag Δz fb`, with `fb` the
Lilly stratification correction from `lilly_stratification_correction`.

Mutates the following fields of `p.precomputed`:

  - `ᶜS`, `ᶠS`: strain rate tensor on centers and faces [1/s].
  - `ᶜS_norm_h`, `ᶜS_norm_v`: horizontal and vertical strain rate norms on centers [1/s].
  - `ᶜνₜ_h`, `ᶜνₜ_v`: horizontal and vertical eddy viscosities on centers [m²/s].
  - `ᶜD_h`, `ᶜD_v`: horizontal and vertical eddy diffusivities on centers [m²/s].

The `::Nothing` method is a no-op for runs without a Smagorinsky-Lilly model.

# Arguments

  - `Y`: The model state.
  - `p`: The simulation cache (`AtmosCache`).
  - `model`: The `SmagorinskyLilly` model instance (or `nothing`).

See also `horizontal_smagorinsky_lilly_tendency!` and
`vertical_smagorinsky_lilly_tendency!`, which consume these quantities.
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

    # ᶜfb = lilly_stratification_correction(Y, p, ᶜS)
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

"""
    horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, model)

Add the horizontal Smagorinsky-Lilly subgrid-scale flux divergences to `Yₜ` in place;
return `nothing`.

Momentum receives `-∇ₕ·(ρ τ)/ρ` with the SGS momentum flux tensor `τ = -2 νₜ_h S`; total
energy receives `+∇ₕ·(ρ D_h ∇ₕh_tot)`; each grid-scale tracer `χ` receives
`+∇ₕ·(ρ D_h ∇ₕχ)`, and the `ρq_tot` diffusion is also added to `Yₜ.c.ρ` so that moisture
diffusion conserves mass. Reads `ᶜS`, `ᶠS`, `ᶜνₜ_h`, `ᶜD_h`, and `ᶜh_tot` from
`p.precomputed` (set by `set_smagorinsky_lilly_precomputed_quantities!`).

This tendency is always applied explicitly. It is a no-op unless the model's axes include
the horizontal directions (`is_smagorinsky_horizontal`); the `::Nothing` method is a
no-op. See also `vertical_smagorinsky_lilly_tendency!`.
"""
function horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, model::SmagorinskyLilly)
    is_smagorinsky_horizontal(model) || return nothing
    (; ᶜS, ᶠS, ᶜνₜ_h, ᶜD_h) = p.precomputed
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
    (; ᶜh_tot) = p.precomputed
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

"""
    vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, model)

Add the vertical Smagorinsky-Lilly subgrid-scale flux divergences to `Yₜ` in place;
return `nothing`.

Momentum receives `-∇ᵥ·(ρ τ)/ρ` with the SGS momentum flux tensor `τ = -2 νₜ_v S`; total
energy and each grid-scale tracer receive the vertical diffusive-flux divergence
`ᶜdiffusive_flux_divergenceᵥ` with face diffusivity `ᶠρ D_v` (subtracted, since it is a
flux divergence), and the `ρq_tot` diffusion is also applied to `Yₜ.c.ρ` so that moisture
diffusion conserves mass. Reads `ᶜS`, `ᶠS`, `ᶜνₜ_v`, and `ᶜh_tot` from `p.precomputed`.

This tendency is always applied explicitly, including the vertical diffusion; it is not
part of the implicit solver. It is a no-op unless the model's axes include the vertical
direction (`is_smagorinsky_vertical`); the `::Nothing` method is a no-op. See also
`horizontal_smagorinsky_lilly_tendency!`.
"""
function vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, model::SmagorinskyLilly)
    is_smagorinsky_vertical(model) || return nothing
    (; ᶜS, ᶠS, ᶜνₜ_v) = p.precomputed
    (; ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW, ᶠtemp_scalar, ᶠtemp_scalar_2) = p.scratch
    Pr_t = CAP.Prandtl_number_0(CAP.turbconv_params(p.params))
    ᶜρ = Y.c.ρ
    ᶠρ = @. ᶠtemp_scalar = ᶠinterp(ᶜρ)

    # Subgrid-scale momentum flux tensor, `τ = -2 νₜ ∘ S`
    ᶠνₜ_v = @. lazy(ᶠinterp(ᶜνₜ_v))
    ᶜτ_smag = @. ᶜtemp_UVWxUVW = -2 * ᶜνₜ_v * ᶜS
    ᶠτ_smag = @. ᶠtemp_UVWxUVW = -2 * ᶠνₜ_v * ᶠS

    # Turbulent diffusivity
    ᶠD_smag = @. lazy(ᶠνₜ_v / Pr_t)
    ᶠρD = @. lazy(ᶠρ * ᶠD_smag)

    # Apply to tendencies
    ## Horizontal momentum tendency
    @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(ᶠρ * ᶠτ_smag) / ᶜρ)
    ## Vertical momentum tendency
    @. Yₜ.f.u₃ -= C3(ᶠdiffdivᵥ_u₃(ᶜρ * ᶜτ_smag) / ᶠρ)

    ## Total energy tendency
    (; ᶜh_tot) = p.precomputed
    ᶜ∇ᵥρD∇h_totₜ = ᶜdiffusive_flux_divergenceᵥ(ᶠρD, ᶜh_tot)
    @. Yₜ.c.ρe_tot -= ᶜ∇ᵥρD∇h_totₜ

    ## Tracer diffusion and associated mass changes
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, ᶜρ))
        ᶜ∇ᵥρD∇χₜ = ᶜdiffusive_flux_divergenceᵥ(ᶠρD, ᶜχ)
        @. ᶜρχₜ -= ᶜ∇ᵥρD∇χₜ
        # Rain and snow does not affect the mass
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ -= ᶜ∇ᵥρD∇χₜ
        end
    end
end
