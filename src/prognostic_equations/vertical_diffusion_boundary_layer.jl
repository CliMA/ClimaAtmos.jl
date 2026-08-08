#####
##### Vertical diffusion boundary layer parameterization
#####

import ClimaCore.Geometry: ⊗

"""
    vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, vert_diff_model)

Computes and applies tendencies due to vertical turbulent diffusion,
representing mixing processes within the planetary boundary layer and free atmosphere.

This function is dispatched based on the type of the vertical diffusion model
(`vert_diff_model`), which is accessed via `p.atmos.vertical_diffusion`.

**Dispatch details:**

1.  **`vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)`**:
    This is the main entry point, which internally calls the more specific method
    using `p.atmos.vertical_diffusion` to determine the diffusion model.

2.  **`vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, ::Nothing)`**:
    If the `vert_diff_model` is `Nothing` (i.e., vertical diffusion is turned off
    in the simulation configuration), this method is called and performs no operations.

3.  **`vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, ::Union{VerticalDiffusion, DecayWithHeightDiffusion})`**:
    This method implements the core logic for K-theory based vertical diffusion when
    a `VerticalDiffusion` or `DecayWithHeightDiffusion` model is active.
    It calculates tendencies for:
    - **Momentum (`uₕ`)**: Based on the divergence of a stress tensor,
      `τ = 2 ρ K_u S`, where `K_u` is the eddy viscosity
      and `S` is the strain rate tensor. The tendency is applied as
      `1/ρ ∇ ⋅ τ`. Default zero-flux boundary
      conditions are assumed for this diffusive term, as surface stresses
      are often handled by `surface_flux_tendency!`.
    - **Total Energy (`ρe_tot`)**: Divergence of a single-gradient enthalpy flux
      `F_E = - ρ K_h [∇_v s_d + (h_eff + Φ) ∇_v q_tot_eff]`, where
      `s_d = h_d + Φ` is the dry static energy, `q_tot_eff = q_tot - q_rai - q_sno`
      is the aggregate water that diffuses (rain/snow excluded), and
      `h_eff = (h_v q_v + h_l q_lcl + h_i q_icl) / max(q_water_nonneg, ε)` is the
      clipped-input mass-weighted enthalpy of the diffusing water. Zero-flux
      boundary conditions.
    - **Tracers**: Total water diffuses on `q_tot_eff` with flux
      `F = - ρ K_h ∇_v q_tot_eff`, applied to `ρq_tot` and (with the same sign)
      to `ρ` for moist-air mass conservation. Cloud mass species (`ρq_lcl`,
      `ρq_icl`) inherit their share by pure tendency scaling with the clipped
      ratio `min(q_μ / q_tot_eff, 1)`; their corresponding number densities
      scale proportionally to preserve mean particle mass. Rain, snow, and
      rain number density (`ρq_rai`, `ρq_sno`, `ρn_rai`) receive no diffusion.
      Passive (non-microphysics) tracers diffuse independently with full `K_h`.

This function is acting as a wrapper around the specific implementations
for different turbulence and convection models.

The primary role of this function is to dispatch to the correct turbulence model's
tendency function. It operates on the state `Y` and its tendency `Yₜ`, using
the model-specific cache `p`.

Arguments:
- `Yₜ`: The tendency state vector.
- `Y`: The current state vector.
- `p`: Cache containing parameters, atmospheric model configurations
       (like `p.atmos.vertical_diffusion`), precomputed thermodynamic
       quantities, and scratch space.
- `t`: Current simulation time (not directly used in diffusion calculations).
- `vert_diff_model` (for dispatched methods): The specific vertical diffusion model instance.

Modifies components of tendency vector `Yₜ.c` (e.g., `Yₜ.c.uₕ`, `Yₜ.c.ρe_tot`, `Yₜ.c.ρ`, and
various tracer fields such as `Yₜ.c.ρq_tot`).
"""

vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t) =
    vertical_diffusion_boundary_layer_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.vertical_diffusion,
    )

vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function vertical_diffusion_boundary_layer_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::Union{VerticalDiffusion, DecayWithHeightDiffusion},
)
    FT = eltype(Y)
    ϵ_FT = eps(FT)
    (; vertical_diffusion) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜu, ᶜp, ᶜT, ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg) = p.precomputed
    ᶜK_h = p.scratch.ᶜtemp_scalar
    if vertical_diffusion isa DecayWithHeightDiffusion
        ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(Y.c.ρ, vertical_diffusion)
    elseif vertical_diffusion isa VerticalDiffusion
        ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(
            Y.c.uₕ,
            ᶜp,
            vertical_diffusion,
        )
    end

    # Face diffusivities use a harmonic mean (reciprocal of interpolated
    # reciprocal), so the diffusive flux collapses at faces separating a
    # turbulent layer from quiescent, strongly stratified air (e.g., a
    # capping inversion), where arithmetic averaging would assign ≈ K/2.
    ᶠρK = @. lazy(ᶠinterp(Y.c.ρ) / ᶠinterp(1 / max(ᶜK_h, ϵ_FT)))
    if !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu)
        @. Yₜ.c.uₕ -= C12(
            ᶜdivᵥ(-2 * ᶠρK * ᶠstrain_rate) / Y.c.ρ,
        ) # assumes ᶜK_u = ᶜK_h
    end

    # Enthalpy diffusion. Dry static energy piece applies in all
    # configurations (including dry); the water enthalpy piece is added
    # below when ρq_tot is prognostic.
    (; ᶜΦ) = p.core
    @. Yₜ.c.ρe_tot -=
        ᶜdiffdivᵥ(-(ᶠρK * ᶠgradᵥ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ))))

    # Water diffusion on q_tot_eff = q_tot - q_rai - q_sno (rain/snow
    # excluded). Cloud species (lcl, icl) inherit their share of the
    # aggregate q_tot diffusion via clipped ratio; rain, snow, and n_rai
    # do not diffuse. Enthalpy water contribution uses h_eff-weighted
    # single-gradient form.
    if !(p.atmos.microphysics_model isa DryModel)
        ᶜq_tot_eff =
            p.atmos.microphysics_model isa
            Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ?
            (@. lazy(specific(Y.c.ρq_tot - Y.c.ρq_rai - Y.c.ρq_sno, Y.c.ρ))) :
            (@. lazy(specific(Y.c.ρq_tot, Y.c.ρ)))
        ᶜρq_tot_diff = p.scratch.ᶜtemp_scalar_2
        @. ᶜρq_tot_diff = ᶜdiffdivᵥ(-(ᶠρK * ᶠgradᵥ(ᶜq_tot_eff)))
        @. Yₜ.c.ρq_tot -= ᶜρq_tot_diff
        @. Yₜ.c.ρ -= ᶜρq_tot_diff

        # Water enthalpy contribution: -ρK·(h_eff+Φ)·∇q_tot_eff.
        ᶜq_vap = @. lazy(TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
        ᶜq_lcl, ᶜq_icl =
            p.atmos.microphysics_model isa
            Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ?
            (
                (@. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))),
                (@. lazy(specific(Y.c.ρq_icl, Y.c.ρ))),
            ) :
            (ᶜq_liq, ᶜq_ice)
        ᶜh_eff_plus_Φ = p.scratch.ᶜtemp_scalar_3
        @. ᶜh_eff_plus_Φ =
            (
                TD.enthalpy_vapor(thermo_params, ᶜT) * max(FT(0), ᶜq_vap) +
                TD.enthalpy_liquid(thermo_params, ᶜT) * max(FT(0), ᶜq_lcl) +
                TD.enthalpy_ice(thermo_params, ᶜT) * max(FT(0), ᶜq_icl)
            ) /
            max(max(FT(0), ᶜq_vap) + max(FT(0), ᶜq_lcl) + max(FT(0), ᶜq_icl), ϵ_FT) + ᶜΦ
        @. Yₜ.c.ρe_tot -=
            ᶜdiffdivᵥ(-(ᶠρK * ᶠinterp(ᶜh_eff_plus_Φ) * ᶠgradᵥ(ᶜq_tot_eff)))

        # Distribute ρq_tot_diff to cloud mass (and number) species.
        ᶜratio = p.scratch.ᶜtemp_scalar_4
        for (ρq_name, ρn_name) in
            ((@name(c.ρq_lcl), @name(c.ρn_lcl)), (@name(c.ρq_icl), @name(c.ρn_icl)))
            MatrixFields.has_field(Y, ρq_name) || continue
            ᶜρq = MatrixFields.get_field(Y, ρq_name)
            ᶜρqₜ = MatrixFields.get_field(Yₜ, ρq_name)
            @. ᶜratio =
                max(FT(0), min(FT(1), specific(ᶜρq, Y.c.ρ) / max(ᶜq_tot_eff, ϵ_FT)))
            @. ᶜρqₜ -= ᶜratio * ᶜρq_tot_diff
            if MatrixFields.has_field(Y, ρn_name)
                ᶜρn = MatrixFields.get_field(Y, ρn_name)
                ᶜρnₜ = MatrixFields.get_field(Yₜ, ρn_name)
                @. ᶜρnₜ -= ᶜratio * max(FT(0), ᶜρn) / max(ᶜρq, ϵ_FT) * ᶜρq_tot_diff
            end
        end
    end

    # Passive (non-microphysics) grid-scale tracers: independent diffusion at
    # full K_h. Skip microphysics species (handled above or no diffusion).
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ρχ_name in microphysics_tracer_names(Y) && return
        @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠρK * ᶠgradᵥ(specific(ᶜρχ, Y.c.ρ))))
    end
end
