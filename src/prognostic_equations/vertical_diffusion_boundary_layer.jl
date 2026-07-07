#####
##### Vertical diffusion boundary layer parameterization
#####

import ClimaCore.Geometry: ⊗
import ClimaCore.Operators as Operators

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
    - **Total Energy (`ρe_tot`)**: Based on the divergence of an enthalpy flux
      in dry-static-energy + water-enthalpy form,
      `F_E = - ρ K_h (∇_v s_d + Σ_μ h_tot,μ ∇_v q_μ)`, where `K_h` is the eddy
      diffusivity for heat, `s_d = h_d + Φ` is the dry static energy, and
      `h_tot,μ = h_μ + Φ` is the total enthalpy carried by water constituent
      `μ ∈ {vap, liq, ice}`. Zero-flux boundary conditions are explicitly
      applied at the top and bottom for this term.
    - **Tracers (e.g., `ρq_tot`, `ρq_lcl`)**: Based on the divergence of tracer fluxes,
      `F_χ = - ρ K_{h,scaled} ∇_v χ`, where `χ` is the specific
      tracer quantity and `K_{h,scaled}` is the (potentially scaled for certain
      tracers like rain and snow using `α_vert_diff_tracer`) eddy diffusivity
      for scalars. Zero-flux boundary conditions are explicitly applied.
    - **Note on mass conservation for `q_tot` diffusion**: The current implementation
      also modifies the tendency of total moist air density `Yₜ.c.ρ` based on the
      diffusion tendency of total specific humidity `ρq_tot`:
      `Yₜ.c.ρ -= ᶜρχₜ_diffusion_for_q_tot`.

This function is acting as a wrapper around the specific implementations
for different turbulence and convection models.

The primary role of this function is to dispatch to the correct turbulence model's
tendency function. It operates on the state `Y` and its tendency `Yₜ`, using
the model-specific cache `p`.

Arguments:
- `Yₜ`: The tendency state vector.
- `Y`: The current state vector.
- `p`: Cache containing parameters (e.g., `p.params` for `CAP.α_vert_diff_tracer`),
       atmospheric model configurations (like `p.atmos.vertical_diffusion`), and scratch space.
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
    (; vertical_diffusion) = p.atmos
    α_vert_diff_microphysics = CAP.α_vert_diff_tracer(p.params)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜu, ᶜp, ᶜT, ᶜq_liq, ᶜq_ice) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ
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

    if !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu)
        @. Yₜ.c.uₕ -= C12(
            ᶜdivᵥ(-2 * ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h) * ᶠstrain_rate) / Y.c.ρ,
        ) # assumes ᶜK_u = ᶜK_h
    end

    # Total enthalpy diffusion, using the dry-static-energy + water-enthalpy
    # decomposition F_h = -K_h ∇s_d + Σ_μ h_tot,μ (-K_h ∇q_μ); see the
    # matching term in `edmfx_sgs_diffusive_flux_tendency!` for details.
    # Note: F_qμ for liquid and ice uses unscaled K_h (omitting the tracer
    # vertical diffusion factor α_vert_diff_tracer) to maintain exact energetic
    # consistency with the unscaled ρq_tot diffusion equation, preserve total
    # water invariance, and align with the implicit solver's Jacobian.
    ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(0)),
        bottom = Operators.SetValue(C3(0)),
    )
    (; ᶜΦ) = p.core
    (; ᶜq_tot_nonneg) = p.precomputed
    ᶜq_vap = @. lazy(TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
    @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(
        -(
            ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h) *
            (
                ᶠgradᵥ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)) +
                ᶠinterp(TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ) *
                ᶠgradᵥ(ᶜq_vap) +
                ᶠinterp(TD.enthalpy_liquid(thermo_params, ᶜT) + ᶜΦ) *
                ᶠgradᵥ(ᶜq_liq) +
                ᶠinterp(TD.enthalpy_ice(thermo_params, ᶜT) + ᶜΦ) *
                ᶠgradᵥ(ᶜq_ice)
            )
        ),
    )

    ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar_2
    ᶜK_h_scaled = p.scratch.ᶜtemp_scalar_3

    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        if ρχ_name in (
            @name(ρq_lcl), @name(ρq_icl), @name(ρq_rai),
            @name(ρq_sno), @name(ρn_lcl), @name(ρn_rai)
        )
            @. ᶜK_h_scaled = α_vert_diff_microphysics * ᶜK_h
        else
            @. ᶜK_h_scaled = ᶜK_h
        end
        ᶜdivᵥ_ρχ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(0)),
            bottom = Operators.SetValue(C3(0)),
        )
        @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρχ(
            -(
                ᶠinterp(Y.c.ρ) *
                ᶠinterp(ᶜK_h_scaled) *
                ᶠgradᵥ(specific(ᶜρχ, Y.c.ρ))
            ),
        )
        @. ᶜρχₜ -= ᶜρχₜ_diffusion
        # Only add contribution from total water diffusion to mass tendency
        # (exclude contributions from diffusion of condensate, precipitation)
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion
        end
    end
end
