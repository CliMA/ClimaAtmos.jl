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
    - **Total Energy (`ρe_tot`)**: Based on the divergence of an enthalpy flux,
      `F_E = - ρ K_h ∇_v h_{tot}`, where `K_h` is the eddy diffusivity for
      heat and `h_{tot}` is the specific total enthalpy. Zero-flux boundary
      conditions are explicitly applied at the top and bottom for this term.
    - **Tracers (e.g., `ρq_tot`, `ρq_liq`)**: Based on the divergence of tracer fluxes,
      `F_χ = - ρ K_{h,scaled} ∇_v χ`, where `χ` is the specific
      tracer quantity and `K_{h,scaled}` is the (potentially scaled for certain
      tracers like rain and snow using `α_vert_diff_tracer`) eddy diffusivity
      for scalars. Zero-flux boundary conditions are explicitly applied.
    - **Note on mass conservation for `q_tot` diffusion**: The current implementation
      also modifies the tendency of total moist air density `Yₜ.c.ρ` based on the
      diffusion tendency of total specific humidity `ρq_tot`:
      `Yₜ.c.ρ -= ᶜρχₜ_diffusion_for_q_tot`.

Arguments for all methods:
- `Yₜ`: The tendency state vector, modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters (e.g., `p.params` for `CAP.α_vert_diff_tracer`),
       precomputed fields (e.g., `ᶜK_u`, `ᶜK_h`, `ᶜh_tot`, `ᶜspecific` tracer values),
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
    α_vert_diff_tracer = CAP.α_vert_diff_tracer(p.params)
    (; ᶜh_tot, ᶜspecific, ᶜK_u, ᶜK_h) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    if !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        ᶠstrain_rate = p.scratch.ᶠtemp_UVWxUVW
        ᶠstrain_rate = compute_strain_rate_face(ᶜu_lazy(Y.c.uₕ, Y.f.u₃))
        @. Yₜ.c.uₕ -= C12(
            ᶜdivᵥ(-2 * ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_u) * ᶠstrain_rate) / Y.c.ρ,
        )
    end

    ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(0)),
        bottom = Operators.SetValue(C3(0)),
    )
    @. Yₜ.c.ρe_tot -=
        ᶜdivᵥ_ρe_tot(-(ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜh_tot)))

    ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
    ᶜK_h_scaled = p.scratch.ᶜtemp_scalar_2
    ᶜdivᵥ_ρχ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(0)),
        bottom = Operators.SetValue(C3(0)),
    )
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        if ρχ_name in (@name(ρq_rai), @name(ρq_sno), @name(ρn_rai))
            @. ᶜK_h_scaled = α_vert_diff_tracer * ᶜK_h
        else
            @. ᶜK_h_scaled = ᶜK_h
        end
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
