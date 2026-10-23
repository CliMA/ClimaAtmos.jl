# Diffusion

Diffusion in ClimaAtmos represents physical subgrid-scale turbulent transport:
the vertical mixing of a boundary layer the grid does not resolve, and the
horizontal mixing an eddy-viscosity closure supplies. It is distinct from
[Hyperdiffusion](hyperdiffusion.md), which is a numerical filter with no
physical content, though both smooth gradients and both therefore reduce the
spurious oscillations and negative tracer values described in
[Microphysics](microphysics.md).

This page collects what the diffusive terms act on and how they are split
between the implicit and explicit halves of the timestep. The closures that
supply the diffusivities are documented elsewhere:
[Large-Eddy Simulation Closures](les_sgs.md) for the resolved-eddy closures,
[Turbulence kinetic energy and eddy diffusivity](prophet_closures.md#Turbulence-kinetic-energy-and-eddy-diffusivity)
for the TKE-based eddy diffusivity, and the two prescribed profiles below.

## Vertical diffusion

Two prescribed-diffusivity closures are available through the `vert_diff`
configuration key. Both are boundary-layer schemes: the diffusivity is a
function of height or pressure alone.

| `vert_diff`                  | Type                                          | Eddy diffusivity                                                              |
|:---------------------------- |:--------------------------------------------- |:----------------------------------------------------------------------------- |
| `~` (default)                | none                                          | No vertical diffusion                                                         |
| `"VerticalDiffusion"`        | [`ClimaAtmos.VerticalDiffusion`](@ref)        | ``K_E = C_E \|\boldsymbol{u}_a\| z_a`` below the 850 hPa level, tapered above |
| `"DecayWithHeightDiffusion"` | [`ClimaAtmos.DecayWithHeightDiffusion`](@ref) | ``K = D_0 \exp[-(z - z_s) / H]``                                              |

Here ``\|\boldsymbol{u}_a\|`` and ``z_a`` are the wind speed and height of the
lowest model level, and ``C_E`` is a dimensionless coefficient. Where the
pressure falls below 850 hPa, above the boundary layer, the surface-driven
diffusivity decays as

```math
K_E \exp \left[ - \left( \frac{p_{pbl} - p}{p_{strato}} \right)^2 \right] ,
\qquad p_{pbl} = 850 \, \mathrm{hPa} , \quad p_{strato} = 100 \, \mathrm{hPa} ,
```

which confines the mixing to the boundary layer and the lower free
troposphere. The same value serves as the eddy viscosity for momentum;
`disable_momentum_vertical_diffusion` restricts either closure to the scalars.

Face diffusivities are formed as a harmonic mean of the two neighboring center
values rather than an arithmetic one. The flux then collapses at a face
separating a turbulent layer from quiescent, strongly stratified air, where an
arithmetic mean would leave about ``K/2``.

Vertical diffusion can instead come from [PROPHET](prophet.md), whose diffusive
flux is closed by a prognostic turbulence kinetic energy, or from the
[LES closures](les_sgs.md): a vertically acting Smagorinsky–Lilly, or AMD,
which always acts on both axes. These paths are alternatives, and a
configuration that sets `vert_diff` alongside `turbconv`, `amd_les`, or a
vertically acting `smagorinsky_lilly` is rejected at model construction, as is
`edmfx_sgs_horizontal_diffusive_flux` alongside an LES closure in the
horizontal.

### What each variable receives

**Energy** diffuses through the enthalpy flux of [Yatunin2026](@cite) (Eqs. 30
and 33),

```math
\boldsymbol{\mathcal{F}}_E = -\rho K_h \left[ \nabla_v s_d
  + (h_v + \Phi) \nabla_v q_t^{\mathrm{eff}}
  + (h_l - h_v) \nabla_v q_l^{\mathrm{cl}}
  + (h_i - h_v) \nabla_v q_i^{\mathrm{cl}} \right] ,
```

with ``s_d = h_d + \Phi`` the dry static energy, ``q_t^{\mathrm{eff}} = q_t - q_{\mathrm{rai}} - q_{\mathrm{sno}}``
the suspended water that diffuses, ``q_l^{\mathrm{cl}}`` and
``q_i^{\mathrm{cl}}`` the suspended cloud liquid and cloud ice specific
humidities, and ``h_\mu(T)`` the specific enthalpy of water phase
``\mu \in \{v, l, i\}``. The dry term applies in every configuration, the water
terms when ``\rho q_t`` is prognostic.

Three physical and mathematical properties determine this form:

  - **Energy–species consistency.** In the dry-air reference frame
    [Romps2008](@cite), the subgrid-scale flux of dry air is zero and each
    diffusing water phase ``\mu \in \{v, l, i\}`` carries its total specific
    enthalpy ``h_{\mathrm{tot},\mu} = h_\mu(T) + \Phi``, so the water-enthalpy
    flux is ``\sum_\mu (h_\mu + \Phi) \boldsymbol{\mathcal{F}}_{q_\mu}``
    ([Yatunin2026](@cite), Eq. 30). Eliminating the vapor specific humidity via
    ``q_v = q_t^{\mathrm{eff}} - q_l^{\mathrm{cl}} - q_i^{\mathrm{cl}}`` and
    closing each suspended phase flux down its own gradient,
    ``\boldsymbol{\mathcal{F}}_{q_\mu} = -\rho K_h \nabla_v q_\mu``, yields the
    bracket above: ``(h_v + \Phi) \nabla_v q_t^{\mathrm{eff}}`` is the enthalpy
    carried by the net suspended-water flux, and
    ``(h_l - h_v) \nabla_v q_l^{\mathrm{cl}} = -L_v(T) \nabla_v q_l^{\mathrm{cl}}``
    and
    ``(h_i - h_v) \nabla_v q_i^{\mathrm{cl}} = -L_s(T) \nabla_v q_i^{\mathrm{cl}}``
    carry the latent enthalpy of condensate diffusing one way and vapor the
    other at fixed ``q_t^{\mathrm{eff}}``. Each thermodynamic coefficient is
    evaluated at cell centers and interpolated to faces outside the gradient so
    that the flux factors into ``\sum_\mu h_{\mathrm{tot},\mu} \boldsymbol{\mathcal{F}}_{q_\mu}``.
    Diffusing total moist enthalpy ``-\rho K_h \nabla_v (h + \Phi)`` instead
    would differentiate ``-h_d q_t^{\mathrm{eff}}`` as if diffusing water
    displaced dry air rather than adding moist mass
    ``\partial_t \rho = -\nabla_v \cdot (\rho \boldsymbol{\mathcal{F}}_{q_t})``,
    and would differentiate ``h_\mu(T)`` inside the gradient.
  - **Moist-adiabatic neutrality.** Turbulent eddies exchange parcels that
    conserve ``q_t`` and moist static energy
    ``h + \Phi = s_d + q_t (h_v - h_d) - L_v(T) q_l^{\mathrm{cl}} - L_s(T) q_i^{\mathrm{cl}}``
    during reversible adiabatic displacements. In a saturated cloud layer that
    is well mixed in ``(h + \Phi, q_t)``, temperature decreases with height while
    condensate increases with height so that
    ``\nabla_v s_d \approx L_v \nabla_v q_l^{\mathrm{cl}} + L_s \nabla_v q_i^{\mathrm{cl}} > 0``.
    The phase-exchange terms ``(h_l - h_v) \nabla_v q_l^{\mathrm{cl}}`` and
    ``(h_i - h_v) \nabla_v q_i^{\mathrm{cl}}`` cancel ``\nabla_v s_d`` up to the
    dry-air-frame specific-heat residual ``-(c_{pm} - c_{pd}) \nabla_v T``
    (about ``1\%`` of ``\nabla_v s_d``), leaving a moist-adiabatic cloud layer
    near diffusive equilibrium. Weighting ``\nabla_v q_t^{\mathrm{eff}}`` by a
    single phase-averaged enthalpy ``h_{\mathrm{eff}} + \Phi`` (as in
    fourth-order [Hyperdiffusion](hyperdiffusion.md)) would omit those latent
    phase-exchange fluxes when condensate and vapor counter-diffuse at uniform
    ``q_t^{\mathrm{eff}}``.
  - **Reference-temperature invariance.** Under a shift
    ``T_0 \to T_0 + \delta T_0`` with Kirchhoff-consistent reference latent
    heats, all three water-phase enthalpies shift by the same constant
    ``-c_{pl} \delta T_0`` (see
    [Thermodynamics and the Working Fluid](thermodynamics.md)). The phase
    differences ``h_l - h_v`` and ``h_i - h_v`` are invariant, and the shift of
    ``(h_v + \Phi) \nabla_v q_t^{\mathrm{eff}}`` cancels the shift of
    ``\partial_t (\rho e_{\mathrm{tot}})`` through the moist-mass continuity
    source ``\partial_t \rho = \partial_t (\rho q_t)|_{\mathrm{diff}}``
    (whereas ``-\rho K_h \nabla_v (h + \Phi)`` shifts by
    ``(c_{pd} - c_{pl}) \delta T_0 \nabla_v q_t^{\mathrm{eff}}``, leaving an
    uncompensated ``c_{pd} \delta T_0 \, \partial_t (\rho q_t)|_{\mathrm{diff}}``
    residual).

**Water** diffuses through the effective total water
``q_t^{\mathrm{eff}} = q_t - q_{\mathrm{rai}} - q_{\mathrm{sno}}``. The same
flux increments both ``\rho q_t`` and ``\rho``, so diffusing water carries the
corresponding moist-air mass. In the non-equilibrium schemes (`1M`, `2M`, and
`2MP3`), the prognostic suspended cloud species (`ρq_lcl`, `ρq_icl`, `ρn_lcl`,
`ρn_ice`, and the rime mass `ρq_rim` and rime volume `ρb_rim` that P3 cloud ice
carries) diffuse down their own specific gradients with the same ``\rho K_h``,
matching the species fluxes ``\boldsymbol{\mathcal{F}}_{q_l^{\mathrm{cl}}}`` and
``\boldsymbol{\mathcal{F}}_{q_i^{\mathrm{cl}}}`` whose latent enthalpy the
energy equation carries (`add_moisture_diffusionᵥ!`). Because the rime fields
diffuse with the same linear operator as cloud ice, the rime mass stays within
the ice mass. In the equilibrium 0-moment scheme, ``q_l^{\mathrm{cl}}`` and ``q_i^{\mathrm{cl}}`` are
diagnostic (`ᶜq_liq` and `ᶜq_ice`) and have no separate prognostic equations.

**Rain, snow, and rain number density** do not diffuse under the prescribed and
PROPHET closures. They evolve through sedimentation and microphysical sources;
see [Microphysics](microphysics.md). The LES closures, whose subgrid eddies carry
falling precipitation, diffuse them as well (see below).

**Passive grid-mean tracers** diffuse independently with the full ``K_h``.

**Momentum** receives ``\nabla \cdot \boldsymbol{\tau} / \rho`` with the stress
``\boldsymbol{\tau} = 2 \rho K_u \boldsymbol{S}`` and ``\boldsymbol{S}`` the
vertical strain rate, under zero-flux boundary conditions. The surface stress is
added by the separate surface-flux tendency, described in
[Surface Conditions](surface_conditions.md).

### Implicit or explicit

Vertical diffusion is stiff at the timesteps of interest, so
`implicit_diffusion: true` moves the whole tendency into the implicit solve. The
model's `diff_mode` follows that key: under `Implicit()` the tendency is
evaluated from `implicit_tendency!` and linearized in the Jacobian; under
`Explicit()` it is evaluated from `additional_tendency!`. The global
configurations with a 60 km model top set `implicit_diffusion: true`, as does
every PROPHET configuration except the advection test, and those pair it with
`approximate_linear_solve_iters: 2`. See [Implicit Solver](implicit_solver.md).

## Horizontal diffusion

Horizontal diffusion comes from an eddy-viscosity closure, never from a
prescribed profile: Smagorinsky–Lilly [Sridhar2022](@cite), anisotropic minimum
dissipation [Akbar2016](@cite), a constant horizontal diffusivity, or the
optional PROPHET horizontal diffusive flux. All of them are explicit. See
[Large-Eddy Simulation Closures](les_sgs.md) and
[PROPHET Horizontal Diffusion](prophet_horizontal_diffusion.md).

All four horizontal closures (as well as the second-order horizontal
[viscous sponge](sponge.md)) use the same water-enthalpy decomposition as the
vertical closures (`add_water_enthalpy_diffusionₕ!` and
`add_moisture_diffusionₕ!`). PROPHET horizontal diffusion and the viscous sponge
diffuse ``q_t^{\mathrm{eff}}`` and the suspended cloud species down their own
horizontal gradients while rain and snow do not diffuse. The LES closures,
horizontal and vertical (Smagorinsky–Lilly, anisotropic minimum dissipation, and
constant horizontal diffusion), diffuse the total water ``q_t`` and every
microphysics species, rain, snow, and rain number included, down its own
gradient; in the enthalpy bracket, rain enters with the liquid water and snow
with the ice, ``q_l = q_l^{\mathrm{cl}} + q_{\mathrm{rai}}`` and
``q_i = q_i^{\mathrm{cl}} + q_{\mathrm{sno}}`` (`ᶜdiffusing_water_phases` with
`precipitation = true`). Smagorinsky–Lilly, constant horizontal diffusion,
PROPHET horizontal diffusion, and the viscous sponge use a single scalar
diffusivity for ``s_d`` and all water terms, while anisotropic minimum
dissipation evaluates ``D(\nabla_h s_d)`` for the dry static energy term and
``D(\nabla_h q_t)`` for the water terms and microphysics species so that each
species flux and its enthalpy use the same diffusivity.

## Where this is implemented

| Concept                                | Source                                                                                                                                                 |
|:-------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Prescribed vertical diffusion          | [vertical_diffusion_boundary_layer.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/vertical_diffusion_boundary_layer.jl) |
| Shared enthalpy and species helpers    | [eddy_diffusion_closures.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/eddy_diffusion_closures.jl)                     |
| Eddy-diffusivity profiles              | [src/types.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/types.jl)                                                                          |
| LES closures (horizontal and vertical) | [les_sgs_models/](https://github.com/CliMA/ClimaAtmos.jl/tree/main/src/parameterized_tendencies/les_sgs_models)                                        |
| PROPHET diffusive fluxes               | [edmfx_sgs_flux.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_sgs_flux.jl)                                       |
| Implicit/explicit dispatch             | [implicit/implicit_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/implicit_tendency.jl)               |
| Surface-flux tendency                  | [src/prognostic_equations/surface_flux.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/surface_flux.jl)                  |
| Model types                            | [`ClimaAtmos.AbstractVerticalDiffusion`](@ref), [`ClimaAtmos.VerticalDiffusion`](@ref), [`ClimaAtmos.DecayWithHeightDiffusion`](@ref)                  |

The keys named here are `vert_diff` and `implicit_diffusion`; the horizontal
closures are selected with `smagorinsky_lilly`, `amd_les`,
`constant_horizontal_diffusion`, and `edmfx_sgs_horizontal_diffusive_flux`. See
[Configuration Options](configuration_options.md).
