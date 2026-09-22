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
[PROPHET: Closures](prophet_closures.md) for the TKE-based eddy diffusivity,
and the two prescribed profiles below.

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

**Energy** diffuses as a two-piece enthalpy flux rather than as a lumped total
enthalpy,

```math
\boldsymbol{\mathcal{F}}_E = -\rho K_h \left[ \nabla_v s_d
  + (h_{\mathrm{eff}} + \Phi) \nabla_v q_t^{\mathrm{eff}} \right] ,
```

with ``s_d = h_d + \Phi`` the dry static energy and ``h_{\mathrm{eff}}`` the
mass-weighted specific enthalpy of the water that actually diffuses. The dry
term applies in every configuration, the water term only when ``\rho q_t`` is
prognostic. [Thermodynamics and the Working Fluid](thermodynamics.md) explains
why the split is needed: diffusing a lumped ``h_{tot}`` is not invariant to the
reference temperature ``T_0``, so it does not conserve energy and water
together.

**Water** diffuses through the effective total water
``q_t^{\mathrm{eff}} = q_t - q_{rai} - q_{sno}``. The same flux increments both
``\rho q_t`` and ``\rho``, so diffusing water carries the corresponding
moist-air mass. Cloud liquid and cloud ice have no flux of their own: each takes
a share of the aggregate tendency, scaled by the clipped ratio
``\min(q_\mu / q_t^{\mathrm{eff}}, 1)``, and the matching number densities scale
with it, which preserves the mean particle mass.

**Rain, snow, and rain number density** are exempt. Diffusing them would smear
features such as rain shafts, which evolve through sedimentation and
microphysics; see
[Microphysics](microphysics.md).

**Passive grid-mean tracers** diffuse independently with the full ``K_h``.

**Momentum** receives ``\nabla \cdot \boldsymbol{\tau} / \rho`` with the stress
``\boldsymbol{\tau} = 2 \rho K_u \boldsymbol{S}`` and ``\boldsymbol{S}`` the
vertical strain rate, under zero-flux boundary conditions. The surface stress is
not applied here; it is added by the separate surface-flux tendency, described
in [Surface Conditions](surface_conditions.md).

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

The species treatment differs from the vertical closures above.
Smagorinsky–Lilly and the constant horizontal diffusion apply a single scalar
diffusivity to all grid-scale tracers, precipitation included, while
anisotropic minimum dissipation forms a diffusivity per scalar.

## Where this is implemented

| Concept                                | Source                                                                                                                                                 |
|:-------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Prescribed vertical diffusion          | [vertical_diffusion_boundary_layer.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/vertical_diffusion_boundary_layer.jl) |
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
