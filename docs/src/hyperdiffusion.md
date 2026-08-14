# Hyperdiffusion

Hyperdiffusion serves numerical rather than physical ends: it stabilizes the
model and damps the small-scale oscillations that a high-order horizontal
discretization admits. It acts along terrain-following coordinate surfaces, so
it depends on the model's notion of "horizontal"
[Jablonowski2011, Yatunin2026](@cite).

Even so, hyperdiffusion is written to respect the model's conservation laws to
floating-point precision. That requirement shapes its form, and is the main
subject of this page.

## Biharmonic form

For a generic scalar ``\psi``, the hyperdiffusive flux is proportional to the
third derivative of the field,

```math
\boldsymbol{\mathcal{H}}_\psi = \nu_\psi \nabla_h
  \left[ \nabla_h^2 (\psi - \psi_r) \right],
```

with a hyperdiffusion coefficient ``\nu_\psi`` and the horizontal del operator
``\nabla_h`` taken along terrain-following coordinate surfaces.

A vertically varying reference state ``\psi_r`` is subtracted from dry static
energy and total specific humidity. This reduces the spurious vertical mixing
that arises when diffusion acts along coordinate surfaces that warp over steep
topography [Herrington2022](@cite). The reference state is built from the
reference temperature profile ``T_r(\Pi)`` used for the pressure-gradient
force (see [Governing Equations](equations.md)): the reference dry static
energy is

```math
s_{d,r} = c_{pd} (T_r - T_0) + \Phi_r ,
```

with ``\Phi_r`` the associated reference geopotential, and the reference total
specific humidity is the specific humidity at a constant relative humidity of
50% along that profile,

```math
q_{t,r} = 0.5 \, q_v^*(T_r, p) ,
```

where ``q_v^*`` is the saturation specific humidity.

!!! note "TODO: not yet implemented"

    The reference-state subtraction is part of the formulation in
    [Yatunin2026](@cite) but is not yet in the code, which hyperdiffuses ``s_d``
    and the effective total water directly. The reference geopotential
    ``\Phi_r`` is used in the pressure-gradient force; see
    [Governing Equations](equations.md).

The tendency is applied as two successive Laplacians with the direct stiffness
summation operator applied in between. This removes the discontinuities across
element boundaries and preserves the discrete inner product, so the operation
remains globally conservative. In the code this is the split between
`prep_hyperdiffusion_tendency!`, which computes and DSSes the Laplacians into
cache fields, and `apply_hyperdiffusion_tendency!`, which takes the second
derivative.

## Water: partitioning to the suspended species

Hyperdiffusion is applied to the total specific humidity ``q_t``. How the
resulting tendency is distributed among the individual water species is a
physical question, not a numerical one.

Horizontal diffusion of precipitating species would smear out features such as
rain shafts, which evolve through vertical sedimentation and microphysics. The
hyperdiffusive tendency for total water therefore goes **only to the suspended
species** — water vapor, cloud liquid, and cloud ice — in proportion to their
share of the suspended total humidity ``q_t^{cl}``, and is zero for rain and
snow:

```math
-\frac{1}{\rho} \nabla_h \cdot (\rho \boldsymbol{\mathcal{H}}_{q_\mu^\sigma}) =
\begin{cases}
  \dfrac{q_\mu^\sigma}{q_t^{cl}}
    \left( -\dfrac{1}{\rho} \nabla_h \cdot
      (\rho \boldsymbol{\mathcal{H}}_{q_t}) \right)
    & \text{if } \sigma = \mathrm{cl} \text{ (cloud)}, \\[2ex]
  0 & \text{if } \sigma = \mathrm{pr} \text{ (precipitation)}.
\end{cases}
```

The water vapor tendency follows implicitly, since its specific humidity is
diagnosed from the total and the condensate specific humidities.

The implementation adds two details to the expression above. The share each
cloud species receives is **clipped** to the interval ``[0, 1]``, so a species'
specific humidity exceeding the suspended total through limiter or round-off
effects cannot produce a share above one. Where a two-moment scheme carries
number densities, those scale with their corresponding mass species; rain number
density, like rain and snow mass, receives no hyperdiffusion.

## Energy: an enthalpy-consistent flux

The total hyperdiffusive enthalpy flux is built from two components, in parallel
with the turbulent enthalpy flux described in
[Thermodynamics and the Working Fluid](thermodynamics.md). It is not the
diffusion of a lumped total enthalpy:

```math
\boldsymbol{\mathcal{H}}_h = \boldsymbol{\mathcal{H}}_{s_d}
  + h_{\mathrm{tot,cl}} \, \boldsymbol{\mathcal{H}}_{q_t}
= \nu_h \nabla_h \left[ \nabla_h^2 (s_d - s_{d,r}) \right]
  + h_{\mathrm{tot,cl}} \, \boldsymbol{\mathcal{H}}_{q_t} .
```

The first term hyperdiffuses the dry static energy ``s_d = h_d + \Phi``. The
second accounts for the enthalpy transported by the hyperdiffusive flux of total
water, weighted by the mass-weighted average specific total enthalpy of the
suspended water species,

```math
h_{\mathrm{tot,cl}} = \frac{q_v h_{\mathrm{tot},v}
  + q_l^{cl} h_{\mathrm{tot},l} + q_i^{cl} h_{\mathrm{tot},i}}{q_t^{cl}} .
```

Given the water partitioning above, this expression for the water-enthalpy flux
is exact. In the code these are the cache fields `ᶜ∇²s_d` and `ᶜ∇²q_tot_eff`,
with the weighting held in `ᶜh_eff_plus_Φ` (the average
``h_{\mathrm{tot,cl}}`` includes the geopotential); energy hyperdiffusion does
not act on a lumped `h_tot`.

The decomposition allows total energy and water to be conserved together.
Hyperdiffusing potential temperature or another intensive thermodynamic variable
instead creates sources and sinks of total energy that then need an energy fixer
to offset them. Here every hyperdiffusive term is a flux divergence along a
coordinate surface, so the operators move quantities around without adding to or
removing from the global total.

The hyperdiffusion coefficients for dry static energy and total specific
humidity are equal, which corresponds to a turbulent Lewis number of one.

## Momentum

Horizontal biharmonic hyperviscosity is applied to the three-dimensional
velocity, ``\boldsymbol{\mathcal{H}}_u = \nu_u \nabla_h (\Delta_h \boldsymbol{u})``, with the horizontal vector Laplacian

```math
\Delta_h \boldsymbol{u} = \nabla_h (\nabla_h \cdot \boldsymbol{u})
  - \nabla_h \times (\nabla_h \times \boldsymbol{u}).
```

A tensor divergence in non-orthogonal coordinates brings in curvature terms, so
the model uses the cheaper form common in atmosphere models,

```math
-\frac{1}{\rho} \nabla_h \cdot (\rho \boldsymbol{\mathcal{H}}_u)
\approx -\nu_u \left[ \delta_{div} \nabla_h (\nabla_h \cdot \Delta_h \boldsymbol{u})
  - \nabla \times (\nabla_h \times \Delta_h \boldsymbol{u}) \right].
```

With ``\delta_{div} = 1`` and constant ``\nu_u`` this amounts to neglecting
density variations along horizontal coordinate surfaces. Values ``\delta_{div} > 1`` increase the damping of two-dimensional divergent flow components, which
include gravity waves; the default is ``\delta_{div} = 5``, set by the
`divergence_damping_factor` configuration key.

## Coefficients

For numerical stability the hyperviscosity must grow with horizontal grid
spacing, typically like ``(\delta x)^3``
[Lauritzen2018, Skamarock2014](@cite). Both
coefficients in ClimaAtmos are set that way, from the mean nodal distance ``h``
of the horizontal grid:

```math
\nu_u = c_\nu h^3, \qquad
\nu_\psi = \frac{\nu_u}{\mathrm{Pr}_t} ,
```

where ``c_\nu`` is the `vorticity_hyperdiffusion_coefficient`, with default
0.1857 in units of m s⁻¹, and ``\mathrm{Pr}_t`` is the
`hyperdiffusion_prandtl_number`, with default 0.2. A Prandtl number below one
damps scalars more strongly than vorticity; with these defaults, five times as
strongly.

Setting `hyperdiff` to `CAM_SE` selects preset coefficients matching the
hyperviscosity of [Lauritzen2018](@cite) instead; setting it to `~` disables
hyperdiffusion entirely.

## Where this is implemented

| Concept                     | Source                                                                                                                                    |
|:--------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------- |
| Coefficients and cache      | [src/prognostic_equations/hyperdiffusion.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/hyperdiffusion.jl) |
| `Hyperdiffusion` model type | [`ClimaAtmos.Hyperdiffusion`](@ref)                                                                                                       |
| Operators used              | [Discretization and Operators](discretization.md)                                                                                         |
| Parameter values            | [ClimaParams.jl](https://github.com/CliMA/ClimaParams.jl)                                                                                 |

The relevant configuration keys are `hyperdiff`,
`vorticity_hyperdiffusion_coefficient`, `hyperdiffusion_prandtl_number`, and
`divergence_damping_factor`; see
[Configuration Options](configuration_options.md).
