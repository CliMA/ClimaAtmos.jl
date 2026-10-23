# Large-Eddy Simulation Closures

When the grid resolves the energy-containing turbulent eddies, the subgrid-scale
flux can be carried by a local eddy-viscosity closure rather than by
[PROPHET](prophet.md). ClimaAtmos provides three such closures: Smagorinsky–Lilly
[Smagorinsky1963, Lilly1962, Sridhar2022](@cite), anisotropic minimum
dissipation [Rozema2015, Akbar2016](@cite), and a constant horizontal
diffusivity. They are used in box and plane configurations — large-eddy
simulation, cloud-resolving radiative-convective equilibrium, density-current
tests — and they act on the grid-mean variables only.

The horizontal tendencies are explicit. The vertical Smagorinsky–Lilly tendency
follows `implicit_diffusion`, like the prescribed vertical diffusion; the
vertical AMD tendency currently is explicit.

## The eddy-viscosity form

Each closure supplies an eddy viscosity ``\nu_t`` and an eddy diffusivity
``D``. The momentum flux is proportional to the resolved strain rate,

```math
\boldsymbol{\mathcal{T}} = -2 \nu_t \boldsymbol{S} , \qquad
\boldsymbol{S} = \tfrac{1}{2} \left[ \nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^{T} \right] ,
```

and the scalar flux is down-gradient,

```math
\boldsymbol{\mathcal{F}}_\psi = -\rho D \nabla \psi .
```

The strain rate is assembled on both centers and faces from the full
three-dimensional velocity gradient, and its norm is
``|\boldsymbol{S}| = \sqrt{2 \boldsymbol{S} : \boldsymbol{S}}``, evaluated after
projection onto whichever axes the closure uses.

The energy flux is split, as in the other diffusive terms of the model (the
[prescribed vertical diffusion](diffusion.md), the PROPHET fluxes,
[hyperdiffusion](hyperdiffusion.md), and the [viscous sponge](sponge.md)), into
a dry-static-energy part and the enthalpy carried by the diffusing water,

```math
\boldsymbol{\mathcal{F}}_h = -\rho D \left( \nabla s_d
  + (h_{\mathrm{eff}} + \Phi) \nabla q_t^{\mathrm{eff}} \right) ,
```

with ``h_{\mathrm{eff}}`` the mass-weighted enthalpy of the suspended water and
``q_t^{\mathrm{eff}}`` the total water less rain and snow. The split makes the
flux invariant to the reference temperature ``T_0`` and ties the energy the
water flux carries to the water flux itself (see
[Thermodynamics and the Working Fluid](thermodynamics.md)). Anisotropic minimum
dissipation forms a diffusivity per scalar, so the two parts of the split use
the diffusivity of the gradient each acts on. Smagorinsky–Lilly and the
constant horizontal diffusion use a single scalar diffusivity for both parts of
the split and for all grid-scale tracers: Smagorinsky–Lilly divides its eddy
viscosity by the turbulent Prandtl number (below), and the constant horizontal
diffusion prescribes the diffusivity outright. Because total water is part of
the working fluid, its diffusive tendency is also added to the density tendency.
All water species, precipitation included, diffuse with the same diffusivity and
keep their own tendencies; the enthalpy that rain and snow carry is left out of
the energy flux, since ``q_t^{\mathrm{eff}}`` excludes them.

## Smagorinsky–Lilly

The eddy viscosity is the product of a squared mixing length and the strain-rate
norm, and the diffusivity follows from a constant turbulent Prandtl number,

```math
\nu_t = L^2 |\boldsymbol{S}| , \qquad D = \frac{\nu_t}{Pr_t} .
```

The mixing length is the filter scale times the Smagorinsky coefficient
``c_s``, and it is reduced in stable stratification by the Lilly correction

```math
f_b = \left( 1 - \frac{Ri}{Pr_t} \right)^{1/4} , \qquad
Ri = \frac{N^2}{|\boldsymbol{S}_v|^2} ,
```

with ``f_b = 1`` where ``Ri \le 0``, and with ``N^2`` computed from the vertical
gradient of virtual potential temperature. The Richardson number always uses the
vertical strain-rate norm, whatever axes the closure acts on. The correction
vanishes at ``Ri = Pr_t``, which switches the closure off in strongly stable
layers.

The `smagorinsky_lilly` configuration key selects which axes the closure acts
on, and that choice also sets the filter scale:

| Value    | Acts on                                                   | Mixing lengths                                             |
|:-------- |:--------------------------------------------------------- |:---------------------------------------------------------- |
| `"UVW"`  | Horizontal and vertical, treated as one isotropic closure | ``L_h = L_v = c_s (\Delta x \Delta y \Delta z)^{1/3} f_b`` |
| `"UV"`   | Horizontal only                                           | ``L_h = c_s \Delta x``                                     |
| `"W"`    | Vertical only                                             | ``L_v = c_s \Delta z f_b``                                 |
| `"UV_W"` | Both, with separate horizontal and vertical coefficients  | ``L_h = c_s \Delta x``, ``L_v = c_s \Delta z f_b``         |

Here, ``\Delta x`` is the horizontal node spacing and ``\Delta z`` the local
layer thickness. In the coupled `"UVW"` case, one cube-root filter width and one
strain-rate norm are used for both directions; in the other three cases the
horizontal and vertical viscosities are formed from separate lengths and from
strain-rate norms projected onto the corresponding axes. The stratification
correction enters the vertical length only, except under `"UVW"`
where there is a single length to correct.

| Parameter | ClimaParams name                 | Default | Meaning                          |
|:--------- |:-------------------------------- |:------- |:-------------------------------- |
| ``c_s``   | `c_smag`                         | 0.2     | Smagorinsky coefficient          |
| ``Pr_t``  | `mixing_length_Prandtl_number_0` | 0.74    | Neutral turbulent Prandtl number |

!!! note "A second Smagorinsky length, for cloud fraction"

    When no EDMF is active, the grid-mean subgrid-scale closure that supplies
    the cloud fraction needs a mixing length of its own, and
    `smagorinsky_lilly_length` in `gm_sgs_closures.jl` computes it. It is the
    same expression as ``L_v`` above, ``c_s \Delta z (1 - Ri / Pr)^{1/4}``,
    with the same coefficient ``c_s`` and the same vertical grid scale, but
    with three different inputs: ``Pr`` is the Richardson-dependent turbulent
    Prandtl number in place of the neutral constant ``Pr_t``; ``N^2`` is the
    cloud-fraction-blended moist buoyancy gradient, not the vertical gradient of
    ``\theta_v``; and the strain-rate norm is taken over the full tensor built
    from vertical gradients, which includes the shear of the horizontal wind,
    rather than over its vertical component alone. The result is
    used as a mixing length for the subgrid-scale variances, never to form an
    eddy viscosity ``L^2 |\boldsymbol{S}|``, so the two uses of the name do not
    overlap.

## Anisotropic minimum dissipation

The minimum-dissipation closure chooses the smallest eddy viscosity that keeps
the resolved kinetic energy from growing at the grid scale, which makes it
sensitive to grid anisotropy without an explicit filter-width prescription.
Writing ``\hat{\partial}_i = \Delta_i \partial_i`` for the grid-scaled
derivative, the viscosity is

```math
\nu_t = \max \left[ 0, \, -c_{amd} \frac{(\hat{\partial}_k u_i)(\hat{\partial}_k u_j) S_{ij}}{(\partial_l u_m)(\partial_l u_m)} \right] ,
```

and each scalar gets its own diffusivity,

```math
D_\psi = \max \left[ 0, \, -c_{amd} \frac{(\hat{\partial}_k u_i)(\hat{\partial}_k \psi)(\partial_i \psi)}{(\partial_l \psi)(\partial_l \psi)} \right] ,
```

in which the scalar derivatives ``\partial_i \psi`` are the horizontal ones in
the horizontal tendency and the vertical one in the vertical tendency, so each
scalar has a separate horizontal and vertical diffusivity. The velocity
gradients in the viscosity are the full three-dimensional ones. The numerators
use the grid-scaled derivatives and the denominators do not, so the ratio has
the dimensions of a diffusivity. Clipping at zero removes backscatter. There is
no Prandtl number: momentum and each scalar are treated separately.

The closure has a single switch: `amd_les: true` applies both the horizontal and the vertical tendency.

The coefficient ``c_{amd}`` is set by the `c_amd` configuration key, default
0.29. It is the one coefficient among these closures that is read from the YAML
configuration rather than from ClimaParams; the default configuration has a
TODO to move it to `parameters.toml`.

## Constant horizontal diffusion

The simplest option applies a fixed horizontal diffusivity to the scalars —
the split energy flux and the grid-scale tracers — and leaves momentum alone.
It is switched on with `constant_horizontal_diffusion: true`, and the
diffusivity comes from the `D_horizontal_diffusion` parameter, default
``5 \times 10^5`` m² s⁻¹. No configuration in the repository currently enables
it.

## Boundary conditions and surface fluxes

The vertical scalar fluxes vanish at the surface and at the model top: the
divergence operator applied to energy and the tracers imposes a zero-flux
boundary condition at both ends. Momentum is handled differently. The face
strain rate is built with a zero vertical velocity gradient at the boundaries,
but the horizontal gradients still contribute to the boundary stress, and the
horizontal-momentum divergence applies that stress as computed; the
vertical-momentum operator imposes zero divergence at the boundaries rather than
zero flux. Either way the closure supplies no wall-layer stress of its own.
Surface fluxes do not enter through these boundary conditions. They are added as
a separate explicit tendency confined to the surface-adjacent cell, so surface
drag and the surface enthalpy and moisture fluxes are applied whether or not an
LES closure is active; see [Surface Conditions](surface_conditions.md).

## Relation to the other diffusion schemes

**Prescribed vertical diffusion** (`vert_diff`) is a boundary-layer closure with
a prescribed diffusivity profile, and it differs in
two ways beyond the diffusivity itself: it interpolates the diffusivity to faces
as a harmonic rather than an arithmetic mean, and it diffuses the effective
total water ``q_t - q_{rai} - q_{sno}`` and redistributes the resulting mass
among the cloud species. See [Diffusion](diffusion.md) for both closures and
the water-species bookkeeping.

**PROPHET.** The LES closures and the PROPHET horizontal diffusive flux both
apply horizontal subgrid-scale diffusion to the same fields, so combining
`edmfx_sgs_horizontal_diffusive_flux` with `smagorinsky_lilly` or `amd_les` is
rejected at configuration time; see
[PROPHET Horizontal Diffusion](prophet_horizontal_diffusion.md).

**Hyperdiffusion** is a numerical filter rather than a turbulence closure, and
nothing in the code couples the two: whether it runs alongside an LES closure is
left to the configuration. The Smagorinsky and AMD configurations set
`hyperdiff: ~`, since the closure already damps the grid scale. See
[Hyperdiffusion](hyperdiffusion.md).

**Implicit vertical diffusion.** With `implicit_diffusion: true`, the vertical
Smagorinsky–Lilly tendency moves into the implicit tendency, and the eddy
viscosity is refreshed on every Newton iterate so that it follows the current
state. The Jacobian linearizes the enthalpy, tracer, and horizontal-momentum
flux divergences with the eddy viscosity frozen; the ``u_3`` term, the
horizontal-gradient part of ``\boldsymbol{\mathcal{T}}``, and the per-species
condensate diagonals have no Jacobian contribution, which slows Newton
convergence but leaves the tendency exact. A vertically acting Smagorinsky
closure satisfies the configuration check for implicit diffusion on its own;
the AMD closure has no Jacobian block and stays explicit. See
[Implicit Solver](implicit_solver.md).

## Configuration and diagnostics

| Key                             | Default | Values                                |
|:------------------------------- |:------- |:------------------------------------- |
| `smagorinsky_lilly`             | `~`     | `~`, `"UVW"`, `"UV"`, `"W"`, `"UV_W"` |
| `amd_les`                       | `false` | Boolean                               |
| `c_amd`                         | 0.29    | Float                                 |
| `constant_horizontal_diffusion` | `false` | Boolean                               |

Worked examples:
[`les_isdac_box.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/les_isdac_box.yml)
and
[`box_density_current_test.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/box_density_current_test.yml)
use `"UVW"`,
[`rcemipii_box_CRM_1M.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/rcemipii_box_CRM_1M.yml)
uses `"UV_W"`, and
[`plane_density_current_test_amd.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/plane_density_current_test_amd.yml)
and
[`baroclinic_wave_equil_amd.yml`](https://github.com/CliMA/ClimaAtmos.jl/blob/main/config/model_configs/baroclinic_wave_equil_amd.yml)
use AMD.

The Smagorinsky closure exposes its diffusivities and strain-rate norms as the
`Dh_smag`, `Dv_smag`, `strainh_smag`, and `strainv_smag` diagnostics. AMD
exposes none.

## Symbols

| Symbol                                   | Meaning                                                   | Units  |
|:---------------------------------------- |:--------------------------------------------------------- |:------ |
| ``c_s``, ``c_{amd}``                     | Smagorinsky and minimum-dissipation coefficients          |        |
| ``D``, ``D_\psi``                        | Eddy diffusivity, of scalar ``\psi``                      | m² s⁻¹ |
| ``f_b``                                  | Stratification correction to the mixing length            |        |
| ``L_h``, ``L_v``                         | Horizontal and vertical mixing length                     | m      |
| ``N``                                    | Buoyancy frequency from the virtual potential temperature | s⁻¹    |
| ``Pr_t``                                 | Neutral turbulent Prandtl number                          |        |
| ``Ri``                                   | Gradient Richardson number                                |        |
| ``\boldsymbol{S}``, ``\boldsymbol{S}_v`` | Resolved strain-rate tensor and its vertical part         | s⁻¹    |
| ``\Delta x``, ``\Delta y``, ``\Delta z`` | Horizontal node spacing and layer thickness               | m      |
| ``\nu_t``                                | Eddy viscosity                                            | m² s⁻¹ |

## Where this is implemented

| Concept                                                            | Source                                                                                                                                                                               |
|:------------------------------------------------------------------ |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Smagorinsky–Lilly viscosity, stratification correction, tendencies | [smagorinsky_lilly.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/les_sgs_models/smagorinsky_lilly.jl)                                            |
| Anisotropic minimum dissipation                                    | [anisotropic_minimum_dissipation.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/les_sgs_models/anisotropic_minimum_dissipation.jl)                |
| Constant horizontal diffusion                                      | [constant_horizontal_diffusion.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/les_sgs_models/constant_horizontal_diffusion.jl)                    |
| Strain rate and its norm                                           | [src/utils/utilities.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/utils/utilities.jl)                                                                                    |
| Prescribed vertical diffusion, for contrast                        | [Diffusion](diffusion.md)                                                                                                                                                            |
| Surface-flux tendency                                              | [src/prognostic_equations/surface_flux.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/surface_flux.jl)                                                |
| Model types                                                        | [`ClimaAtmos.EddyViscosityModel`](@ref), [`ClimaAtmos.SmagorinskyLilly`](@ref), [`ClimaAtmos.AnisotropicMinimumDissipation`](@ref), [`ClimaAtmos.ConstantHorizontalDiffusion`](@ref) |

See [Configuration Options](configuration_options.md) for the keys named here.
