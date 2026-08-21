# Governing Equations

ClimaAtmos solves the fully compressible, nonhydrostatic equations of motion for
a deep atmosphere [Yatunin2026](@cite). This page states those equations in
continuous form. Their discretization is treated separately, in
[Discretization and Operators](discretization.md), and the thermodynamic
quantities they contain are defined in
[Thermodynamics and the Working Fluid](thermodynamics.md).

The equations appear first in a form independent of any coordinate system, which
is what lets the same equation set serve Cartesian geometries for large-eddy and
cloud-resolving simulation and the sphere for global weather and climate
simulation.

That coordinate-independent form is also the one ClimaAtmos implements. The
tendencies in `src/prognostic_equations/` are written as the equations are
written here — divergences, gradients, and curls of vectors — with no metric
terms spelled out. Vectors carry their basis in their type, so ClimaCore
converts to the generalized-coordinate components and applies the metric terms
when it evaluates an operator. A later section gives that generalized-coordinate
form, which is what ClimaCore works in, and which a reader comparing the code
against the metric expressions will need.

## Prognostic variables

The prognostic variables for water are the total water specific humidity
``q_t``, the liquid specific humidities ``q_l^\sigma``, and the ice specific
humidities ``q_i^\sigma``. The superscript ``\sigma`` denotes cloud condensate
(`cl`) or precipitation (`pr`) where these are separate water mass categories,
and is omitted where they are not. Under the assumption of local thermodynamic
equilibrium, ``q_l^{cl}`` and ``q_i^{cl}`` follow from the other thermodynamic
variables by saturation adjustment.

The prognostic variable for energy is the specific total energy of moist air
[Romps2008, Bott2008, Sridhar2022](@cite),

```math
e_{tot} = I + \Phi + \kappa + \kappa_{SGS} .
```

Here ``\Phi = g z`` is the geopotential in the approximation of Earth as a
sphere [White2005](@cite), with ``z`` the altitude above a reference level and
``g`` taken constant; ``\kappa = \tfrac{1}{2} \|\boldsymbol{u}\|^2`` is the
specific kinetic energy of the resolved three-dimensional velocity
``\boldsymbol{u}``, which following [Romps2008](@cite) is the velocity of dry
air; and ``\kappa_{SGS}`` is the parameterized subgrid-scale kinetic energy
where a scheme provides one.

This choice of energy variable, together with a conservative discretization and
the consistent thermodynamics, conserves total energy, air mass, and water mass
to floating-point precision. Conservation then holds over complex topography and
in the presence of parameterizations, without the ad hoc fixers common in
atmosphere models [Lauritzen2022](@cite). See
[Conservation Properties](conservation.md).

When prognostic turbulence kinetic energy or PROPHET is enabled, the state also
carries ``\rho \, \mathrm{tke}`` and the PROPHET subdomain variables described in
the [PROPHET equations](@ref "PROPHET: Overview and Equations"); a slab-ocean
surface adds a prognostic surface state. [Notation and
Symbols](notation.md) maps the symbols used here onto the names in the code.

## Coordinate-independent equations of motion

Scalar quantities use flux form, which is what makes discrete conservation
attainable. Momentum uses the vector-invariant form, in which advection is
expressed through vorticity and a kinetic-energy gradient by the identity
``\boldsymbol{u} \cdot \nabla \boldsymbol{u} = \nabla \|\boldsymbol{u}\|^2 / 2 + (\nabla \times \boldsymbol{u}) \times \boldsymbol{u}``. This form represents
rotational wave modes accurately in the discrete model
[Ringler2010, TaylorFournier2010](@cite) and avoids the curvature terms
(Christoffel symbols) that arise in advection terms from coordinate derivatives
in non-orthogonal coordinates.

The governing equations for a general deep atmosphere are as follows
[White2005, Romps2008](@cite).

**Mass continuity:**

```math
\frac{\partial \rho}{\partial t}
  + \nabla \cdot \left[ \rho (\boldsymbol{u} - W_{q_t} \hat{\boldsymbol{k}}) \right]
  = \rho \hat{S}_{q_t}
```

**Momentum:**

```math
\frac{\partial \boldsymbol{u}}{\partial t}
  + (2 \boldsymbol{\Omega} + \boldsymbol{\omega}) \times \boldsymbol{u}
  = -c_{pd} (\theta_v - \theta_{v,r}) \nabla \Pi
    - \nabla (\Phi - \Phi_r) - \nabla \kappa + \boldsymbol{S}_u
    - \frac{1}{\rho} \nabla \cdot
      \left[ \rho (\boldsymbol{\mathcal{T}} + \boldsymbol{\mathcal{H}}_u) \right]
```

**Total energy:**

```math
\frac{\partial}{\partial t} (\rho e_{tot})
  + \nabla \cdot \left[ \rho (h_{tot} \boldsymbol{u} - W_h \hat{\boldsymbol{k}}) \right]
  = -\nabla \cdot \left[ \rho \left( \boldsymbol{\mathcal{F}}_R
    + \boldsymbol{\mathcal{F}}_h + \boldsymbol{\mathcal{H}}_h
    + \boldsymbol{u} \cdot (\boldsymbol{\mathcal{T}}
      + \boldsymbol{\mathcal{H}}_u) \right) \right]
```

!!! note "TODO: not yet implemented"

    The work done by the subgrid-scale and hyperdiffusive momentum fluxes, the
    term ``\boldsymbol{u} \cdot (\boldsymbol{\mathcal{T}} + \boldsymbol{\mathcal{H}}_u)`` in the energy equation above, is part of the
    formulation in [Yatunin2026](@cite) but is not yet in the code. The energy
    tendency currently carries advection, vertical diffusion, subgrid-scale and
    surface fluxes, sedimentation, and hyperdiffusion.

**Total water:**

```math
\frac{\partial}{\partial t} (\rho q_t)
  + \nabla \cdot \left[ \rho (q_t \boldsymbol{u} - W_{q_t} \hat{\boldsymbol{k}}) \right]
  = -\nabla \cdot \left[ \rho (\boldsymbol{\mathcal{F}}_{q_t}
    + \boldsymbol{\mathcal{H}}_{q_t}) \right]
  \equiv \rho \hat{S}_{q_t}
```

**Condensate and precipitation**, for ``\mu \in \{l, i\}`` and
``\sigma \in \{\cdot, \mathrm{cl}, \mathrm{pr}\}`` as needed:

```math
\frac{\partial}{\partial t} (\rho q_\mu^\sigma)
  + \nabla \cdot \left[ \rho q_\mu^\sigma
    (\boldsymbol{u} - w_\mu^\sigma \hat{\boldsymbol{k}}) \right]
  = \rho S_{q_\mu^\sigma}
    - \nabla \cdot \left[ \rho (\boldsymbol{\mathcal{F}}_{q_\mu^\sigma}
      + \boldsymbol{\mathcal{H}}_{q_\mu^\sigma}) \right]
```

**Tracers:**

```math
\frac{\partial}{\partial t} (\rho \chi)
  + \nabla \cdot \left[ \rho \chi
    (\boldsymbol{u} - w_\chi \hat{\boldsymbol{k}}) \right]
  = \rho S_\chi
    - \nabla \cdot \left[ \rho (\boldsymbol{\mathcal{F}}_\chi
      + \boldsymbol{\mathcal{H}}_\chi) \right]
```

**Equation of state:**

```math
p = \rho R_m T
```

### Symbols

| Symbol                                                                                            | Meaning                                                                    | Units           |
|:------------------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------- |:--------------- |
| ``e_{tot}``                                                                                       | Specific total energy of moist air                                         | J kg⁻¹          |
| ``h_{tot}``                                                                                       | Specific total enthalpy, ``h_{tot} = e_{tot} + p/\rho``                    | J kg⁻¹          |
| ``I``                                                                                             | Specific internal energy of moist air                                      | J kg⁻¹          |
| ``\hat{\boldsymbol{k}}``                                                                          | Vertical unit vector                                                       |                 |
| ``p``                                                                                             | Pressure                                                                   | Pa              |
| ``q_t``, ``q_l``, ``q_i``                                                                         | Total water, liquid, and ice specific humidity                             | kg kg⁻¹         |
| ``R_m``                                                                                           | Gas constant of moist air, dependent on the specific humidities            | J kg⁻¹ K⁻¹      |
| ``s_d``                                                                                           | Dry static energy, ``s_d = c_{pd}(T - T_0) + \Phi``                        | J kg⁻¹          |
| ``T``                                                                                             | Temperature                                                                | K               |
| ``\boldsymbol{u}``                                                                                | Three-dimensional velocity of dry air                                      | m s⁻¹           |
| ``w_\mu^\sigma``                                                                                  | Sedimentation or fall velocity, positive downward                          | m s⁻¹           |
| ``W_h``, ``W_{q_t}``                                                                              | Sedimentation flux of enthalpy, of total water                             | W m kg⁻¹, m s⁻¹ |
| ``\theta_v``, ``\theta_{v,r}``                                                                    | Virtual potential temperature and its reference                            | K               |
| ``\kappa``, ``\kappa_{SGS}``                                                                      | Resolved and subgrid-scale specific kinetic energy                         | J kg⁻¹          |
| ``\Pi``                                                                                           | Exner function, ``\Pi = (p/p_0)^{R_d/c_{pd}}``                             |                 |
| ``\rho``                                                                                          | Density of moist air                                                       | kg m⁻³          |
| ``\Phi``, ``\Phi_r``                                                                              | Geopotential and its reference                                             | m² s⁻²          |
| ``\chi``                                                                                          | Generic tracer                                                             | kg kg⁻¹         |
| ``\boldsymbol{\omega}``                                                                           | Relative vorticity, ``\boldsymbol{\omega} = \nabla \times \boldsymbol{u}`` | s⁻¹             |
| ``\boldsymbol{\Omega}``                                                                           | Angular velocity of planetary rotation                                     | s⁻¹             |
| ``\boldsymbol{S}_u``                                                                              | Specific momentum source                                                   | m s⁻²           |
| ``S_\psi``, ``\hat{S}_{q_t}``                                                                     | Source of scalar ``\psi``; effective source of total water                 | s⁻¹             |
| ``\boldsymbol{\mathcal{F}}_R``                                                                    | Radiative energy flux                                                      | W m kg⁻¹        |
| ``\boldsymbol{\mathcal{F}}_h``, ``\boldsymbol{\mathcal{F}}_\psi``                                 | Subgrid-scale flux of enthalpy, of scalar ``\psi``                         | W m kg⁻¹, m s⁻¹ |
| ``\boldsymbol{\mathcal{H}}_h``, ``\boldsymbol{\mathcal{H}}_\psi``, ``\boldsymbol{\mathcal{H}}_u`` | Hyperdiffusive fluxes                                                      |                 |
| ``\boldsymbol{\mathcal{T}}``                                                                      | Subgrid-scale momentum flux tensor                                         | m² s⁻²          |

Under the shallow-atmosphere approximation, the planetary rotation vector is
``\boldsymbol{\Omega} = \Omega \sin(\phi) \boldsymbol{e}^v``, with ``\phi``
latitude and ``\boldsymbol{e}^v`` the unit radial vector, so that its horizontal
contravariant component vanishes. For a deep atmosphere it is
``\boldsymbol{\Omega} = (0, 0, \Omega)``, aligned with the rotation axis. For
Earth, ``\Omega = 7.2921159 \times 10^{-5}`` s⁻¹ (the
`angular_velocity_planet_rotation` parameter).

## Properties of the equation set

**Conservation.** The equations for mass, energy, and total water are written in
flux-conservative form: their right-hand sides contain only flux-divergence
terms, with no non-conservative sources or sinks. With a flux-conservative
discretization, the volume-integrated air mass, total water, and total energy
change only through fluxes across the boundaries. This holds in the presence of
the parameterizations contained in the subgrid-scale fluxes.

**Closures and the equation of state.** The advective flux in the total energy
equation carries the specific total enthalpy ``h_{tot} = e_{tot} + p/\rho``,
which includes kinetic and potential contributions. Pressure follows from the
ideal-gas equation of state, which neglects pressure contributions from averaging
over correlated subgrid-scale fluctuations.

**Dry-air reference frame.** Following [Romps2008](@cite), ``\boldsymbol{u}`` is
the velocity of dry air. The differential motion of water relative to dry air,
such as hydrometeor fall, appears explicitly as mass and energy flux terms. The
wall-normal velocity of dry air can then be set to zero at a rigid boundary even
where surface moisture fluxes are present.

**Simplifications.** The model disregards the transport of momentum and the
kinetic energy associated with the differential motion of water relative to dry
air [Romps2008](@cite). The omitted momentum terms describe an internal
redistribution whose global integral is zero, and a precise local momentum
balance matters less for long-term stability than an exact balance for mass and
energy, where small spurious sources cause climate drift [Thuburn2008](@cite).

**Temperature.** Temperature is computed from ``e_{tot}``. Subtracting kinetic
energy from total energy does not cause catastrophic cancellation of round-off
errors, because kinetic energy is a small fraction of the total, in the ratio of
the squared Mach number ``(U/c_s)^2 \lesssim 10^{-2}`` for the atmosphere
[Peixoto1991](@cite). For perturbations alone, the ratio of kinetic to thermal
perturbations scales as ``0.5 (\delta U)^2 / (c_{vd} \delta T)``, still of order
``10^{-2}`` for ``\delta U \sim 10`` m s⁻¹ and ``\delta T \sim 5`` K.

**Pressure-gradient force.** Following [Taylor2020](@cite), the pressure-gradient
acceleration uses the Exner function and the virtual potential temperature,

```math
-\frac{1}{\rho} \nabla p = -c_{pd} \theta_v \nabla \Pi,
\qquad \theta_v = \frac{R_m}{R_d} \frac{T}{\Pi} .
```

Paired with a mimetic discretization, this avoids spurious conversions between
potential and kinetic energy in the pressure work terms [Taylor2020](@cite). To
improve accuracy and reduce discretization noise near topography, the force is
expressed as a deviation from a hydrostatically balanced reference state
[Golaz2022, Herrington2022](@cite). The reference temperature profile is

```math
T_r(\Pi) = T_{\min} + (T_{sfc} - T_{\min}) \Pi^{n_s} ,
```

which passes smoothly from ``T_{sfc}`` at the surface to ``T_{\min}`` in the
stratosphere. Two of the three numbers in it are configurable:

  - ``T_{sfc}`` and ``T_{\min}`` are Thermodynamics parameters, defaulting to
    290 K and 220 K. Override them like any other parameter, through the
    `temperature_surface_reference` and `temperature_min_reference` entries of a
    TOML file; see [Creating custom configurations](configuration.md).
  - ``n_s`` is the exponent `s_ref`, fixed at 7 by a constant in
    `src/utils/refstate_thermodynamics.jl`. Changing it requires editing the
    source.

With ``\theta_{v,r} = T_r / \Pi`` and

```math
\Phi_r = -c_{pd} \left[ T_{\min} \log \Pi
  + \frac{T_{sfc} - T_{\min}}{n_s} \left( \Pi^{n_s} - 1 \right) \right],
```

the reference state satisfies ``c_{pd} \theta_{v,r} \nabla \Pi + \nabla \Phi_r = 0`` for any ``\Pi``, which gives the combined pressure-gradient and geopotential
terms in the momentum equation.

**Reference temperature invariance.** The energetics are invariant under shifts
of the reference temperature ``T_0``, so the choice of ``T_0`` affects physical
results only through the quality of the linearization of the latent heats
[Ambaum2020](@cite). See
[Thermodynamics and the Working Fluid](thermodynamics.md).

## Parameterized terms

Several terms above come from parameterizations, each documented on its own page.

**Subgrid-scale fluxes.** The momentum flux tensor ``\boldsymbol{\mathcal{T}}``
and the scalar fluxes ``\boldsymbol{\mathcal{F}}_h``,
``\boldsymbol{\mathcal{F}}_{q_t}``, ``\boldsymbol{\mathcal{F}}_{q_\mu^\sigma}``,
and ``\boldsymbol{\mathcal{F}}_\chi`` represent the effect of unresolved
processes on the resolved flow. They may have diffusive and advective components,
the latter associated with convection closures; see
[PROPHET](prophet.md). Energetic consistency and invariance to the
reference temperature constrain the diffusive part: the total enthalpy flux is
decomposed into the enthalpy carried by the diffusive water flux and a thermal
diffusion of dry static energy, rather than modeled as the gradient of a single
lumped enthalpy.

**Hyperdiffusion.** The fluxes ``\boldsymbol{\mathcal{H}}`` serve numerical
stability rather than physics, and act along terrain-following coordinate
surfaces. See [Hyperdiffusion](hyperdiffusion.md).

**Sedimentation and fall velocities.** Condensate sediments and precipitation
falls with velocities ``w_\mu^\sigma``, defined positive downward. A microphysics
scheme relates them to hydrometeor size distributions; see
[Microphysics](microphysics.md). The total water sedimentation flux is the
weighted sum ``W_{q_t} = q_l w_l + q_i w_i``, and the sedimentation flux of
enthalpy is ``W_h = q_l h_{tot,l} w_l + q_i h_{tot,i} w_i``.

**Mass sources and sinks.** The effective moisture source ``\hat{S}_{q_t}`` in
the continuity equation arises from the subgrid-scale water fluxes. Taking the
difference between the continuity and total water equations yields an exact
conservation law for the dry air mass density ``\rho (1 - q_t)``, with no source
term. In Earth's atmosphere ``\hat{S}_{q_t}`` is usually at least two orders of
magnitude smaller than the other terms, and most models neglect it
[Abbott2024](@cite). It is retained here for generality, and matters where the
condensable species is a major atmospheric constituent, as carbon dioxide is on
Mars [Soto2015](@cite).

**Boundaries.** The wall-normal dry-air velocity vanishes at the surface, and the
subgrid-scale fluxes there follow bulk exchange laws with coefficients from
Monin–Obukhov similarity theory; see
[Surface Conditions](surface_conditions.md). The model top is a rigid lid with a
sponge layer beneath it; see [Model Top and Sponge Layer](sponge.md).

## Generalized coordinates

This section gives the form ClimaCore evaluates. Nothing here appears in the
ClimaAtmos tendencies, which stay in the coordinate-independent form above; it is
included because the metric terms determine what the operators do, and a reader
tracing an operator or a boundary condition will meet them.

The grid uses a height-based, terrain-following, stretched vertical coordinate
``\xi^3`` together with horizontal coordinates ``(\xi^1, \xi^2)`` that
parameterize a cubed sphere [Sadourny1972, Ronchi1996](@cite). In the presence of
topography these coordinates are curvilinear and non-orthogonal. See
[Topography Representation](topography.md) for the vertical coordinate and
[Grids](grids.md) for the meshes.

Write ``\boldsymbol{e}_i = \partial \boldsymbol{r} / \partial \xi^i`` for the
covariant basis vectors, tangent to the coordinate surfaces, and
``\boldsymbol{e}^i = \nabla \xi^i`` for the contravariant basis vectors,
orthogonal to them. The metric tensor components are ``g_{ij} = \boldsymbol{e}_i \cdot \boldsymbol{e}_j`` and ``g^{ij} = \boldsymbol{e}^i \cdot \boldsymbol{e}^j``, and ``J = (\det(\boldsymbol{g}))^{1/2}`` is the Jacobian
determinant, the volume element in generalized coordinates. Any vector has
covariant and contravariant components, ``\boldsymbol{u} = u_i \boldsymbol{e}^i = u^i \boldsymbol{e}_i``, related by ``u_i = g_{ij} u^j``. The differential
operators become

```math
\nabla = \boldsymbol{e}^i \frac{\partial}{\partial \xi^i}, \qquad
\nabla \cdot {} = \frac{1}{J} \frac{\partial}{\partial \xi^i} J (\boldsymbol{e}^i)^\top, \qquad
\nabla \times {} = \mathcal{E}^{ijk} \boldsymbol{e}_i
  \frac{\partial}{\partial \xi^j} (\boldsymbol{e}_k)^\top ,
```

with ``\mathcal{E}^{ijk} = \varepsilon^{ijk} / J`` the contravariant alternating
tensor. Applying these to the equations above gives the same system in
generalized coordinates.

The covariant velocity components ``u_i`` are the prognostic variables, following
[Gardner2018](@cite). That choice isolates the vertical pressure and geopotential
gradients in the ``i = 3`` component of the momentum equation, which keeps
hydrostatic balance straightforward to maintain and confines the implicit
vertical solve to one momentum component.

Evaluating a tensor divergence in these coordinates brings in curvature terms.
The model avoids computing them by transforming to Cartesian coordinates, taking
the divergence there, and transforming back [Vinokur1974](@cite).

## Where this is implemented

Each equation is assembled from tendency functions across
`src/prognostic_equations/`. Use this table to trace a term from the mathematics
to the code.

| Equation                                 | Primary source                                                                                                                                                                                                                                                           |
|:---------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Mass continuity                          | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl) (horizontal), [implicit/implicit_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/implicit_tendency.jl) (vertical) |
| Momentum, horizontal                     | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl)                                                                                                                                                                   |
| Momentum, vertical and pressure gradient | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl), [implicit/implicit_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/implicit_tendency.jl)                         |
| Total energy                             | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl) (horizontal), [implicit/implicit_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/implicit_tendency.jl) (vertical) |
| Total water and condensate advection     | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl) (horizontal), [implicit/implicit_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/implicit_tendency.jl) (vertical) |
| Sedimentation and fall velocities        | [water_advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/water_advection.jl)                                                                                                                                                       |
| Hyperdiffusive fluxes                    | [hyperdiffusion.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/hyperdiffusion.jl)                                                                                                                                                         |
| Vertical diffusion                       | [vertical_diffusion_boundary_layer.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/vertical_diffusion_boundary_layer.jl)                                                                                                                   |
| Sponge terms                             | [sponge/](https://github.com/CliMA/ClimaAtmos.jl/tree/main/src/parameterized_tendencies/sponge)                                                                                                                                                                          |
| Subgrid-scale fluxes (PROPHET)           | [edmfx_sgs_flux.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_sgs_flux.jl)                                                                                                                                                         |
| Assembly of all tendencies               | [remaining_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/remaining_tendency.jl)                                                                                                                                                 |

The discretized form of each equation, and the operators it is written in, are
given in [Discretization and Operators](discretization.md).
