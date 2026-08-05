# PROPHET Sub-Grid Scale Equations

This describes the equations of PROPHET (Prognostic Representation Of Physics
for Eddy Transport), an extended, prognostic eddy-diffusivity mass-flux (EDMF)
scheme that is still called EDMFX in the code, and their discretizations. Where
possible, we use a coordinate invariant form: the ClimaCore operators generally
handle the conversions between bases internally.

!!! warning "Discretization details under revision"

    Parts of this page predate the current implicit implementation of the
    draft equations. In the code, the draft mass (`ρa`) and vertical velocity
    are advanced by analytic implicit-stage solves
    (`src/prognostic_equations/implicit/initialize_implicit_problem.jl`),
    horizontal fluxes use split (skew-symmetric) forms, buoyancy is computed
    relative to the grid-mean density, and hyperdiffusion acts on decomposed,
    unweighted subdomain fields (`src/prognostic_equations/hyperdiffusion.jl`).
    The discretization sections below are being revised to match; consult the
    source files above for the current forms.

## Dycore variables

  - ``\boldsymbol{\Omega}`` is the planetary angular velocity. The default is the deep-atmosphere form ``\boldsymbol{\Omega} = (0, 0, \Omega)`` aligned with the rotation axis (`deep_atmosphere: true`); the shallow-atmosphere approximation is
    ```math
    \boldsymbol{\Omega} = \Omega \sin(\phi) \boldsymbol{e}^v
    ```
    where ``\phi`` is latitude, and ``\Omega`` is the planetary rotation rate in rads/sec (for Earth, ``7.29212 \times 10^{-5} s^{-1}``) and ``\boldsymbol{e}^v`` is the unit radial basis vector. This implies that the horizontal contravariant component ``\boldsymbol{\Omega}^h`` is zero.
  - ``\boldsymbol{u}_h = u_1 \boldsymbol{e}^1 + u_2 \boldsymbol{e}^2`` is the projection onto horizontal covariant components (covariance here means with respect to the reference element), stored at cell centers.
  - ``\Phi = g z`` is the geopotential, where ``g`` is the gravitational acceleration rate and ``z`` is altitude above the mean sea level.
  - ``\rho`` is the grid-mean density; draft buoyancy is computed relative to it (the code no longer carries a separate reference-state density or pressure for the drafts).
  - ``p`` is air pressure, derived from the thermodynamic state, reconstructed at cell centers.

## Prognostic variables

  - ``\hat{\rho}^j``: _effective density_ in kg/m³. Superscript ``j`` represents the sub-domain. ``\hat{\rho}^j = \rho^j a^j`` where ``\rho^j`` is the sub-domain density and ``a^j`` is the sub-domain area fraction. This is discretized at cell centers.
  - ``\boldsymbol{u}^j`` _velocity_, a vector in m/s. This is discretized via ``\boldsymbol{u}^j = \boldsymbol{u}_h + \boldsymbol{u}_v^j`` where
      + ``\boldsymbol{u}_v^j = u_3^j \boldsymbol{e}^3`` is the projection onto the vertical covariant components, stored at cell faces.
  - ``h_s^j``: _specific moist static energy_ in J/kg (the state field `mse`). Unlike the grid-mean energy, this is a specific (not density-weighted) variable, evolved in advective form. This is discretized at cell centers.
  - ``q^j``: specific moisture tracers in kg/kg (total water `q_tot`, and with 1M/2M microphysics the condensate and precipitation species). These are specific variables stored at cell centers.
  - ``\chi^j``: other specific tracers (aerosol, ...), again stored at cell centers.

## Operators

We make use of the following operators.

!!! note

    On ClimaCore `main`, the strong- and weak-form horizontal spectral operators
    have been unified: `Divergence`, `Gradient`, and `Curl` take a form-type
    parameter (`StrongForm`, the default, or `WeakForm`), so the weak divergence,
    for example, is `Divergence{I, WeakForm}`. The ClimaAtmos code does not use
    the unified names yet; the links below point to the operator documentation
    in the latest ClimaCore release.

### Reconstruction

  - ``I^c`` is the face-to-center reconstruction operator [`ClimaCore.Operators.InterpolateF2C`](@extref) (arithmetic mean).
  - ``I^f`` is the center-to-face reconstruction operator [`ClimaCore.Operators.InterpolateC2F`](@extref) (arithmetic mean).
  - ``WI^f`` is the center-to-face weighted reconstruction operator [`ClimaCore.Operators.WeightedInterpolateC2F`](@extref).
      + ``WI^f(J, x) = I^f(J*x) / I^f(J)``, where ``J`` is the value of the Jacobian for use in the weighted interpolation operator.
  - ``U^f`` is the 1st-order ([`ClimaCore.Operators.UpwindBiasedProductC2F`](@extref)) or 3rd-order ([`ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F`](@extref)) center-to-face upwind product operator.

### Differential operators

  - ``D_h`` is the discrete horizontal spectral divergence [`ClimaCore.Operators.Divergence`](@extref).
  - ``\hat{\mathcal{D}}_h`` is the discrete horizontal spectral weak divergence [`ClimaCore.Operators.WeakDivergence`](@extref).
  - ``D^c_v`` is the face-to-center vertical divergence [`ClimaCore.Operators.DivergenceF2C`](@extref).
  - ``G_h`` is the discrete horizontal spectral gradient [`ClimaCore.Operators.Gradient`](@extref).
  - ``G^f_v`` is the center-to-face vertical gradient [`ClimaCore.Operators.GradientC2F`](@extref).
      + the gradient is set to 0 at the top and bottom boundaries.
  - ``C_h`` is the curl components involving horizontal derivatives [`ClimaCore.Operators.Curl`](@extref).
      + ``C_h[\boldsymbol{u}_h]`` returns a vector with only vertical _contravariant_ components.
      + ``C_h[\boldsymbol{u}_v]`` returns a vector with only horizontal _contravariant_ components.
  - ``\hat{\mathcal{C}}_h`` is the weak curl components involving horizontal derivatives [`ClimaCore.Operators.WeakCurl`](@extref).
  - ``C^f_v`` is the center-to-face curl involving vertical derivatives [`ClimaCore.Operators.CurlC2F`](@extref).
      + ``C^f_v[\boldsymbol{u}_h]`` returns a vector with only a horizontal _contravariant_ component.
      + the curl is set to 0 at the top and bottom boundaries.

### Projection

  - ``\mathcal{P}`` is the [direct stiffness summation (DSS) operation](@extref ClimaCore DSS), which computes the projection onto the continuous spectral element basis.

## Auxiliary and derived quantities

  - ``\tilde{\boldsymbol{u}}^j`` is the mass-weighted reconstruction of velocity at the interfaces,
    obtained by interpolation of contravariant components:
    ```math
    \tilde{\boldsymbol{u}}^j = WI^f \left( \rho J, \boldsymbol{u}_h \right) + \boldsymbol{u}_v^j.
    ```

Technically, from mass conservation, the weighting factor should be ``\hat{\rho}^j J``.
However, to avoid issues from near-zero sub-domain area fractions, the code uses
the grid-mean weight ``\rho J`` (the same weighted interpolation as for the
grid-mean velocity).

  - ``\bar{\boldsymbol{u}}^j`` is the reconstruction of velocity at cell-centers,
    carried out by linear interpolation of the covariant vertical component:

    ```math
    \bar{\boldsymbol{u}}^j = \boldsymbol{u}_h + I_{c}(\boldsymbol{u}_v^j),
    ```

  - ``\boldsymbol{b}^j`` is the reduced gravitational acceleration

    ```math
    \boldsymbol{b}^j = - \frac{\rho^j - \rho}{\rho^j} \nabla \Phi,
    ```

  - ``K^j = \tfrac{1}{2} \|\boldsymbol{u}^j\|^2`` is the specific kinetic energy (J/kg), reconstructed at cell centers by

    ```math
    K^j = \tfrac{1}{2} \left(\boldsymbol{u}_{h}^j \cdot \boldsymbol{u}_{h}^j + 2 \boldsymbol{u}_{h}^j \cdot I_{c} (\boldsymbol{u}_{v}^j) + I_{c}(\boldsymbol{u}_{v}^j \cdot \boldsymbol{u}_{v}^j) \right),
    ```

    where ``\boldsymbol{u}_{h}^j`` is defined on cell-centers, ``\boldsymbol{u}_{v}^j`` is defined on cell-faces, and ``I_{c} (\boldsymbol{u}_{v})`` is interpolated using covariant components.

  - ``\nu_u``, ``\nu_h``, and ``\nu_\chi`` are hyperdiffusion coefficients, and ``c`` is the divergence damping factor.

  - No-flux boundary conditions are enforced by requiring the third contravariant component ``\boldsymbol{\tilde{u}}^{v,j}`` of the face-valued velocity at the boundary to be zero. The vertical covariant velocity component is computed as

    ```math
    \tilde{u}_{v}^j = - \frac{u_{1}g^{31} + u_{2}g^{32}}{g^{33}}.
    ```

## Equations and discretizations

### Mass

Follows the continuity equation

```math
\frac{\partial}{\partial t} \hat{\rho}^j = - \nabla \cdot (\hat{\rho}^j \boldsymbol{u}^j)  + RHS.
```

This is discretized using the following

```math
\frac{\partial}{\partial t} \hat{\rho}^j
= - D_h \left[ \hat{\rho}^j (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v^j)) \right] - D^c_v \left[WI^f( J, \hat{\rho}^j) \tilde{\boldsymbol{u}^j} \right] + RHS.
```

### Momentum

Uses the advective form equation

```math
\frac{\partial}{\partial t} \boldsymbol{u}^j  = - (2 \boldsymbol{\Omega} + \nabla \times \boldsymbol{u}^j) \times \boldsymbol{u}^j - \frac{1}{\rho^j} \nabla p'  + \boldsymbol{b}^j - \nabla K^j + RHS.
```

By breaking the curl and cross product terms into horizontal and vertical contributions, and removing zero terms (e.g. ``\nabla_v \times \boldsymbol{u}_v = 0``), we obtain
the vertical momentum equation. The horizontal momentum equation is only solved in the grid-mean.

#### Vertical momentum

```math
\frac{\partial}{\partial t} \boldsymbol{u}_v^j  =
  - (\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v^j) \times \boldsymbol{u}^h
  - \frac{\rho^j - \rho}{\rho^j} \nabla_v \Phi - \nabla_v K^j + RHS .
```

This is stabilized by adding 4th-order vector hyperviscosity

```math
-\nu_u \nabla_h^2(\nabla_h^2(\boldsymbol{u}^j)),
```

projected onto the third contravariant direction.

The ``(\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v^j) \times \boldsymbol{u}^h`` term is discretized as

```math
(C^f_v[\boldsymbol{u}_h] + C_h[\boldsymbol{u}_v^j]) \times I^f(\boldsymbol{u}^h) ,
```

and the ``-\frac{\rho^j - \rho}{\rho^j} \nabla_v \Phi - \nabla_v K^j`` terms as

```math
- \frac{I^f(\rho^j - \rho)}{I^f(\rho^j)} G^f_v[\Phi] - G^f_v[K^j] ,
```

The hyperviscosity term is

```math
- \nu_u \hat{\mathcal{D}}_h (\mathcal{G}_h (\psi) ),
```

where

```math
\psi = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h (w^j)\right) \right].
```

### Total energy

```math
\frac{\partial}{\partial t} \hat{\rho}^j e^j = - \nabla \cdot((\hat{\rho}^j e^j + \frac{\hat{\rho}^j}{\rho^j}p) \boldsymbol{u}^j) - \frac{p}{\rho} \frac{\partial}{\partial t} \hat{\rho}^j + RHS
```

which is stabilized by adding a 4th-order hyperdiffusion term on total enthalpy:

```math
- \nu_h \nabla \cdot \left( \hat{\rho}^j \nabla^3 \left(\frac{\rho^j e^j + p}{\rho^j} \right)\right).
```

The equation is discretized as

```math
\frac{\partial}{\partial t} \hat{\rho}^j e^j \approx
- D_h \left[
    \left( \hat{\rho}^j e^j + \frac{\hat{\rho}^j}{\rho^j}p \right)
    \left( \boldsymbol{u}_h + I^c(\boldsymbol{u}_v^j) \right)
  \right]
- D^c_v \left[
    WI^f(J,\hat{\rho}^j) \,  \tilde{\boldsymbol{u}}^j \, I^f \left(\frac{\hat{\rho}^j e^j + \frac{\hat{\rho}^j}{\rho^j}p}{\hat{\rho}^j} \right)
  \right]
  - \frac{p}{\rho} \frac{\partial}{\partial t} \hat{\rho}^j - \nu_h \hat{\mathcal{D}}_h( \rho \mathcal{G}_h(\psi^j) ) + RHS .
```

where

```math
\psi^j = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h \left(\frac{\rho^j e^j + p}{\rho^j} \right)\right) \right]
```

!!! note

    The vertical advection reconstruction is controlled by the
    `edmfx_mse_q_tot_upwinding` configuration argument; the default is
    first-order upwinding, and the central reconstruction shown here
    corresponds to `none`.

### Moisture tracers

For a sub-domain moisture scalar ``q^j``, the density-weighted scalar ``\hat{\rho}^j q^j`` obeys the conservation law

```math
\frac{\partial}{\partial t} \hat{\rho}^j q^j = - \nabla \cdot(\hat{\rho}^j q^j (\boldsymbol{u}^j - w_q^j \hat{\boldsymbol{k}})) + RHS .
```

where ``\hat{\boldsymbol{k}}`` is the vertical unit vector and ``w_q^j`` is the terminal velocity.

This is stabilized by adding a 4th-order hyperdiffusion term

```math
- \nu_q \nabla \cdot(\hat{\rho}^j \nabla^3(q^j))
```

This is discretized using the following

```math
\frac{\partial}{\partial t} \hat{\rho}^j q^j \approx
- D_h[ \hat{\rho}^j q^j (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v^j))]
- D^c_v \left[ WI^f(J,\hat{\rho}^j) \, U^f\left( \tilde{\boldsymbol{u}}^j,  \frac{\hat{\rho}^j q^j}{\hat{\rho}^j} \right) \right]
- \nu_\chi \hat{\mathcal{D}}_h ( \hat{\rho^j} \, \mathcal{G}_h (\psi^j) ) + sedimentation + RHS.
```

where

```math
\psi^j = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h \left( \frac{\hat{\rho}^j q^j}{\hat{\rho}^j} \right)\right) \right]
```

The `none` option corresponds to the central reconstruction

```math
- D^c_v \left[ WI^f(J,\hat{\rho}^j) \, \tilde{\boldsymbol{u}}^j \, I^f\left( \frac{\hat{\rho}^j q^j}{\hat{\rho}^j} \right) \right]
```

!!! note

    The vertical advection reconstruction is controlled by the
    `edmfx_mse_q_tot_upwinding` configuration argument; the default is
    first-order upwinding, and the central reconstruction shown here
    corresponds to `none`. The discretization of the sedimentation term
    is not yet written down here.

### Other tracers

For a sub-domain scalar ``\chi^j``, the density-weighted scalar ``\hat{\rho}^j \chi^j`` follows the continuity equation

```math
\frac{\partial}{\partial t} \hat{\rho}^j \chi^j = - \nabla \cdot(\hat{\rho}^j \chi^j \boldsymbol{u}^j) + RHS .
```

This is stabilized by adding a 4th-order hyperdiffusion term

```math
- \nu_\chi \nabla \cdot(\hat{\rho}^j \nabla^3(\chi^j))
```

This is discretized using the following

```math
\frac{\partial}{\partial t} \hat{\rho}^j \chi^j \approx
- D_h[ \hat{\rho^j} \chi^j (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v^j))]
- D^c_v \left[ WI^f(J,\hat{\rho^j}) \, U^f\left( \tilde{\boldsymbol{u}}^j,  \frac{\hat{\rho}^j \chi^j}{\hat{\rho^j}} \right) \right]
- \nu_\chi \hat{\mathcal{D}}_h ( \hat{\rho^j} \, \mathcal{G}_h (\psi^j) ) + RHS.
```

where

```math
\psi^j = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h \left( \frac{\hat{\rho}^j \chi^j}{\hat{\rho}^j} \right)\right) \right]
```

The `none` option corresponds to the central reconstruction

```math
- D^c_v \left[ WI^f(J,\hat{\rho}^j) \, \tilde{\boldsymbol{u}}^j \, I^f\left( \frac{\hat{\rho}^j \chi^j}{\hat{\rho}^j} \right) \right]
```

!!! note

    The vertical advection reconstruction is controlled by the
    `edmfx_tracer_upwinding` configuration argument; the default is
    first-order upwinding, and the central reconstruction shown here
    corresponds to `none`.
