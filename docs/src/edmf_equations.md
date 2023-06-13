# Sub-grid scale equations

This describes the EDMF scheme equations and its discretizations. Where possible, we use a coordinate invariant form: the ClimaCore operators generally handle the conversions between bases internally.


## Dycore variables

* ``\boldsymbol{\Omega}`` is the planetary angular velocity. We currently use a shallow-atmosphere approximation, with
  ```math
  \boldsymbol{\Omega} = \Omega \sin(\phi) \boldsymbol{e}^v
  ```
  where ``\phi`` is latitude, and ``\Omega`` is the planetary rotation rate in rads/sec (for Earth, ``7.29212 \times 10^{-5} s^{-1}``) and ``\boldsymbol{e}^v`` is the unit radial basis vector. This implies that the horizontal contravariant component ``\boldsymbol{\Omega}^h`` is zero.
* ``\boldsymbol{u}_h = u_1 \boldsymbol{e}^1 + u_2 \boldsymbol{e}^2`` is the projection onto horizontal covariant components (covariance here means with respect to the reference element), stored at cell centers.
* ``\Phi = g z`` is the geopotential, where ``g`` is the gravitational acceleration rate and ``z`` is altitude above the mean sea level.
* ``\rho_{\text{ref}}`` is the reference state density
* ``p`` is air pressure, derived from the thermodynamic state, reconstructed at cell centers.
* ``p_{\text{ref}}`` is the reference state pressure. It is related to the reference state density by analytical hydrostatic balance: ``\nabla p_{\text{ref}} = - \rho_{\text{ref}} \nabla \Phi``.

## Prognostic variables

* ``\hat{\rho}^j``: _effective density_ in kg/m³. Superscript ``j`` represents the sub-domain. ``\hat{\rho}^j = \rho^j a^j`` where ``\rho^j`` is the sub-domain density and ``a^j`` is the sub-domain area fraction. This is discretized at cell centers.
* ``\boldsymbol{u}^j`` _velocity_, a vector in m/s. This is discretized via ``\boldsymbol{u}^j = \boldsymbol{u}_h + \boldsymbol{u}_v^j`` where
  - ``\boldsymbol{u}_v^j = u_3^j \boldsymbol{e}^3`` is the projection onto the vertical covariant components, stored at cell faces.
* ``\hat{\rho}^j e^j``: _total energy_ in J/m³. This is discretized at cell centers.
* ``\hat{\rho}^j q^j``: moisture tracers. ``q^j`` stands for the sub-domain total (liquid, ice, rain, snow) specific humidity in kg/kg. This is stored at cell centers.
* ``\hat{\rho}^j \chi^j``: other tracers (aerosol, ...), again stored at cell centers.

## Operators

We make use of the following operators

### Reconstruction

* ``I^c`` is the [face-to-center reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateF2C) (arithmetic mean)
* ``I^f`` is the [center-to-face reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateC2F) (arithmetic mean)
* ``WI^f`` is the [center-to-face weighted reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeightedInterpolateC2F)
  - ``WI^f(J, x) = I^f(J*x) / I^f(J)``, where ``J`` is the value of the Jacobian for use in the weighted interpolation operator
* ``U^f`` is the [1st or 3rd-order center-to-face upwind product operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F) # fix link

### Differential operators

- ``D_h`` is the [discrete horizontal spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Divergence).
- ``\hat{\mathcal{D}}_h`` is the [discrete horizontal spectral weak divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakDivergence).
- ``D^c_v`` is the [face-to-center vertical divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C).
- ``G_h`` is the [discrete horizontal spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient).
- ``G^f_v`` is the [center-to-face vertical gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.GradientC2F).
  - the gradient is set to 0 at the top and bottom boundaries.
- ``C_h`` is the [curl components involving horizontal derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Curl)
  - ``C_h[\boldsymbol{u}_h]`` returns a vector with only vertical _contravariant_ components.
  - ``C_h[\boldsymbol{u}_v]`` returns a vector with only horizontal _contravariant_ components.
- ``\hat{\mathcal{C}}_h`` is the [weak curl components involving horizontal derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakCurl)
- ``C^f_v`` is the [center-to-face curl involving vertical derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.CurlC2F).
  - ``C^f_v[\boldsymbol{u}_h]`` returns a vector with only a horizontal _contravariant_ component.
  - the curl is set to 0 at the top and bottom boundaries.
    - We need to clarify how best to handle this.

### Projection

- ``\mathcal{P}`` is the [direct stiffness summation (DSS) operation](https://clima.github.io/ClimaCore.jl/stable/operators/#DSS), which computes the projection onto the continuous spectral element basis.

## Auxiliary and derived quantities

* ``\tilde{\boldsymbol{u}^j}`` is the mass-weighted reconstruction of velocity at the interfaces:
  by interpolation of contravariant components
    ```math
  \tilde{\boldsymbol{u}^j} = WI^f(\rho^j J, \boldsymbol{u}_h) + \boldsymbol{u}_v^j
  ```
Technically, from mass conservation, the weighting factor should be ``\hat{\rho}^j J``.
However, in order to avoid issues coming from close to zero sub-domain area fractions,
we can instead use ``\rho^j J`` or even ``\rho J``.

* ``\bar{\boldsymbol{u}}^j`` is the reconstruction of velocity at cell-centers,
  carried out by linear interpolation of the covariant vertical component:
  ```math
  \bar{\boldsymbol{u}}^j = \boldsymbol{u}_h + I_{c}(\boldsymbol{u}_v^j)
  ```

* ``\boldsymbol{b}^j`` is the reduced gravitational acceleration
  ```math
  \boldsymbol{b}^j = - \frac{\rho^j - \rho_{\text{ref}}}{\rho^j} \nabla \Phi
  ```
* ``K^j = \tfrac{1}{2} \|\boldsymbol{u}^j\|^2 `` is the specific kinetic energy (J/kg), reconstructed at cell centers by
  ```math
  K^j = \tfrac{1}{2} (\boldsymbol{u}_{h}^j \cdot \boldsymbol{u}_{h}^j + 2 \boldsymbol{u}_{h}^j \cdot I_{c} (\boldsymbol{u}_{v}^j) + I_{c}(\boldsymbol{u}_{v}^j \cdot \boldsymbol{u}_{v}^j)),
  ```
  where ``\boldsymbol{u}_{h}^j`` is defined on cell-centers, ``\boldsymbol{u}_{v}^j`` is defined on cell-faces, and ``I_{c} (\boldsymbol{u}_{v})`` is interpolated using covariant components.

* ``\nu_u``, ``\nu_h``, and ``\nu_\chi`` are hyperdiffusion coefficients, and ``c`` is the divergence damping factor.

* No-flux boundary conditions are enforced by requiring the third contravariant component of the face-valued velocity at the boundary, ``\boldsymbol{\tilde{u}}^{v,j}``, to be zero. The vertical covariant velocity component is computed as
  ```math
  \tilde{u}_{v}^j = - \tfrac{u_{1}g^{31} + u_{2}g^{32}}{g^{33}}.
  ```

## Equations and discretizations

### Mass

Follows the continuity equation
```math
\frac{\partial}{\partial t} \hat{\rho}^j = - \nabla \cdot (\hat{\rho}^j \boldsymbol{u}^j)  + RHS
```

This is discretized using the following
```math
\frac{\partial}{\partial t} \hat{\rho}^j
= - D_h[ \hat{\rho}^j (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v^j))] - D^c_v \left[WI^f( J, \hat{\rho}^j) \tilde{\boldsymbol{u}^j} \right] + RHS
```

### Momentum

Uses the advective form equation
```math
\frac{\partial}{\partial t} \boldsymbol{u}^j  = - (2 \boldsymbol{\Omega} + \nabla \times \boldsymbol{u}^j) \times \boldsymbol{u}^j - \frac{1}{\rho^j} \nabla (p - p_{\text{ref}})  + \boldsymbol{b}^j - \nabla K^j + RHS
```
By breaking the curl and cross product terms into horizontal and vertical contributions, and removing zero terms (e.g. ``\nabla_v  \times \boldsymbol{u}_v = 0``), we obtain
the vertical momentum equation. The horizontal momentum equation is only solved in the grid-mean.

#### Vertical momentum
```math
\frac{\partial}{\partial t} \boldsymbol{u}_v^j  =
  - (\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v^j) \times \boldsymbol{u}^h
  - \frac{1}{\rho^j} \nabla_v (p - p_{\text{ref}}) - \frac{\rho^j - \rho_{\text{ref}}}{\rho^j} \nabla_v \Phi - \nabla_v K^j + RHS .
```

This is stabilized with the addition of 4th-order vector hyperviscosity
```math
-\nu_u \nabla_h^2(\nabla_h^2(\boldsymbol{u}^j)),
```
projected onto the third contravariant direction.

The ``(\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v^j) \times \boldsymbol{u}^h`` term is discretized as
```math
(C^f_v[\boldsymbol{u}_h] + C_h[\boldsymbol{u}_v^j]) \times I^f(\boldsymbol{u}^h) ,
```
and the ``-\frac{1}{\rho^j} \nabla_v (p - p_{\text{ref}}) - \frac{\rho^j - \rho_{\text{ref}}}{\rho^j} \nabla_v \Phi - \nabla_v K^j`` term as
```math
-\frac{1}{I^f(\rho^j)} G^f_v[p - p_{\text{ref}}] - \frac{I^f(\rho^j - \rho_{\text{ref}})}{I^f(\rho^j)} G^f_v[\Phi] - G^f_v[K^j] ,
```

The hyperviscosity term is
```math
- \nu_u \hat{\mathcal{D}}_h (\mathcal{G}_h (\psi) )
```
where
```math
\psi = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h (w^j)\right) \right]
```

### Total energy

```math
\frac{\partial}{\partial t} \hat{\rho}^j e^j = - \nabla \cdot((\hat{\rho}^j e^j + \frac{\hat{\rho}^j}{\rho^j}p) \boldsymbol{u}^j) - \frac{p}{\rho} \frac{\partial}{\partial t} \hat{\rho}^j + RHS
```
which is stabilized with the addition of a 4th-order hyperdiffusion term on total enthalpy:
```math
- \nu_h \nabla \cdot \left( \hat{\rho}^j \nabla^3 \left(\frac{\rho^j e^j + p}{\rho^j} \right)\right)
```

is discretized using
```math
\frac{\partial}{\partial t} \hat{\rho}^j e^j \approx
- D_h[ (\hat{\rho^j} e^j + \frac{\hat{\rho^j}}{\rho^j}p) (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v^j))]
- D^c_v \left[ WI^f(J,\hat{\rho}^j) \,  \tilde{\boldsymbol{u}}^j \, I^f \left(\frac{\hat{\rho^j} e^j + \frac{\hat{\rho^j}}{\rho^j}p}{\hat{\rho}^j} \right)
  \right] - \frac{p}{\rho} \frac{\partial}{\partial t} \hat{\rho}^j - \nu_h \hat{\mathcal{D}}_h( \rho \mathcal{G}_h(\psi^j) ) + RHS .
```
where
```math
\psi^j = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h \left(\frac{\rho^j e^j + p}{\rho^j} \right)\right) \right]
```

!!! todo
    Need to change this to first order upwinding.

### Moisture tracers

For a sub-domain moisture scalar ``q^j``, the density-weighted scalar ``\hat{\rho}^j q^j`` obeys the conservation law

```math
\frac{\partial}{\partial t} \hat{\rho}^j q^j = - \nabla \cdot(\hat{\rho}^j q^j (\boldsymbol{u}^j - w_q^j \hat{\boldsymbol{k}})) + RHS .
```
where ``\hat{\boldsymbol{k}}`` is the vertical unit vector and ``w_q^j`` is the terminal velocity.

This is stabilized with the addition of a 4th-order hyperdiffusion term
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

Currently we use the central reconstruction
```math
- D^c_v \left[ WI^f(J,\hat{\rho}^j) \, \tilde{\boldsymbol{u}}^j \, I^f\left( \frac{\hat{\rho}^j q^j}{\hat{\rho}^j} \right) \right]
```
!!! todo
    Need to change this to first order upwinding.

!!! todo
    Write down the discretization for sedimentation. Assume the sedimentation velocity is zero for now.

### Other tracers

For a sub-domain scalar ``\chi^j``, the density-weighted scalar ``\hat{\rho}^j \chi^j`` follows the continuity equation

```math
\frac{\partial}{\partial t} \hat{\rho}^j \chi^j = - \nabla \cdot(\hat{\rho}^j \chi^j \boldsymbol{u}^j) + RHS .
```

This is stabilized with the addition of a 4th-order hyperdiffusion term
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

Currently we use the central reconstruction
```math
- D^c_v \left[ WI^f(J,\hat{\rho}^j) \, \tilde{\boldsymbol{u}}^j \, I^f\left( \frac{\hat{\rho}^j \chi^j}{\hat{\rho}^j} \right) \right]
```
!!! todo
    Need to change this to first order upwinding.
