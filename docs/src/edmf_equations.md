# Equations


This describes the EDMF scheme equations and its discretizations. Where possible, we use a coordinate invariant form: the ClimaCore operators generally handle the conversions between bases internally.



## Prognostic variables

* $\hat{\rho}^j$: _effective density_ in kg/m³. ``\hat{\rho}^j = \rho^j a^j`` where ``\rho`` is density and ``a`` is the sub-domain area fraction. Superscript ``j`` represents the sub-domain. This is discretized at cell centers.
* ``\boldsymbol{u}^j`` _velocity_, a vector in m/s. This is discretized via ``\boldsymbol{u}^j = \boldsymbol{u}_h^j + \boldsymbol{u}_v^j`` where
  - ``\boldsymbol{u}_h^j = u_1^j \boldsymbol{e}^1 + u_2^j \boldsymbol{e}^2`` is the projection onto horizontal covariant components (covariance here means with respect to the reference element), stored at cell centers.
  - ``\boldsymbol{u}_v^j = u_3^j \boldsymbol{e}^3`` is the projection onto the vertical covariant components, stored at cell faces.
* - ``\hat{\rho}^j e^j``: _total energy_ in J/m³. This is discretized at cell centers.
* ``\hat{\rho}^j \q^j``: moisture (total, liquid, ice, rain, snow), stored at cell centers.
* ``\hat{\rho}^j \chi^j``: tracers (aerosol, ...), again stored at cell centers.

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
- ``D^c_v`` is the [face-to-center vertical divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C).
!!! todo
    Add vertical diffusive tendencies (including surface fluxes)

- ``G_h`` is the [discrete horizontal spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient).
- ``G^f_v`` is the [center-to-face vertical gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.GradientC2F).
  - the gradient is set to 0 at the top and bottom boundaries.
- ``C_h`` is the [curl components involving horizontal derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Curl)
  - ``C_h[\boldsymbol{u}_h]`` returns a vector with only vertical _contravariant_ components.
  - ``C_h[\boldsymbol{u}_v]`` returns a vector with only horizontal _contravariant_ components.
- ``C^f_v`` is the [center-to-face curl involving vertical derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.CurlC2F).
  - ``C^f_v[\boldsymbol{u}_h]`` returns a vector with only a horizontal _contravariant_ component.
  - the curl is set to 0 at the top and bottom boundaries.
    - We need to clarify how best to handle this.

## Auxiliary and derived quantities

* ``\boldsymbol{\Omega}`` is the planetary angular velocity. We currently use a shallow-atmosphere approximation, with
  ```math
  \boldsymbol{\Omega} = \Omega \sin(\phi) \boldsymbol{e}^v
  ```
  where ``\phi`` is latitude, and ``\Omega`` is the planetary rotation rate in rads/sec (for Earth, ``7.29212 \times 10^{-5} s^{-1}``) and ``\boldsymbol{e}^v`` is the unit radial basis vector. This implies that the horizontal contravariant component ``\boldsymbol{\Omega}^h`` is zero.
* ``\tilde{\boldsymbol{u}}`` is the mass-weighted reconstruction of velocity at the interfaces:
  by interpolation of contravariant components
    ```math
  \tilde{\boldsymbol{u}} = WI^f(\rho J, \boldsymbol{u}_h) + \boldsymbol{u}_v
  ```
  and ``\bar{\boldsymbol{u}}`` is the reconstruction of velocity at cell-centers, carried out by linear interpolation of the covariant vertical component:
* ``\bar{\boldsymbol{u}} = \boldsymbol{u}_h + I_{c}(\boldsymbol{u}_v)``

* ``\Phi = g z`` is the geopotential, where ``g`` is the gravitational acceleration rate and ``z`` is altitude above the mean sea level.
* ``\boldsymbol{b}^j`` is the reduced gravitational acceleration
  ```math
  \boldsymbol{b}^j = - \frac{\rho^j - \rho_{\text{ref}}}{\rho^j} \nabla \Phi
  ```
* ``\rho_{\text{ref}}`` is the reference state density
* ``K^j = \tfrac{1}{2} \|\boldsymbol{u}^j\|^2 `` is the specific kinetic energy (J/kg), reconstructed at cell centers by
  ```math
  K = \tfrac{1}{2} (\boldsymbol{u}_{h}^j \cdot \boldsymbol{u}_{h}^j + 2 \boldsymbol{u}_{h}^j \cdot I_{c} (\boldsymbol{u}_{v}^j) + I_{c}(\boldsymbol{u}_{v}^j \cdot \boldsymbol{u}_{v}^j)),
  ```
  where ``\boldsymbol{u}_{h}^j`` is defined on cell-centers, ``\boldsymbol{u}_{v}^j`` is defined on cell-faces, and ``I_{c} (\boldsymbol{u}_{v})`` is interpolated using covariant components.  

* ``p`` is air pressure, derived from the thermodynamic state, reconstructed at cell centers.
* ``p_{\text{ref}}`` is the reference state pressure. It is related to the reference state density by analytical hydrostatic balance: ``\nabla p_{\text{ref}} = - \rho_{\text{ref}} \nabla \Phi``.
* ``\boldsymbol{F}_R`` are the radiative fluxes: these are assumed to align vertically (i.e. the horizontal contravariant components are zero), and are constructed at cell faces from [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl).

* No-flux boundary conditions are enforced by requiring the third contravariant component of the face-valued velocity at the boundary, ``\boldsymbol{\tilde{u}}^{v}``, to be zero. The vertical covariant velocity component is computed as
  ```math
  \tilde{u}_{v} = \tfrac{-(u_{1}g^{31} + u_{2}g^{32})}{g^{33}}.
  ```

## Equations and discretizations

### Mass

Follows the continuity equation
```math
\frac{\partial}{\partial t} \rho\hat^j = - \nabla \cdot(\rho\hat^j \boldsymbol{u}^j)  + RHS
```

This is discretized using the following
```math
\frac{\partial}{\partial t} \rho 
= - D_h[ \rho (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v))] - D^c_v \left[WI^f( J, \rho) \tilde{\boldsymbol{u}} \right]
```

with the
```math
-D^c_v[WI^f(J, \rho) \boldsymbol{u}_v]
```
term treated implicitly (check this)


### Momentum

Uses the advective form equation
```math
\frac{\partial}{\partial t} \boldsymbol{u}^j  = - (2 \boldsymbol{\Omega} + \nabla \times \boldsymbol{u}) \times \boldsymbol{u} - \frac{1}{\rho} \nabla (p - p_{\text{ref}})  + \boldsymbol{b} - \nabla K .
```

#### Horizontal momentum

By breaking the curl and cross product terms into horizontal and vertical contributions, and removing zero terms (e.g. ``\nabla_v  \times \boldsymbol{u}_v = 0``), we obtain

```math
\frac{\partial}{\partial t} \boldsymbol{u}_h  =
  - (\nabla_v \times \boldsymbol{u}_h +  \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v
  - (2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h
  - \frac{1}{\rho} \nabla_h (p - p_{\text{ref}})  - \nabla_h (\Phi + K) ,
```
where ``\boldsymbol{u}^h`` and ``\boldsymbol{u}^v`` are the horizontal and vertical _contravariant_ vectors. The effect of topography is accounted for through the computation of the contravariant velocity components (projections from the covariant velocity representation) prior to computing the cross-product contributions. 

The ``(\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v`` term is discretized as: 
```math
\frac{I^c((C^f_v[\boldsymbol{u}_h] + C_h[\boldsymbol{u}_v]) \times (I^f(\rho J)\tilde{\boldsymbol{u}}^v))}{\rho J}
```
where 
```math
\omega^{h} = (\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v)
```

The ``(2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h`` term is discretized as
```math
(2 \boldsymbol{\Omega}^v + C_h[\boldsymbol{u}_h]) \times \boldsymbol{u}^h
```
and the ``\frac{1}{\rho} \nabla_h (p - p_h)  + \nabla_h (\Phi + K)`` as
```math
\frac{1}{\rho} G_h[p - p_{\text{ref}}] + G_h[\Phi + K] ,
```
where all these terms are treated explicitly.

#### Vertical momentum
Similarly for vertical velocity
```math
\frac{\partial}{\partial t} \boldsymbol{u}_v  =
  - (\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^h
  - \frac{1}{\rho} \nabla_v (p - p_{\text{ref}}) - \frac{\rho - \rho_{\text{ref}}}{\rho} \nabla_v \Phi - \nabla_v K .
```
The ``(\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^h`` term is discretized as
```math
(C^f_v[\boldsymbol{u}_h] + C_h[\boldsymbol{u}_v]) \times I^f(\boldsymbol{u}^h) ,
```
and the ``\frac{1}{\rho} \nabla_v (p - p_{\text{ref}}) - \frac{\rho - \rho_{\text{ref}}}{\rho} \nabla_v \Phi - \nabla_v K`` term as
```math
\frac{1}{I^f(\rho)} G^f_v[p - p_{\text{ref}}] - \frac{I^f(\rho - \rho_{\text{ref}})}{I^f(\rho)} G^f_v[\Phi] - G^f_v[K] ,
```
with the latter treated implicitly.


### Total energy

```math
\frac{\partial}{\partial t} \rho e = - \nabla \cdot((\rho e + p) \boldsymbol{u} + \boldsymbol{F}_R)
```
is discretized using
```math
\frac{\partial}{\partial t} \rho e \approx
- D_h[ (\rho e + p) (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v))]
- D^c_v \left[ WI^f(J,\rho) \,  \tilde{\boldsymbol{u}} \, I^f \left(\frac{\rho e + p}{\rho} \right)
  + \boldsymbol{F}_R 
  \right] .
```
Currently the central reconstruction
```math
- D^c_v \left[ WI^f(J,\rho) \,  \tilde{\boldsymbol{u}} \, I^f \left(\frac{\rho e + p}{\rho} \right) \right]
```
is treated implicitly.

!!! todo
    The Jacobian computation should be updated so that the upwinded term
    ```math
    - D^c_v\left[WI^f(J, \rho) U^f\left(\boldsymbol{u}_v, \frac{\rho e + p}{\rho} \right)\right]
    ```
    is treated implicitly.


### Internal energy

```math
\frac{\partial}{\partial t} \rho e_\text{int} = - \nabla \cdot((\rho e_\text{int} + p) \boldsymbol{u} + \boldsymbol{F}_R) + (\nabla p) \cdot \boldsymbol{u} .
```
The ``\nabla \cdot((\rho e_\text{int} + p) \boldsymbol{u}`` term is discretized the same as in total energy:

```math
D_h[ (\rho e_\text{int} + p) (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v))] +
- D^c_v \left[ WI^f(J,\rho) \,  \tilde{\boldsymbol{u}} \, I^f \left(\frac{\rho e_\text{int} + p}{\rho} \right)
  + \boldsymbol{F}_R
\right] .
```
The ``(\nabla p) \cdot \boldsymbol{u}`` term is discretized as
```math
I^c( G^f_v(p) \cdot \boldsymbol{u}_v) + G_h(p) \cdot \boldsymbol{u}_h .
```
!!! todo
    We will need to add ``\nabla_h \cdot u_v`` and ``\nabla_v \cdot u_h`` terms, as they will be non-zero in the presence of topography.

Currently the central reconstruction
```math
- D^c_v \left[ WI^f(J,\rho) \,  \tilde{\boldsymbol{u}} \, I^f \left(\frac{\rho e_\text{int} + p}{\rho} \right) + I^c( G^f_v(p) \cdot \boldsymbol{u}_v) \right]
```
is treated implicitly.

!!! todo
    This should be updated so that the upwinded term
    ```math
    - D^c_v\left[WI^f(J, \rho) U^f\left(\boldsymbol{u}_v, \frac{\rho e_\text{int} + p}{\rho} \right)\right]
    ```
    is treated implicitly.


### Scalars

For an arbitrary scalar ``\chi``, the density-weighted scalar ``\rho\chi`` follows the continuity equation

```math
\frac{\partial}{\partial t} \rho \chi = - \nabla \cdot(\rho \chi \boldsymbol{u}) .
```

This is discretized using the following
```math
\frac{\partial}{\partial t} \rho \chi \approx
- D_h[ \rho \chi (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v))]
- D^c_v \left[ WI^f(J,\rho) \, U^f\left( \tilde{\boldsymbol{u}},  \frac{\rho \chi}{\rho} \right) \right].
```
Currently the central reconstruction
```math
- D^c_v \left[ WI^f(J,\rho) \, \tilde{\boldsymbol{u}} \, I^f\left( \frac{\rho \chi}{\rho} \right) \right]
```
is treated implicitly.

!!! todo
    The Jacobian computation should be updated so that the upwinded term
    ```math
    - D^c_v\left[WI^f(J, \rho) U^f\left(I^f(\boldsymbol{u}_h) + \boldsymbol{u}_v, \frac{\rho \chi}{\rho} \right) \right]
    ```
    is treated implicitly.
