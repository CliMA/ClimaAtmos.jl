# Equations

!!! note

    This follows what is _currently_ implemented in `examples`: it should be kept up-to-date as code is modified. If you think something _should_ be changed (but hasn't been), please add a note.

This describes the ClimaAtmos model equations and its discretizations. Where possible, we use a coordinate invariant form: the ClimaCore operators generally handle the conversions between bases internally.



## Prognostic variables

* ``\rho``: _density_ in kg/m³. This is discretized at cell centers.
* ``\boldsymbol{u}`` _velocity_, a vector in m/s. This is discretized via ``\boldsymbol{u} = \boldsymbol{u}_h + \boldsymbol{u}_v`` where
  - ``\boldsymbol{u}_h = u_1 \boldsymbol{e}^1 + u_2 \boldsymbol{e}^2`` is the projection onto horizontal covariant components (covariance here means with respect to the reference element), stored at cell centers.
  - ``\boldsymbol{u}_v = u_3 \boldsymbol{e}^3`` is the projection onto the vertical covariant components, stored at cell faces.
* _energy_, stored at cell centers; can be either:
  - ``\rho e``: _total energy_ in J/m³
  - ``\rho e_\text{int}``: _internal energy_ in J/m³
* ``\rho \chi``: _other conserved scalars_ (moisture, tracers, etc), again stored at cell centers.

## Operators

We make use of the following operators

### Reconstruction

* ``I^c`` is the [face-to-center reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateF2C)
* ``I^f`` is the [center-to-face reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateC2F)
  - Currently this is just the arithmetic mean, but we will need to use a weighted version with stretched vertical grids.
* ``U^f`` is the [3rd-order center-to-face upwind product operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F)
  - We use first-order upwinded reconstruction at the first interior face.

### Differentiation operators

- ``D_h`` is the [discrete horizontal spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Divergence).
- ``D^c_v`` is the [face-to-center vertical divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C).
  - Currently fluxes are set to zero at the top and bottom boundaries: this will need to be modified with the addition of diffusive fluxes.
- ``G_h`` is the [discrete horizontal spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient).
- ``G^f_v`` is the [center-to-face vertical gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.GradientC2F).
  - the gradient is set to 0 at the top and bottom boundaries.
- ``C_h`` is the [horizontal curl](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Curl)
  - ``C_h[\boldsymbol{u}_h]`` returns a vector with only vertical _contravariant_ components.
  - ``C_h[\boldsymbol{u}_v]`` returns a vector with only horizontal _contravariant_ components.
- ``C^f_v`` is the [center-to-face vertical curl](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.CurlC2F).
  - ``C^f_v[\boldsymbol{u}_h]`` returns a vector with only a horizontal _contravariant_ component.
  - the curl is set to 0 at the top and bottom boundaries.
    - We need to clarify how best to handle this.

## Auxiliary and derived quantities

* ``\boldsymbol{\Omega}`` is the planetary angular velocity. We currently use a shallow-atmosphere approximation, with
  ```math
  \boldsymbol{\Omega} = \Omega \sin(\phi) \boldsymbol{e}^v
  ```
  where ``\phi`` is latitude, and ``\Omega`` is the planetary rotation rate in rads/sec (for Earth, ``7.29212 \times 10^{-5}``) and ``\boldsymbol{e}^v`` is the unit radial basis vector. This implies that the horizontal contravariant component ``\boldsymbol{\Omega}^h`` is zero.
* ``\Phi = g z`` is the geopotential, where ``g`` is the gravitational acceleration rate and ``z`` is altitude above the mean sea level.
* ``K = \tfrac{1}{2} \|\boldsymbol{u}\|^2 `` is the specific kinetic energy (J/kg), reconstructed at cell centers by
  ```math
  K = \tfrac{1}{2} \|\boldsymbol{u}_h + I^c(\boldsymbol{u}_v)\|^2.
  ```
  where ``\|\boldsymbol{u}\|^2 = g^{ij} \boldsymbol{u}_i \boldsymbol{u}_j`` and ``g^{ij}`` is the contravariant metric tensor.
* ``p`` is air pressure, derived from the thermodynamic state, reconstructed at cell centers.
* ``\boldsymbol{F}_R`` are the radiative fluxes: these are assumed to align vertically (i.e. the horizontal contravariant components are zero), and are constructed at cell faces from [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl).

## Equations and discretizations

### Mass

Follows the continuity equation
```math
\frac{\partial}{\partial t} \rho = - \nabla \cdot(\rho \boldsymbol{u}) .
```

This is discretized using the following
```math
\frac{\partial}{\partial t} \rho \approx - D_h[ \rho (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v))] - D^c_v[I^f(\rho \boldsymbol{u}_h)) + I^f(\rho) \boldsymbol{u}_v)]
```
with the
```math
-D^c_v[I^f(\rho) \boldsymbol{u}_v)]
```
term treated implicitly.


### Momentum

Uses the advective form equation
```math
\frac{\partial}{\partial t} \boldsymbol{u}  = - (2 \boldsymbol{\Omega} + \nabla \times \boldsymbol{u}) \times \boldsymbol{u} - \frac{1}{\rho} \nabla p  - \nabla(\Phi + K) .
```

#### Horizontal momentum

By breaking the curl and cross product terms into horizontal and vertical contributions, and removing zero terms (e.g. ``\nabla_v  \times \boldsymbol{u}_v = 0``), we obtain

```math
\frac{\partial}{\partial t} \boldsymbol{u}_h  =
  - (\nabla_v \times \boldsymbol{u}_h +  \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v
  - (2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h
  - \frac{1}{\rho} \nabla_h p  - \nabla_h (\Phi + K) ,
```
where ``\boldsymbol{u}^h`` and ``\boldsymbol{u}^v`` are the horizontal and vertical _contravariant_ vectors.

!!! todo
    Without topography, these are equal to their covariant vectors (i.e. ``\boldsymbol{u}^h = \boldsymbol{u}_h``), but these will need to be updated with the addition of topography.

The ``(\nabla_v \times \boldsymbol{u}_h +  \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v`` term is discretized as
```math
I^c((C^f_v[\boldsymbol{u}_h] + C_h[\boldsymbol{u}_v]) \times \boldsymbol{u}^v)
```
The ``(2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h`` term is discretized as
```math
(2 \boldsymbol{\Omega}^v + C_h[\boldsymbol{u}_h]) \times \boldsymbol{u}^h
```
and the ``\frac{1}{\rho} \nabla_h p  + \nabla_h (\Phi + K)`` as
```math
\frac{1}{\rho} G_h[p] + G_h[\Phi + K] ,
```
where all these terms are treated explicitly.

#### Vertical momentum
Similarly for vertical velocity
```math
\frac{\partial}{\partial t} \boldsymbol{u}_v  =
  - (\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^h
  - \frac{1}{\rho} \nabla_v p - \nabla_v(\Phi + K) .
```
The ``(\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^h`` term is discretized as
```math
(C^f_v[\boldsymbol{u}_h] + C_h[\boldsymbol{u}_v]) \times I^f(\boldsymbol{u}_h) ,
```
and the ``\frac{1}{\rho} \nabla_v p + \nabla_v(\Phi + K)`` term as
```math
\frac{1}{I^f(\rho)} G^f_v[p] + G^f[K + \Phi] ,
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
- D^c_v \left[
  I^f(\rho) U^f\left(\boldsymbol{u}_v, \frac{\rho e + p}{\rho} \right)
  + (\rho e + p) I^f(\boldsymbol{u}_h)
  + \boldsymbol{F}_R
  \right] .
```
Currently the central reconstruction
```math
- D^c_v[ I^f(\rho e + p) \boldsymbol{u}_v ]
```
is treated implicitly.

!!! todo
    The Jacobian computation should be updated so that the upwinded term
    ```math
    - D^c_v\left[I^f(\rho) U^f\left(\boldsymbol{u}_v, \frac{\rho e + p}{\rho} \right)\right]
    ```
    is treated implicitly.


### Internal energy

```math
\frac{\partial}{\partial t} \rho e_\text{int} = - \nabla \cdot((\rho e_\text{int} + p) \boldsymbol{u} + \boldsymbol{F}_R) + (\nabla p) \cdot \boldsymbol{u} .
```
The ``\nabla \cdot((\rho e_\text{int} + p) \boldsymbol{u}`` term is discretized the same as in total energy:

```math
D_h[ (\rho e_\text{int} + p) (\boldsymbol{u}_h + I^c(\boldsymbol{u}_v))] +
D^c_v\left[
  I^f(\rho) U^f\left(I^f(\boldsymbol{u}_v, \frac{\rho e_\text{int} + p}{\rho} \right)
  + (\rho e_\text{int} + p)  I^f(\boldsymbol{u}_h)
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
- D^c_v[ I^f(\rho e_\text{int} + p) \boldsymbol{u}_v ] + I^c( G^f_v(p) \cdot \boldsymbol{u}_v)
```
is treated implicitly.

!!! todo
    This should be updated so that the upwinded term
    ```math
    - D^c_v\left[I^f(\rho) U^f\left(\boldsymbol{u}_v, \frac{\rho e_\text{int} + p}{\rho} \right)\right]
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
- D^c_v\left[I^f(\rho) U^f\left(I^f(\boldsymbol{u}_h) + \boldsymbol{u}_v, \frac{\rho \chi}{\rho} \right) \right] .
```
Currently the central reconstruction
```math
- D^c_v[ I^f(\rho \chi) \boldsymbol{u}_v ]
```
is treated implicitly.

!!! todo
    The Jacobian computation should be updated so that the upwinded term
    ```math
    - D^c_v\left[I^f(\rho) U^f\left(I^f(\boldsymbol{u}_h) + \boldsymbol{u}_v, \frac{\rho \chi}{\rho} \right) \right]
    ```
    is treated implicitly.
