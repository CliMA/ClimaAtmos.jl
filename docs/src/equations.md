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

* ``I^c`` is the [face-to-center reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateF2C) (arithmetic mean)
* ``I^f`` is the [center-to-face reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateC2F) (arithmetic mean)
* ``WI^f`` is the [center-to-face weighted reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeightedInterpolateC2F)
  - ``WI^f(J, x) = I^f(J*x) / I^f(J)``, where ``J`` is the value of the Jacobian for use in the weighted interpolation operator
* ``U^f`` is the [1st or 3rd-order center-to-face upwind product operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F) # fix link

### Differential operators

- ``\mathcal{D}_h`` is the [discrete horizontal spectral divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Divergence).
- ``\hat{\mathcal{D}}_h`` is the [discrete horizontal spectral weak divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakDivergence).
- ``\mathcal{D}^c_v`` is the [face-to-center vertical divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C).
!!! todo
    Add vertical diffusive tendencies (including surface fluxes)

- ``\mathcal{G}_h`` is the [discrete horizontal spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient).
- ``\mathcal{G}^f_v`` is the [center-to-face vertical gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.GradientC2F).
  - the gradient is set to 0 at the top and bottom boundaries.
- ``\mathcal{C}_h`` is the [curl components involving horizontal derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Curl)
  - ``\mathcal{C}_h[\boldsymbol{u}_h]`` returns a vector with only vertical _contravariant_ components.
  - ``\mathcal{C}_h[\boldsymbol{u}_v]`` returns a vector with only horizontal _contravariant_ components.
- ``\hat{\mathcal{C}}_h`` is the [weak curl components involving horizontal derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakCurl)
- ``\mathcal{C}^f_v`` is the [center-to-face curl involving vertical derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.CurlC2F).
  - ``\mathcal{C}^f_v[\boldsymbol{u}_h]`` returns a vector with only a horizontal _contravariant_ component.
  - the curl is set to 0 at the top and bottom boundaries.
    - We need to clarify how best to handle this.

### Projection

- ``\mathcal{P}`` is the [direct stiffness summation (DSS) operation](https://clima.github.io/ClimaCore.jl/stable/operators/#DSS), which computes the projection onto the continuous spectral element basis.

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
* ``\bar{\boldsymbol{u}}`` is the reconstruction of velocity at cell-centers, carried out by linear interpolation of the covariant vertical component:
  ```math
  \bar{\boldsymbol{u}} = \boldsymbol{u}_h + I_{c}(\boldsymbol{u}_v)
  ```

* ``\Phi = g z`` is the geopotential, where ``g`` is the gravitational acceleration rate and ``z`` is altitude above the mean sea level.
* ``\boldsymbol{b}`` is the reduced gravitational acceleration
  ```math
  \boldsymbol{b} = - \frac{\rho - \rho_{\text{ref}}}{\rho} \nabla \Phi
  ```
* ``\rho_{\text{ref}}`` is the reference state density
* ``K = \tfrac{1}{2} \|\boldsymbol{u}\|^2 `` is the specific kinetic energy (J/kg), reconstructed at cell centers by
  ```math
  K = \tfrac{1}{2} (\boldsymbol{u}_{h} \cdot \boldsymbol{u}_{h} + 2 \boldsymbol{u}_{h} \cdot I_{c} (\boldsymbol{u}_{v}) + I_{c}(\boldsymbol{u}_{v} \cdot \boldsymbol{u}_{v})),
  ```
  where ``\boldsymbol{u}_{h}`` is defined on cell-centers, ``\boldsymbol{u}_{v}`` is defined on cell-faces, and ``I_{c} (\boldsymbol{u}_{v})`` is interpolated using covariant components.

* ``p`` is air pressure, derived from the thermodynamic state, reconstructed at cell centers.
* ``p_{\text{ref}}`` is the reference state pressure. It is related to the reference state density by analytical hydrostatic balance: ``\nabla p_{\text{ref}} = - \rho_{\text{ref}} \nabla \Phi``.
* ``\boldsymbol{F}_R`` are the radiative fluxes: these are assumed to align vertically (i.e. the horizontal contravariant components are zero), and are constructed at cell faces from [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl).

* ``\nu_u``, ``\nu_h``, and ``\nu_\chi`` are hyperdiffusion coefficients, and ``c`` is the divergence damping factor.

* No-flux boundary conditions are enforced by requiring the third contravariant component of the face-valued velocity at the boundary, ``\boldsymbol{\tilde{u}}^{v}``, to be zero. The vertical covariant velocity component is computed as
  ```math
  \tilde{u}_{v} = \tfrac{-(u_{1}g^{31} + u_{2}g^{32})}{g^{33}}.
  ```

## Equations and discretizations

### Mass

Follows the continuity equation
```math
\frac{\partial}{\partial t} \rho = - \nabla \cdot(\rho \boldsymbol{u}) .
```

This is discretized using the following
```math
\frac{\partial}{\partial t} \rho
= - \mathcal{D}_h[ \rho \bar{\boldsymbol{u}}] - \mathcal{D}^c_v \left[WI^f( J, \rho) \tilde{\boldsymbol{u}} \right]
```

with the
```math
-\mathcal{D}^c_v[WI^f(J, \rho) \boldsymbol{u}_v]
```
term treated implicitly (check this)


### Momentum

Uses the advective form equation
```math
\frac{\partial}{\partial t} \boldsymbol{u}  = - (2 \boldsymbol{\Omega} + \nabla \times \boldsymbol{u}) \times \boldsymbol{u} - \frac{1}{\rho} \nabla (p - p_{\text{ref}})  + \boldsymbol{b} - \nabla K
```

#### Horizontal momentum

By breaking the curl and cross product terms into horizontal and vertical contributions, and removing zero terms (e.g. ``\nabla_v  \times \boldsymbol{u}_v = 0``), we obtain

```math
\frac{\partial}{\partial t} \boldsymbol{u}_h  =
  - (\nabla_v \times \boldsymbol{u}_h +  \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v
  - (2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h
  - \frac{1}{\rho} \nabla_h (p - p_{\text{ref}})  - \nabla_h (\Phi + K),
```
where ``\boldsymbol{u}^h`` and ``\boldsymbol{u}^v`` are the horizontal and vertical _contravariant_ vectors. The effect of topography is accounted for through the computation of the contravariant velocity components (projections from the covariant velocity representation) prior to computing the cross-product contributions.

This is stabilized with the addition of 4th-order vector hyperviscosity
```math
-\nu_u \, \nabla_h^2 (\nabla_h^2(\boldsymbol{\overbar{u}})),
```
projected onto the first two contravariant directions, where ``\nabla_{h}^2(\boldsymbol{v})`` is the horizontal vector Laplacian. For grid scale hyperdiffusion, ``\boldsymbol{v}`` is identical to ``\boldsymbol{\overbar{u}}``, the cell-center valued velocity vector. 
```math
\nabla_h^2(\boldsymbol{v}) = \nabla_h(\nabla_{h} \cdot \boldsymbol{v}) - \nabla_{h} \times (\nabla_{h} \times \boldsymbol{v}).
```

The ``(\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v`` term is discretized as:
```math
\frac{I^c((\mathcal{C}^f_v[\boldsymbol{u}_h] + \mathcal{C}_h[\boldsymbol{u}_v]) \times (I^f(\rho J)\tilde{\boldsymbol{u}}^v))}{\rho J}
```
where
```math
\omega^{h} = (\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v)
```

The ``(2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h`` term is discretized as
```math
(2 \boldsymbol{\Omega}^v + \mathcal{C}_h[\boldsymbol{u}_h]) \times \boldsymbol{u}^h
```
and the ``\frac{1}{\rho} \nabla_h (p - p_h)  + \nabla_h (\Phi + K)`` as
```math
\frac{1}{\rho} \mathcal{G}_h[p - p_{\text{ref}}] + \mathcal{G}_h[\Phi + K] ,
```
where all these terms are treated explicitly.

The hyperviscosity term is
```math
- \nu_u \left\{ c \, \hat{\mathcal{G}}_h ( \mathcal{D}(\boldsymbol{\psi}_h) ) - \hat{\mathcal{C}}_h( \mathcal{C}_h( \boldsymbol{\psi}_h )) \right\}
```
where
```math
\boldsymbol{\psi}_h = \mathcal{P} \left[ \hat{\mathcal{G}}_h ( \mathcal{D}(\boldsymbol{u}_h) ) - \hat{\mathcal{C}}_h( \mathcal{C}_h( \boldsymbol{u}_h )) \right]
```

#### Vertical momentum
Similarly for vertical velocity
```math
\frac{\partial}{\partial t} \boldsymbol{u}_v  =
  - (\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^h
  - \frac{1}{\rho} \nabla_v (p - p_{\text{ref}}) - \frac{\rho - \rho_{\text{ref}}}{\rho} \nabla_v \Phi - \nabla_v K .
```

The ``(\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^h`` term is discretized as
```math
(\mathcal{C}^f_v[\boldsymbol{u}_h] + \mathcal{C}_h[\boldsymbol{u}_v]) \times I^f(\boldsymbol{u}^h) ,
```
and the ``\frac{1}{\rho} \nabla_v (p - p_{\text{ref}}) - \frac{\rho - \rho_{\text{ref}}}{\rho} \nabla_v \Phi - \nabla_v K`` term as
```math
\frac{1}{I^f(\rho)} \mathcal{G}^f_v[p - p_{\text{ref}}] - \frac{I^f(\rho - \rho_{\text{ref}})}{I^f(\rho)} \mathcal{G}^f_v[\Phi] - \mathcal{G}^f_v[K] ,
```
with the latter treated implicitly.

This is stabilized with the addition of 4th-order vector hyperviscosity
```math
-\nu_u \, \nabla_h^2 (\nabla_h^2(\boldsymbol{\overbar{u}})),
```
projected onto the third contravariant direction.

### Total energy

```math
\frac{\partial}{\partial t} \rho e = - \nabla \cdot((\rho e + p) \boldsymbol{u} + \boldsymbol{F}_R),
```
which is stabilized with the addition of a 4th-order hyperdiffusion term on total enthalpy:
```math
- \nu_h \nabla \cdot \left( \rho \nabla^3 \left(\frac{ρe + p}{ρ} \right)\right)
```

This is discretized using
```math
\frac{\partial}{\partial t} \rho e \approx
- \mathcal{D}_h[ (\rho e + p) \bar{\boldsymbol{u}} ]
- \mathcal{D}^c_v \left[ WI^f(J,\rho) \,  \tilde{\boldsymbol{u}} \, I^f \left(\frac{\rho e + p}{\rho} \right)
  + \boldsymbol{F}_R \right] - \nu_h \hat{\mathcal{D}}_h( \rho \mathcal{G}_h(\psi) ).
```
where
```math
\psi = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h \left(\frac{ρe + p}{ρ} \right)\right) \right]
```

Currently the central reconstruction
```math
- \mathcal{D}^c_v \left[ WI^f(J,\rho) \,  \tilde{\boldsymbol{u}} \, I^f \left(\frac{\rho e + p}{\rho} \right) \right]
```
is treated implicitly.

!!! todo
    The Jacobian computation should be updated so that the upwinded term
    ```math
    - \mathcal{D}^c_v\left[WI^f(J, \rho) U^f\left(\boldsymbol{u}_v, \frac{\rho e + p}{\rho} \right)\right]
    ```
    is treated implicitly.

### Scalars

For an arbitrary scalar ``\chi``, the density-weighted scalar ``\rho\chi`` follows the continuity equation

```math
\frac{\partial}{\partial t} \rho \chi = - \nabla \cdot(\rho \chi \boldsymbol{u}).
```
This is stabilized with the addition of a 4th-order hyperdiffusion term
```math
- \nu_\chi \nabla \cdot(\rho \nabla^3(\chi))
```

This is discretized using
```math
\frac{\partial}{\partial t} \rho \chi \approx
- \mathcal{D}_h[ \rho \chi \bar{\boldsymbol{u}}]
- \mathcal{D}^c_v \left[ WI^f(J,\rho) \, U^f\left( \tilde{\boldsymbol{u}},  \frac{\rho \chi}{\rho} \right) \right]
- \nu_\chi \hat{\mathcal{D}}_h ( \rho \, \mathcal{G}_h (\psi) )
```
where
```math
\psi = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h \left( \frac{\rho \chi}{\rho} \right)\right) \right]
```

Currently the central reconstruction
```math
- \mathcal{D}^c_v \left[ WI^f(J,\rho) \, \tilde{\boldsymbol{u}} \, I^f\left( \frac{\rho \chi}{\rho} \right) \right]
```
is treated implicitly.

!!! todo
    The Jacobian computation should be updated so that the upwinded term
    ```math
    - \mathcal{D}^c_v\left[WI^f(J, \rho) U^f\left(I^f(\boldsymbol{u}_h) + \boldsymbol{u}_v, \frac{\rho \chi}{\rho} \right) \right]
    ```
    is treated implicitly.
