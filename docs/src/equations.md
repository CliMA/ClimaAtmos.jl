# Equations

!!! note

    This follows what is _currently_ implemented in `examples`: it should be kept up-to-date as code is modified. If you think something _should_ be changed (but hasn't been), please add a note.

This describes the ClimaAtmos model equations and its discretizations. Where possible, we use a coordinate invariant form: the ClimaCore operators generally handle the conversions between bases internally.

## Prognostic variables

  - ``\rho``: _density_ in kg/m³. This is discretized at cell centers.
  - ``\boldsymbol{u}`` _velocity_, a vector in m/s. This is discretized via ``\boldsymbol{u} = \boldsymbol{u}_h + \boldsymbol{u}_v`` where
      + ``\boldsymbol{u}_h = u_1 \boldsymbol{e}^1 + u_2 \boldsymbol{e}^2`` is the projection onto horizontal covariant components (covariance here means with respect to the reference element), stored at cell centers.
      + ``\boldsymbol{u}_v = u_3 \boldsymbol{e}^3`` is the projection onto the vertical covariant components, stored at cell faces.
  - _energy_, stored at cell centers; can be either:
      + ``\rho e``: _total energy_ in J/m³
      + ``\rho e_\text{int}``: _internal energy_ in J/m³
  - ``\rho \chi``: _other conserved scalars_ (moisture, tracers, etc), again stored at cell centers.

## Operators

We make use of the following operators

### Reconstruction

  - ``I^c`` is the [face-to-center reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateF2C) (arithmetic mean)
  - ``I^f`` is the [center-to-face reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.InterpolateC2F) (arithmetic mean)
  - ``WI^f`` is the [center-to-face weighted reconstruction operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeightedInterpolateC2F)
      + ``WI^f(J, x) = I^f(J*x) / I^f(J)``, where ``J`` is the value of the Jacobian for use in the weighted interpolation operator
  - ``U^f`` is the [1st or 3rd-order center-to-face upwind product operator](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F) # fix link

### Differential operators

  - ``\hat{\mathcal{D}}_h`` is the [discrete horizontal spectral weak divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakDivergence).
  - ``\mathcal{D}^c_v`` is the [face-to-center vertical divergence](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.DivergenceF2C).

!!! todo

    Add vertical diffusive tendencies (including surface fluxes)

  - ``\mathcal{G}_h`` is the [discrete horizontal spectral gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Gradient).
  - ``\mathcal{G}^f_v`` is the [center-to-face vertical gradient](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.GradientC2F).
      + the gradient is set to 0 at the top and bottom boundaries.
  - ``\mathcal{C}_h`` is the [curl components involving horizontal derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.Curl)
      + ``\mathcal{C}_h[\boldsymbol{u}_h]`` returns a vector with only vertical _contravariant_ components.
      + ``\mathcal{C}_h[\boldsymbol{u}_v]`` returns a vector with only horizontal _contravariant_ components.
  - ``\hat{\mathcal{C}}_h`` is the [weak curl components involving horizontal derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.WeakCurl)
  - ``\mathcal{C}^f_v`` is the [center-to-face curl involving vertical derivatives](https://clima.github.io/ClimaCore.jl/stable/operators/#ClimaCore.Operators.CurlC2F).
      + ``\mathcal{C}^f_v[\boldsymbol{u}_h]`` returns a vector with only a horizontal _contravariant_ component.
      + the curl is set to 0 at the top and bottom boundaries.
          * We need to clarify how best to handle this.

### Projection

  - ``\mathcal{P}`` is the [direct stiffness summation (DSS) operation](https://clima.github.io/ClimaCore.jl/stable/operators/#DSS), which computes the projection onto the continuous spectral element basis.

## Auxiliary and derived quantities

  - ``\boldsymbol{\Omega}`` is the planetary angular velocity. We use either:

      + a _shallow atmosphere_ approximation, with

        ```math
        \boldsymbol{\Omega} = \Omega \sin(\phi) \boldsymbol{e}^v
        ```

        where ``\phi`` is latitude, and ``\Omega`` is the planetary rotation rate in rads/sec (for Earth, ``7.29212 \times 10^{-5} s^{-1}``) and ``\boldsymbol{e}^v`` is the unit radial basis vector. This implies that the horizontal contravariant component ``\boldsymbol{\Omega}^h`` is zero.

      + a _deep atmosphere_, with

        ```math
        \boldsymbol{\Omega} = (0, 0, \Omega)
        ```

        i.e. aligned with Earth's rotational axis.

  - ``\tilde{\boldsymbol{u}}`` is the mass-weighted reconstruction of velocity at the interfaces:
    by interpolation of contravariant components

    ```math
    ```

  - ``\bar{\boldsymbol{u}}`` is the reconstruction of velocity at cell-centers, carried out by linear interpolation of the covariant vertical component:

    ```math
    \bar{\boldsymbol{u}} = \boldsymbol{u}_h + I_{c}(\boldsymbol{u}_v)
    ```

  - ``\Phi = g z`` is the geopotential, where ``g`` is the gravitational acceleration rate and ``z`` is altitude above the mean sea level.

  - ``K = \tfrac{1}{2} \|\boldsymbol{u}\|^2`` is the specific kinetic energy (J/kg), reconstructed at cell centers by

    ```math
    K = \tfrac{1}{2} (\boldsymbol{u}_{h} \cdot \boldsymbol{u}_{h} + 2 \boldsymbol{u}_{h} \cdot I_{c} (\boldsymbol{u}_{v}) + I_{c}(\boldsymbol{u}_{v} \cdot \boldsymbol{u}_{v})),
    ```

    where ``\boldsymbol{u}_{h}`` is defined on cell-centers, ``\boldsymbol{u}_{v}`` is defined on cell-faces, and ``I_{c} (\boldsymbol{u}_{v})`` is interpolated using covariant components.

  - ``p`` is air pressure, derived from the thermodynamic state, reconstructed at cell centers.

  - ``\Pi = (\frac{p}{p_0})^{\frac{R_d}{c_{pd}}}`` is the Exner function evaluated with dry-air constants.

  - ``\boldsymbol{F}_R`` are the radiative fluxes: these are assumed to align vertically (i.e. the horizontal contravariant components are zero), and are constructed at cell faces from [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl).

  - ``\nu_u``, ``\nu_h``, and ``\nu_\chi`` are hyperdiffusion coefficients, and ``c`` is the divergence damping factor.

  - No-flux boundary conditions are enforced by requiring the third contravariant component of the face-valued velocity at the boundary, ``\boldsymbol{\tilde{u}}^{v}``, to be zero. The vertical covariant velocity component is computed as

    ```math
    \tilde{u}_{v} = \tfrac{-(u_{1}g^{31} + u_{2}g^{32})}{g^{33}}.
    ```

## Equations and discretizations

### Mass

Follows the continuity equation

```math
\frac{\partial}{\partial t} \rho = - \nabla \cdot(\rho \boldsymbol{u}) + \rho \mathcal{S}_{qt}.
```

This is discretized using the following

```math
\frac{\partial}{\partial t} \rho
= - \hat{\mathcal{D}}_h[ \rho \bar{\boldsymbol{u}}] - \mathcal{D}^c_v \left[WI^f( J, \rho) \tilde{\boldsymbol{u}} \right] + \rho \mathcal{S}_{qt}
```

with the

```math
-\mathcal{D}^c_v[WI^f(J, \rho) \boldsymbol{u}_v]
```

term treated implicitly (check this)

### Momentum

Uses the advective form equation

```math
\frac{\partial}{\partial t} \boldsymbol{u}  = - (2 \boldsymbol{\Omega} + \nabla \times \boldsymbol{u}) \times \boldsymbol{u} - c_{pd} (\theta_v - \theta_{v, r}) \nabla_h \Pi  - \nabla_h [(\Phi - \Phi_r) + K].
```

Here, we use the Exner function to compute pressure gradients and are subtracting a hydrostatic reference state

```math
- \frac{1}{\rho} \nabla p = - c_{pd} \theta_v \Pi
```

where ``\theta_v`` is the virtual potential temperature. ``\theta_{v,r} = T_r / \Pi`` is a reference virtual potential temperature (with reference temperature ``T_r``), and

```math
\Phi_r = -c_{pd} \left[ T_\text{min} \log(\Pi) + \frac{(T_\text{sfc} - T_\text{min})}{n_s} (\Pi^{n_s} - 1) \right],
```

is a reference geopotential, which satisfies the hydrostatic balance equation $c_{pd} \theta_{v,r} \nabla \Pi + \nabla \Phi_r = 0$ for any $\Pi$.
We use the reference temperature profile ``T_r = T_\text{min} + (T_\text{sfc} - T_\text{min}) \Pi^{n_s}``, with constants ``T_\text{min} = 215\,K``, ``T_\text{sfc}= 288\,K``, and ``n_s = 7``.

#### Horizontal momentum

By breaking the curl and cross product terms into horizontal and vertical contributions, and removing zero terms (e.g. ``\nabla_v \times \boldsymbol{u}_v = 0``), we obtain

```math
\frac{\partial}{\partial t} \boldsymbol{u}_h  =
  - (2 \boldsymbol{\Omega}^h + \nabla_v \times \boldsymbol{u}_h +  \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v
  - (2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h
  - c_{pd} (\theta_v - \theta_{v, r}) \nabla_h \Pi  - \nabla_h [(\Phi - \Phi_r) + K],
```

where ``\boldsymbol{u}^h`` and ``\boldsymbol{u}^v`` are the horizontal and vertical _contravariant_ vectors.

The effect of topography is accounted for through the computation of the contravariant velocity components (projections from the covariant velocity representation) prior to computing the cross-product contributions.

This is stabilized with the addition of 4th-order vector hyperviscosity

```math
-\nu_u \, \nabla_h^2 (\nabla_h^2(\boldsymbol{\overline{u}})),
```

projected onto the first two contravariant directions, where ``\nabla_{h}^2(\boldsymbol{v})`` is the horizontal vector Laplacian. For grid scale hyperdiffusion, ``\boldsymbol{v}`` is identical to ``\boldsymbol{\overline{u}}``, the cell-center valued velocity vector.

```math
\nabla_h^2(\boldsymbol{v}) = \nabla_h(\nabla_{h} \cdot \boldsymbol{v}) - \nabla_{h} \times (\nabla_{h} \times \boldsymbol{v}).
```

The ``(2 \boldsymbol{\Omega}^h + \nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v`` term is discretized as:

```math
\frac{I^c\{(2 \boldsymbol{\Omega}^h + \mathcal{C}^f_v[\boldsymbol{u}_h] + \mathcal{C}_h[\boldsymbol{u}_v]) \times (I^f(\rho J)\tilde{\boldsymbol{u}}^v)\}}{\rho J}
```

where

```math
\omega^{h} = (\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v)
```

The ``(2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h`` term is discretized as

```math
(2 \boldsymbol{\Omega}^v + \mathcal{C}_h[\boldsymbol{u}_h]) \times \boldsymbol{u}^h
```

and the ``c_{pd} (\theta_v - \theta_{v,r}) \nabla_h \Pi + \nabla_h (\Phi - \Phi_r + K)`` term is discretized as

```math
c_{pd} (\theta_v - \theta_{v,r}) \mathcal{G}_h[\Pi] + \mathcal{G}_h[\Phi - \Phi_r + K] ,
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
  - (2 \boldsymbol{\Omega}^h + \nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^h
  -c_{pd} (\theta_v - \theta_{v, r}) \nabla_v \Pi  - \nabla_v [(\Phi - \Phi_r)].
```

The ``(2 \boldsymbol{\Omega}^h + \nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^h`` term is discretized as

```math
(2 \boldsymbol{\Omega}^h + \mathcal{C}^f_v[\boldsymbol{u}_h] + \mathcal{C}_h[\boldsymbol{u}_v]) \times I^f(\boldsymbol{u}^h) ,
```

The ``\nabla_v K`` term is discretized as

```math
\mathcal{G}^f_v[K],
```

The ``c_{pd} (\theta_v - \theta_{v,r}) \nabla_v \Pi + \nabla_v (\Phi - \Phi_r)`` term is discretized as

```math
I^f[c_{pd} (\theta_v - \theta_{v, r} ) ] \mathcal{G}^f_v[\Pi] - \mathcal{G}^f_v[\Phi - \Phi_r],
```

and is treated implicitly.

This is stabilized with the addition of 4th-order vector hyperviscosity

```math
-\nu_u \, \nabla_h^2 (\nabla_h^2(\boldsymbol{\overline{u}})),
```

projected onto the third contravariant direction.

### Total energy

```math
\frac{\partial}{\partial t} \rho e = - \nabla \cdot((\rho e + p) \boldsymbol{u} + \boldsymbol{F}_R) + \rho \mathcal{S}_{e},
```

which is stabilized with the addition of a 4th-order hyperdiffusion term on total enthalpy:

```math
- \nu_h \nabla \cdot \left( \rho \nabla^3 \left(\frac{ρe + p}{ρ} \right)\right)
```

This is discretized using

```math
\frac{\partial}{\partial t} \rho e \approx
- \hat{\mathcal{D}}_h[ (\rho e + p) \bar{\boldsymbol{u}} ]
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
\frac{\partial}{\partial t} \rho \chi = - \nabla \cdot(\rho \chi \boldsymbol{u}) + \rho \mathcal{S}_{\chi}.
```

This is stabilized with the addition of a 4th-order hyperdiffusion term

```math
- \nu_\chi \nabla \cdot(\rho \nabla^3(\chi))
```

This is discretized using

```math
\frac{\partial}{\partial t} \rho \chi \approx
- \hat{\mathcal{D}}_h[ \rho \chi \bar{\boldsymbol{u}}]
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

### Hyperdiffusion coefficient and its stability limit

The vorticity hyperviscosity coefficient scales with the mean nodal distance ``h`` as ``\nu_4 = c_4 \, h^3`` ([Lauritzen2018](@cite)), where ``c_4`` is `vorticity_hyperdiffusion_coefficient`.
The scalar hyperdiffusivity is ``\nu_4 / \mathrm{Pr}`` and the divergence damping applies the `divergence_damping_factor` to ``\nu_4``.

When the hyperdiffusion tendency is integrated explicitly, its stability limit is

```math
\Delta t_\mathrm{limit} = \frac{C \, h}{F \, \beta^4 \, c_4},
```

the largest timestep for which ``-\Delta t \, F \, \nu_4 \, \rho(\nabla^4)``, the scaled extreme eigenvalue of the operator with the largest coefficient factor, lies inside the integrator's real-axis stability interval ``[-C, 0]``.
The three factors are derived independently.

The coefficient factor ``F = \max(\text{divergence damping factor}, 1/\mathrm{Pr}, 1)`` is the largest of the divergent-momentum (``\times`` divergence damping factor), scalar (``\times 1/\mathrm{Pr}``), and rotational (``\times 1``) coefficient factors.

The grid factor ``\beta`` is defined by ``\rho(\nabla^4) = (\beta/h)^4``, the largest eigenvalue of the discrete horizontal biharmonic, and is measured on the configured grid once at simulation construction.
The DSS-assembled scalar Laplacian ``\hat{\mathcal{D}}_h \circ \mathcal{G}_h`` is self-adjoint and negative semi-definite in the mass inner product, so its square, the assembled scalar biharmonic, is symmetric positive semi-definite, and a Lanczos iteration with full reorthogonalization in that inner product brackets ``\rho(\nabla^4)`` in ``[\theta, \theta + r]`` once the iteration has resolved the dominant eigenpair, where ``\theta`` is the Rayleigh quotient of the top Ritz vector and ``r`` its residual norm.
``\theta \le \rho(\nabla^4)`` holds unconditionally, and the residual bound certifies an eigenvalue within ``r`` of ``\theta``; that certified eigenvalue is the spectral radius provided the deterministic start vector is not orthogonal to the dominant eigenspace, an assumption validated on the degenerate uniform box, whose repeated top eigenvalue the measurement recovers exactly.
The limit uses the certified upper end with a margin ``\delta`` on the spectral radius, ``\beta = \left((1 + \delta)(\theta + r)\right)^{1/4} h`` with ``\delta = 0.01``.
When the bracket relative width ``r/\theta`` exceeds 1% at the iteration budget (25), construction fails if `hyperdiffusion_dt_safety_factor` is set; with the default it emits a warning and skips the stability-limit warning.
The measurement costs one hyperdiffusion tendency evaluation per iteration and applies to any polynomial degree and any 2D spectral-element grid.
Horizontally-warped meshes, such as the equiangular cubed sphere, are measured exactly; terrain-following vertical warping of the extruded grid is outside this horizontal measurement.

An analytic upper bound ``\beta \le \beta_\mathrm{op}(p) \, M``, the uniform-grid operator factor composed with the peak metric factor, is retained here for intuition; it is exact on a uniform grid (``M = 1``) and conservative on an anisotropic grid, where it overestimates the assembled operator's spectral radius.
The uniform-grid factor ``\beta_\mathrm{op}(p)`` is the measured factor of a uniform, periodic, degree-``p`` grid:

| ``p``                 | 2     | 3     | 4     | 5     | 6     | 7     |
|:--------------------- |:----- |:----- |:----- |:----- |:----- |:----- |
| ``\beta_\mathrm{op}`` | 3.464 | 4.064 | 4.787 | 5.600 | 6.453 | 7.325 |

At degree 3 this is consistent with the 1D grid-scale wavenumber ``\kappa h = 1.8257`` of the degree-3 spectral-element gradient, the spectral radius of the 1D operator in mean-nodal-distance units: the 2D biharmonic corner mode contributes ``(\lambda_x + \lambda_y)^2 = 4\,\lambda_{1D}^2``, a factor ``\sqrt2`` on ``\beta``, with the remainder from the DSS-assembled discrete operator.
The metric factor

```math
M = \sqrt{\max_\mathrm{grid}\left(g^{11} + g^{22} + 2|g^{12}|\right) \Big/ 2\left(\tfrac{2}{p\,h}\right)^2}
```

is the largest tensor-product corner mode of the contravariant metric relative to a uniform grid; it is ``1`` on a Cartesian box and ``\approx 1.51`` on the equiangular cubed sphere at ``h_\mathrm{elem} = 6``.
On that sphere the measured factor is ``\beta \approx 5.19``, which the factorized bound ``\beta_\mathrm{op} \, M \approx 6.14`` overestimates by about 18%, about a factor of 2 in the limit.
The degree-3 uniform periodic box value ``\beta_\mathrm{op}(3) = 4.0637`` serves as the regression anchor of the measurement in the unit tests.

The integrator constant ``C`` is the real-axis stability bound on the negative-real biharmonic spectrum.
The coefficient reduction uses ``C = 2``, the forward-Euler real-axis stability interval ``[-2, 0]``, a conservative bound contained in the interval of the default ARS343 explicit tableau.
The hyperdiffusion tendency is integrated with the explicit tableau of the time integrator, so the warning uses ``C = 2.7853``, the real-axis stability bound of the default ARS343 explicit tableau, whose stability polynomial matches that of RK4.

The formula reproduces two measured stability limits with no tuned constant, at ``c_4 = 0.1857``, ``\mathrm{Pr} = 0.2``, and divergence damping factor ``5``: the forward-Euler limit is ``\approx 0.89`` s at ``h = 113`` m on a degree-3 dry density-current box (measured stable at ``0.9`` s, unstable at ``1.2`` s), and ``\approx 1500`` s on the dry baroclinic-wave cubed sphere at ``h_\mathrm{elem} = 6`` (measured bracket ``[720, 1600]`` s).
The measurements integrated the hyperdiffusion tendency once per step with a forward-Euler update, for which the ``C = 2`` bound applies.

The `hyperdiffusion_dt_safety_factor` option reduces the coefficient using the forward-Euler bound (``C = 2``).
When set to a positive value ``S``, the vorticity coefficient is reduced to

```math
\nu_4 = \min\!\left(c_4 h^3, \; \frac{2 \, h^4}{F \, \beta^4 \, S \, \Delta t}\right),
```

so the hyperdiffusion is explicitly stable for ``S \, \Delta t``; the divergent and scalar coefficients scale with it.
This is required when ``\Delta t`` exceeds ``\Delta t_\mathrm{limit}``.
With `~` (the default) the coefficient is unchanged and a warning is emitted when ``\Delta t`` exceeds ``\Delta t_\mathrm{limit}``.
The limit applies to 2D spectral-element horizontal spaces; column and plane configurations skip the warning, and a positive `hyperdiffusion_dt_safety_factor` raises an error at construction.
