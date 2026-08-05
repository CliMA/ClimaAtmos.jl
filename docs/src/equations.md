# Equations

!!! note

    This follows what is currently implemented in `src/prognostic_equations/`: it should be kept up-to-date as code is modified. If you think something _should_ be changed (but hasn't been), please open an issue.

This describes the ClimaAtmos model equations and its discretizations. Where possible, we use a coordinate invariant form: the ClimaCore operators generally handle the conversions between bases internally.

## Prognostic variables

  - ``\rho``: _density_ in kg/m³. This is discretized at cell centers.
  - ``\boldsymbol{u}`` _velocity_, a vector in m/s. This is discretized via ``\boldsymbol{u} = \boldsymbol{u}_h + \boldsymbol{u}_v`` where
      + ``\boldsymbol{u}_h = u_1 \boldsymbol{e}^1 + u_2 \boldsymbol{e}^2`` is the projection onto horizontal covariant components (covariance here means with respect to the reference element), stored at cell centers.
      + ``\boldsymbol{u}_v = u_3 \boldsymbol{e}^3`` is the projection onto the vertical covariant components, stored at cell faces.
  - ``\rho e``: _total energy_ in J/m³, stored at cell centers (the prognostic field is named `ρe_tot` in the code).
  - ``\rho \chi``: _other conserved scalars_ (moisture, tracers, etc), again stored at cell centers.

When prognostic turbulence kinetic energy or PROPHET is enabled, the state additionally carries ``\rho \, \mathrm{tke}`` (and the PROPHET subdomain variables described in the [PROPHET equations](@ref "PROPHET Sub-Grid Scale Equations")); with a slab-ocean surface, a prognostic surface state `Y.sfc` is added.

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
  - ``U^f`` is the 1st-order ([`ClimaCore.Operators.UpwindBiasedProductC2F`](@extref)) or 3rd-order ([`ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F`](@extref)) center-to-face upwind product operator, or the van Leer flux limiter ([`ClimaCore.Operators.LinVanLeerC2F`](@extref)); the van Leer limiter is the default for grid-mean energy and tracer vertical transport (`energy_q_tot_upwinding`, `tracer_upwinding`).

### Differential operators

  - ``\hat{\mathcal{D}}_h`` is the discrete horizontal spectral weak divergence [`ClimaCore.Operators.WeakDivergence`](@extref).
  - ``\mathcal{D}^{split}_h`` is the split (skew-symmetric) horizontal divergence [`ClimaCore.Operators.SplitDivergence`](@extref), ``\mathcal{D}^{split}_h(\rho\boldsymbol{u}, \psi) = \tfrac12 \hat{\mathcal{D}}_h(\rho\boldsymbol{u}\psi) + \tfrac12(\psi\,\hat{\mathcal{D}}_h(\rho\boldsymbol{u}) + \rho\boldsymbol{u}\cdot\mathcal{G}_h\psi)``. The horizontal advective fluxes of energy, moisture, and tracers use this split form (for ``\psi = 1`` it reduces to the weak divergence, so the mass flux below is written with ``\hat{\mathcal{D}}_h``); the horizontal pressure-gradient term is likewise applied in an analogous split form.
  - ``\mathcal{D}^c_v`` is the face-to-center vertical divergence [`ClimaCore.Operators.DivergenceF2C`](@extref).
  - ``\mathcal{G}_h`` is the discrete horizontal spectral gradient [`ClimaCore.Operators.Gradient`](@extref).
  - ``\mathcal{G}^f_v`` is the center-to-face vertical gradient [`ClimaCore.Operators.GradientC2F`](@extref).
      + the gradient is set to 0 at the top and bottom boundaries.
  - ``\mathcal{C}_h`` is the curl components involving horizontal derivatives [`ClimaCore.Operators.Curl`](@extref).
      + ``\mathcal{C}_h[\boldsymbol{u}_h]`` returns a vector with only vertical _contravariant_ components.
      + ``\mathcal{C}_h[\boldsymbol{u}_v]`` returns a vector with only horizontal _contravariant_ components.
  - ``\hat{\mathcal{C}}_h`` is the weak curl components involving horizontal derivatives [`ClimaCore.Operators.WeakCurl`](@extref).
  - ``\mathcal{C}^f_v`` is the center-to-face curl involving vertical derivatives [`ClimaCore.Operators.CurlC2F`](@extref).
      + ``\mathcal{C}^f_v[\boldsymbol{u}_h]`` returns a vector with only a horizontal _contravariant_ component.
      + the curl is set to 0 at the top and bottom boundaries.

### Projection

  - ``\mathcal{P}`` is the [direct stiffness summation (DSS) operation](@extref ClimaCore DSS), which computes the projection onto the continuous spectral element basis.

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

  - ``\tilde{\boldsymbol{u}}`` is the mass-weighted reconstruction of velocity at the interfaces,
    carried out by weighted interpolation of the horizontal components (see `compute_ᶠuₕ³` in
    `src/cache/precomputed_quantities.jl`):

    ```math
    \tilde{\boldsymbol{u}} = WI^f(\rho J, \boldsymbol{u}_h) + \boldsymbol{u}_v
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

  - ``\nu_u``, ``\nu_h``, and ``\nu_\chi`` are hyperdiffusion coefficients, and ``c`` is the divergence damping factor. In the code there are two coefficients: ``\nu_u`` (scaled with the cube of the element width) and ``\nu_h = \nu_\chi = \nu_u/\mathrm{Pr}``; precipitating tracers are additionally rescaled by `tracer_hyperdiffusion_factor`.

  - No-flux boundary conditions are enforced by requiring the third contravariant component ``\boldsymbol{\tilde{u}}^{v}`` of the face-valued velocity at the boundary to be zero. The vertical covariant velocity component is computed as

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
-\mathcal{D}^c_v[WI^f(J, \rho) \tilde{\boldsymbol{u}}]
```

term treated implicitly (the full face velocity ``\tilde{\boldsymbol{u}}``, including
the topographic contribution of ``\boldsymbol{u}_h``, enters the implicit term).

### Momentum

Uses the advective form equation

```math
\frac{\partial}{\partial t} \boldsymbol{u}  = - (2 \boldsymbol{\Omega} + \nabla \times \boldsymbol{u}) \times \boldsymbol{u} - c_{pd} (\theta_v - \theta_{v, r}) \nabla_h \Pi  - \nabla_h [(\Phi - \Phi_r) + K].
```

Here, we use the Exner function to compute pressure gradients and subtract a hydrostatic reference state

```math
- \frac{1}{\rho} \nabla p = - c_{pd} \theta_v \nabla \Pi
```

where ``\theta_v`` is the virtual potential temperature. ``\theta_{v,r} = T_r / \Pi`` is a reference virtual potential temperature (with reference temperature ``T_r``), and

```math
\Phi_r = -c_{pd} \left[ T_\text{min} \log(\Pi) + \frac{(T_\text{sfc} - T_\text{min})}{n_s} (\Pi^{n_s} - 1) \right],
```

is a reference geopotential, which satisfies the hydrostatic balance equation $c_{pd} \theta_{v,r} \nabla \Pi + \nabla \Phi_r = 0$ for any $\Pi$.
We use the reference temperature profile ``T_r = T_\text{min} + (T_\text{sfc} - T_\text{min}) \Pi^{n_s}``, with ``n_s = 7`` and the ClimaParams defaults ``T_\text{min} = 220\,K`` (`temperature_min_reference`) and ``T_\text{sfc} = 290\,K`` (`temperature_surface_reference`).

#### Horizontal momentum

By breaking the curl and cross product terms into horizontal and vertical contributions, and removing zero terms (e.g. ``\nabla_v \times \boldsymbol{u}_v = 0``), we obtain

```math
\frac{\partial}{\partial t} \boldsymbol{u}_h  =
  - (2 \boldsymbol{\Omega}^h + \nabla_v \times \boldsymbol{u}_h +  \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v
  - (2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h
  - c_{pd} (\theta_v - \theta_{v, r}) \nabla_h \Pi  - \nabla_h [(\Phi - \Phi_r) + K],
```

where ``\boldsymbol{u}^h`` and ``\boldsymbol{u}^v`` are the horizontal and vertical _contravariant_ vectors.

Topography enters through the computation of the contravariant velocity components (projections from the covariant velocity representation) before the cross-product contributions.

This is stabilized by adding 4th-order vector hyperviscosity

```math
-\nu_u \, \nabla_h^2 (\nabla_h^2(\boldsymbol{\overline{u}})),
```

projected onto the first two covariant directions, where ``\nabla_{h}^2(\boldsymbol{v})`` is the horizontal vector Laplacian. For grid scale hyperdiffusion, ``\boldsymbol{v}`` is identical to ``\boldsymbol{\overline{u}}``, the cell-center valued velocity vector.

```math
\nabla_h^2(\boldsymbol{v}) = \nabla_h(\nabla_{h} \cdot \boldsymbol{v}) - \nabla_{h} \times (\nabla_{h} \times \boldsymbol{v}).
```

The ``(2 \boldsymbol{\Omega}^h + \nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v`` term is discretized as:

```math
\frac{I^c\{(2 \boldsymbol{\Omega}^h + \mathcal{C}^f_v[\boldsymbol{u}_h] + \hat{\mathcal{C}}_h[\boldsymbol{u}_v]) \times (I^f(\rho J)\tilde{\boldsymbol{u}}^v)\}}{\rho J}
```

where

```math
\omega^{h} = (\nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v)
```

The ``(2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h`` term is discretized as

```math
(2 \boldsymbol{\Omega}^v + \hat{\mathcal{C}}_h[\boldsymbol{u}_h]) \times \boldsymbol{u}^h
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
(2 \boldsymbol{\Omega}^h + \mathcal{C}^f_v[\boldsymbol{u}_h] + \hat{\mathcal{C}}_h[\boldsymbol{u}_v]) \times I^f(\boldsymbol{u}^h) ,
```

The ``\nabla_v K`` term is discretized as

```math
\mathcal{G}^f_v[K],
```

The ``c_{pd} (\theta_v - \theta_{v,r}) \nabla_v \Pi + \nabla_v (\Phi - \Phi_r)`` term is discretized as

```math
I^f[c_{pd} (\theta_v - \theta_{v, r} ) ] \mathcal{G}^f_v[\Pi] + \mathcal{G}^f_v[\Phi - \Phi_r],
```

and is treated implicitly.

This is stabilized by adding 4th-order vector hyperviscosity

```math
-\nu_u \, \nabla_h^2 (\nabla_h^2(\boldsymbol{\overline{u}})),
```

projected onto the third covariant direction after a ``\rho J``-weighted
interpolation to faces.

### Total energy

```math
\frac{\partial}{\partial t} \rho e = - \nabla \cdot((\rho e + p) \boldsymbol{u} + \boldsymbol{F}_R) + \rho \mathcal{S}_{e},
```

which is stabilized by adding a 4th-order hyperdiffusion term on total enthalpy:

```math
- \nu_h \left[ \nabla \cdot \left( \rho \nabla^3 s_d \right)
  + \sum_\mu \nabla \cdot \left( \rho \, (h_\mu + \Phi) \, \nabla^3 q_\mu \right) \right],
```

where the total enthalpy is decomposed into dry static energy ``s_d`` and the
water-species enthalpies ``h_\mu`` (for ``\mu \in \{v, l, i\}``), so that the
``\nabla^4`` operator never acts on a lumped total enthalpy.

This is discretized using

```math
\frac{\partial}{\partial t} \rho e \approx
- \mathcal{D}^{split}_h[ \rho \bar{\boldsymbol{u}}, \tfrac{\rho e + p}{\rho} ]
- \mathcal{D}^c_v \left[ WI^f(J,\rho) \,  \tilde{\boldsymbol{u}} \, I^f \left(\frac{\rho e + p}{\rho} \right) \right]
- \mathcal{D}^c_v \left[ \boldsymbol{F}_R \right]
- \nu_h \left[ \hat{\mathcal{D}}_h( \rho \mathcal{G}_h(\psi_{s_d}) )
  + \sum_\mu \hat{\mathcal{D}}_h( \rho (h_\mu + \Phi) \mathcal{G}_h(\psi_{q_\mu}) ) \right],
```

where

```math
\psi_x = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h (x) \right) \right],
```

and the radiative flux divergence ``-\mathcal{D}^c_v[\boldsymbol{F}_R]`` is
applied separately as an explicit tendency.

The central reconstruction

```math
- \mathcal{D}^c_v \left[ WI^f(J,\rho) \,  \tilde{\boldsymbol{u}} \, I^f \left(\frac{\rho e + p}{\rho} \right) \right]
```

is treated implicitly.

!!! note

    When upwinding is enabled for energy advection (the van Leer limiter by
    default), the implicit solve still uses the central reconstruction; the
    difference between the upwinded and central fluxes is applied after the
    Newton solve as a `T_post_imp!` correction
    (`correct_implicit_advection_tendency!`), so the upwind flux is evaluated
    with the Newton-solved velocity.

### Scalars

For an arbitrary scalar ``\chi``, the density-weighted scalar ``\rho\chi`` follows the continuity equation

```math
\frac{\partial}{\partial t} \rho \chi = - \nabla \cdot(\rho \chi \boldsymbol{u}) + \rho \mathcal{S}_{\chi}.
```

This is stabilized by adding a 4th-order hyperdiffusion term

```math
- \nu_\chi \nabla \cdot(\rho \nabla^3(\chi))
```

This is discretized using

```math
\frac{\partial}{\partial t} \rho \chi \approx
- \mathcal{D}^{split}_h[ \rho \bar{\boldsymbol{u}}, \chi ]
- \mathcal{D}^c_v \left[ WI^f(J,\rho) \, U^f\left( \tilde{\boldsymbol{u}},  \frac{\rho \chi}{\rho} \right) \right]
- \nu_\chi \hat{\mathcal{D}}_h ( \rho \, \mathcal{G}_h (\psi) )
```

where

```math
\psi = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h \left( \frac{\rho \chi}{\rho} \right)\right) \right]
```

For total water ``\rho q_\mathrm{tot}``, the central reconstruction

```math
- \mathcal{D}^c_v \left[ WI^f(J,\rho) \, \tilde{\boldsymbol{u}} \, I^f\left( \frac{\rho \chi}{\rho} \right) \right]
```

is treated implicitly (as for total energy), with the upwind-central difference
applied after the Newton solve as a `T_post_imp!` correction. All other
grid-mean tracers are advected fully explicitly with the reconstruction
selected by `tracer_upwinding` (the van Leer limiter by default).
