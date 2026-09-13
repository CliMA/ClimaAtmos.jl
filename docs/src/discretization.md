# Discretization and Operators

ClimaAtmos discretizes the [governing equations](equations.md) with a hybrid
scheme: a spectral element method in the horizontal and finite differences on a
staggered grid in the vertical [Yatunin2026](@cite). This page explains that
choice and defines the discrete operators once, for use by the other pages in
these docs.

The operators come from ClimaCore, whose
[Mathematical framework](@extref ClimaCore Mathematical-framework),
[Spectral elements: continuous and discontinuous Galerkin](@extref ClimaCore Spectral-elements:-continuous-and-discontinuous-Galerkin),
[Staggered vertical discretization](@extref ClimaCore Staggered-vertical-discretization),
and [DSS and numerical fluxes](@extref ClimaCore DSS-and-numerical-fluxes)
pages define the nodal basis, the strong and weak forms, the staggered vertical
operators, and direct stiffness summation. This page states the choices
ClimaAtmos makes among those operators — which form each term uses, and which
reconstruction — gives the symbol used for each one in the equations, and
assembles the semi-discrete equations. For the short names the tendencies are
written in, see [Discrete operators](@ref) in the API.

## Why a hybrid scheme

The horizontal and vertical directions of an atmospheric grid have different
requirements, and the scheme treats them differently.

Horizontally, spectral elements give high-order accuracy with one neighbor
exchange per step; vertically, finite differences on a staggered grid couple
only neighboring levels, so the fast vertical waves that fine vertical spacing
admits can be solved implicitly column by column, and the staggering
suppresses the computational modes of an unstaggered vertical grid.
ClimaCore's [Staggered vertical discretization](@extref ClimaCore Staggered-vertical-discretization)
gives the argument in full, including why Lorenz staggering is used. See
[Implicit Solver](implicit_solver.md) for the implicit solve.

The two parts are independent. The vertical staggering, the reconstruction
rules, and the implicit solve make no reference to how horizontal derivatives
are computed, and the [semi-discrete equations](@ref "Semi-discrete equations")
at the end of this page are written in operator symbols rather than in any one
method's terms. An alternative horizontal discretization — a discontinuous
Galerkin method, for example — would supply its own realization of the
horizontal operators and of the projection ``\mathcal{P}``, and leave the rest
of this page, and of the model, unchanged.

## Grid layout and staggering

The domain is divided into ``N_h`` horizontal elements, each extruded into
``N_v`` vertical layers. Horizontal elements carry ``(N_p + 1)^2``
Gauss–Lobatto–Legendre nodal points, so fields are polynomials of order ``N_p``
within an element. On the sphere, the horizontal mesh is an equiangular cubed
sphere [Sadourny1972, Ronchi1996](@cite); in Cartesian geometry the same
machinery discretizes a box. See
[Grids](grids.md) for the constructors and
[Topography Representation](topography.md) for the terrain-following vertical
coordinate.

The vertical arrangement is a **Lorenz staggering**: the covariant
vertical velocity component ``u_3`` is defined on element faces, and every other
variable — including the horizontal velocity components ``u_1`` and ``u_2`` —
is defined on element centers [Yatunin2026](@cite).

Throughout the docs and the code, a ``ᶜ`` prefix marks a center field and a
``ᶠ`` prefix marks a face field; see [Notation and Symbols](notation.md) for the
full convention.

## The horizontal discretization: spectral elements

Within each element, fields are expanded in a nodal polynomial basis on the
Gauss–Lobatto–Legendre points, and horizontal derivatives are computed by
differentiating that expansion. Everything in this section is specific to the
continuous spectral element method; the vertical discretization and the
semi-discrete equations do not depend on it.

### Strong and weak forms: which to use

Every horizontal derivative comes in a **strong** form, which differentiates the
basis functions directly, and a **weak** form, constructed to be the negative
adjoint of the strong form under the discrete inner product — a discrete
integration by parts, from which discrete divergence and Stokes theorems
follow. ClimaCore's
[Spectral elements](@extref ClimaCore Spectral-elements:-continuous-and-discontinuous-Galerkin)
page derives both; what matters here is which one each term uses.

#### The rule

Adjointness is a property of a *pair*. A term conserves what it should when the
two operators acting in it are adjoint to one another, so that summing over the
domain telescopes into boundary terms alone. That gives a short rule:

| Term                                  | Form                                                                              | Why                                                                                                                               |
|:------------------------------------- |:--------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------- |
| Flux divergence of a conserved scalar | **weak**                                                                          | Discrete divergence theorem: the domain integral changes only through boundary fluxes                                             |
| Curl in the momentum equation         | **weak**                                                                          | Discrete Stokes theorem: vorticity is conserved globally                                                                          |
| Gradient in the momentum equation     | **strong**                                                                        | Pairs with the weak divergence, so kinetic energy is conserved when pressure gradients and sources are absent [Taylor2020](@cite) |
| Scalar Laplacian                      | ``\tilde{\nabla}_h \cdot \nabla_h``                                               | Weak divergence of a strong gradient satisfies a second-order integration by parts                                                |
| Vector Laplacian                      | ``\tilde{\nabla}_h (\nabla_h \cdot) - \tilde{\nabla}_h \times (\nabla_h \times)`` | Same identity, with the weak operator on the outside                                                                              |

Mixing the two forms is therefore deliberate rather than incidental: using the
weak divergence for the mass flux and the strong gradient for kinetic energy is
what makes the two terms cancel exactly in the energy budget. Using the same form
for both would break that cancellation and leave a spurious energy source. See
[Conservation Properties](conservation.md).

Note the asymmetry between the two Laplacians. Both put one strong and one weak
operator in the pair, but the scalar Laplacian takes the divergence weakly, while
the vector Laplacian takes the outer operator weakly. Either arrangement
satisfies the identity, because each is the adjoint of the other; what matters is
that the pair contains one of each. Fourth-order hyperdiffusion applies the
Laplacian twice with direct stiffness summation in between, which removes
inter-element discontinuities without disturbing the inner product; see
[Hyperdiffusion](hyperdiffusion.md).

#### In the code

The strong operators are `divₕ`, `gradₕ`, and `curlₕ`; the weak ones take a `w`
prefix: `wdivₕ`, `wgradₕ`, `wcurlₕ`; the split divergence is `split_divₕ`. So
the scalar and vector Laplacians read

```julia
ᶜ∇²s_d = @. wdivₕ(gradₕ(s_d - sd_r))                           # weak div ∘ strong grad
ᶜ∇²u = @. C123(wgradₕ(divₕ(ᶜu))) - C123(wcurlₕ(C123(curlₕ(ᶜu))))
```

The variable-resolution behavior of these operators is examined by
[Guba2014](@cite).

### Projection: direct stiffness summation

The projection ``\mathcal{P}`` onto the continuous spectral element basis is
[direct stiffness summation](@extref ClimaCore DSS-and-numerical-fluxes)
(DSS), the volume-weighted average over the copies of each element-boundary
node. It preserves the discrete inner product, so it leaves the conservation
properties of the weak operators intact, and it is the only operation that
communicates horizontally between elements. ClimaAtmos applies it several times
per step — at stage boundaries, around the implicit solve, and at the end of
the step — which keeps round-off errors from accumulating as discontinuities
across element boundaries.

## The vertical discretization: interpolating between centers and faces

Staggering forces interpolation. The covariant vertical velocity ``u_3`` is
defined on faces and everything else on centers, so any flux term needs one or
the other moved. Which average is used is not a matter of taste: as with the
strong and weak forms, the choice determines whether a discrete conservation law
holds.

Three kinds appear.

**Arithmetic mean** (``I^c``, ``I^f``; `ᶜinterp`, `ᶠinterp`). The plain average of
the two neighbors. Use it for quantities that are not weighted by mass, such as
the covariant velocity components, and wherever no conservation statement rides
on the result.

**Mass-weighted average** (``WI^f``; `ᶠwinterp`). The average weighted by
``\rho J``, that is, ``WI^f(w, x) = I^f(w x) / I^f(w)``. Use it for a quantity
that will be multiplied by a mass flux. The arithmetic and mass-weighted
averages form an adjoint pair satisfying a density-weighted averaging-by-parts
identity, so the vertical flux divergence telescopes; using two
arithmetic means instead would leave a residual. In the code, it appears where
the horizontal velocity is reconstructed onto faces, and in the hyperdiffusive
momentum tendency. The face mass flux itself is not this operator: it is
``I^f(\rho J) \tilde{\boldsymbol{u}} / J^f``, divided by the face Jacobian
rather than by ``I^f(J)``, so that it telescopes exactly.

**Upwind or limited reconstruction** (``U^f``; `ᶠupwind1`, `ᶠupwind3`,
`ᶠlin_vanleer`). A biased or flux-corrected reconstruction
[vanLeer1977, Lin1994, Zalesak1979](@cite). Use it for advected scalars, where
a centered average would produce dispersive oscillations and negative
concentrations. The default for grid-mean energy and tracer transport is the van
Leer limiter of [Lin1994](@cite), constrained by the local extrema of the
neighboring center values.

| You need                                       | Use                   | Because                                                         |
|:---------------------------------------------- |:--------------------- |:--------------------------------------------------------------- |
| A face value of a covariant velocity component | Arithmetic mean       | No mass weighting is involved                                   |
| A face value that multiplies a mass flux       | Mass-weighted average | Forms the adjoint pair that makes the flux divergence telescope |
| A face value of an advected scalar             | Upwind or limited     | Preserves monotonicity and positivity                           |
| A center value of the vertical velocity        | Arithmetic mean       | There is no unique mass-weighted inverse                        |

Two consequences follow. Setting ``\psi = 1`` in the scalar flux reconstruction
recovers the mass flux divergence, so tracer transport stays consistent with
mass transport, and a uniform tracer field stays uniform. At the domain
boundaries, ``I^f`` extrapolates by reusing the nearest interior value, while
the vertical gradient and curl operators are set to zero there.

## Operator reference

!!! note

    Since ClimaCore 0.15, the strong- and weak-form horizontal spectral
    operators are unified: `Divergence`, `Gradient`, and `Curl` take a
    form-type parameter (`StrongForm`, the default, or `WeakForm`), so the weak
    divergence, for example, is
    [`ClimaCore.Operators.Divergence`](@extref)`{WeakForm}`. The ClimaAtmos code
    still uses the legacy names (`WeakDivergence`, `WeakGradient`, `WeakCurl`),
    which ClimaCore keeps as deprecated aliases and which therefore have no
    documentation to link to; the form-parameterized
    [`ClimaCore.Operators.Gradient`](@extref) and
    [`ClimaCore.Operators.Curl`](@extref) are their current spellings.

Each operator below has a short name in the ClimaAtmos source, documented under
[Discrete operators](@ref) in the API.

### Reconstruction between centers and faces

  - ``I^c`` is the face-to-center interpolation
    [`ClimaCore.Operators.InterpolateF2C`](@extref), an arithmetic mean.

  - ``I^f`` is the center-to-face interpolation
    [`ClimaCore.Operators.InterpolateC2F`](@extref), an arithmetic mean with
    constant extrapolation to the domain boundaries.

  - ``WI^f`` is the center-to-face weighted interpolation
    [`ClimaCore.Operators.WeightedInterpolateC2F`](@extref), with
    ``WI^f(w, x) = I^f(w x) / I^f(w)`` for a weight ``w``. It appears with the
    weight ``\rho J``, to reconstruct the horizontal velocity onto faces and in
    the hyperdiffusive vertical-momentum tendency.

  - ``\mathcal{M}^f(\rho)`` is the face mass flux,

    ```math
    \mathcal{M}^f(\rho) = \frac{I^f(\rho J)}{J^f} \, \tilde{\boldsymbol{u}} ,
    ```

    with ``J^f`` the Jacobian on faces. Dividing by ``J^f`` rather than by
    ``I^f(J)`` makes the flux divergence telescope: ``D^c``
    multiplies its argument by ``J^f`` before differencing, so the ``J^f``
    cancels and the flux that leaves one cell is exactly the flux that enters
    the next. The two denominators agree only where ``J^f = I^f(J)``, which
    holds on a flat or linearly warped grid but not under a SLEVE warp; see
    [Topography Representation](topography.md).

  - ``U^f`` is the center-to-face upwind product: first order
    [`ClimaCore.Operators.UpwindBiasedProductC2F`](@extref), third order
    [`ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F`](@extref), or the van
    Leer limiter [`ClimaCore.Operators.LinVanLeerC2F`](@extref). The van Leer
    limiter is the default for grid-mean energy and tracer vertical transport,
    set by the `energy_q_tot_upwinding` and `tracer_upwinding` configuration
    keys.

### Horizontal differential operators

These realize the spectral element forms of the previous section; a different
horizontal discretization would replace this list.

  - ``\mathcal{D}_h`` is the strong horizontal spectral divergence
    [`ClimaCore.Operators.Divergence`](@extref).

  - ``\hat{\mathcal{D}}_h`` is the weak horizontal spectral divergence
    `ClimaCore.Operators.WeakDivergence`.

  - ``\mathcal{D}^{split}_h`` is the split, skew-symmetric horizontal divergence
    [`ClimaCore.Operators.SplitDivergence`](@extref),

    ```math
    \mathcal{D}^{split}_h(\rho \boldsymbol{u}, \psi) =
      \tfrac{1}{2} \hat{\mathcal{D}}_h(\rho \boldsymbol{u} \psi)
      + \tfrac{1}{2} \left[ \psi \, \hat{\mathcal{D}}_h(\rho \boldsymbol{u})
        + \rho \boldsymbol{u} \cdot \mathcal{G}_h \psi \right].
    ```

    The horizontal advective fluxes of energy, moisture, and tracers use this
    entropy-stable split form; for ``\psi = 1`` it reduces to the weak
    divergence, which is why the semi-discrete mass equation below is written
    with ``\hat{\mathcal{D}}_h`` (in the code, the mass flux is
    `split_divₕ(ρu, 1)`). The horizontal pressure-gradient term uses an
    analogous split form.

  - ``\mathcal{G}_h`` is the strong horizontal spectral gradient
    [`ClimaCore.Operators.Gradient`](@extref).

  - ``\hat{\mathcal{G}}_h`` is the weak horizontal spectral gradient
    `ClimaCore.Operators.WeakGradient`, the outer gradient of the
    vector Laplacian.

  - ``\mathcal{C}_h`` is the curl of the components involving horizontal
    derivatives [`ClimaCore.Operators.Curl`](@extref). Applied to
    ``\boldsymbol{u}_h`` it returns a vector with only vertical contravariant
    components; applied to ``\boldsymbol{u}_v`` it returns a vector with only
    horizontal contravariant components.

  - ``\hat{\mathcal{C}}_h`` is the corresponding weak curl
    `ClimaCore.Operators.WeakCurl`.

  - ``\mathcal{P}`` is the projection onto the continuous spectral element
    basis; see [Projection: direct stiffness summation](@ref).

### Vertical differential operators

  - ``D^c`` is the face-to-center vertical divergence
    [`ClimaCore.Operators.DivergenceF2C`](@extref). Separate variants impose the
    boundary conditions for advective, precipitation, and diffusive fluxes.

  - ``G^f`` is the center-to-face vertical gradient
    [`ClimaCore.Operators.GradientC2F`](@extref), set to zero at the top and
    bottom boundaries. The boundary values are placeholders: the vertical
    velocity at the boundaries is fixed instead by the impenetrability
    condition described above.

  - ``G^c`` is the face-to-center vertical gradient
    [`ClimaCore.Operators.GradientF2C`](@extref), used where a face-defined
    field such as the geopotential is differentiated onto centers.

  - ``C^f`` is the center-to-face curl of the components involving
    vertical derivatives [`ClimaCore.Operators.CurlC2F`](@extref), set to zero
    at the top and bottom boundaries. Applied to ``\boldsymbol{u}_h`` it returns
    a vector with only a horizontal contravariant component.

## Reconstructions and conservation

The reconstructions above are the ones that make the global conservation laws
hold [Yatunin2026](@cite); they resemble those of
[SimmonsBurridge1981](@cite) in some respects.

  - **Density** is reconstructed onto faces as ``I^f(\rho J) / J^f``, the
    Jacobian-weighted average that the face mass flux needs.
  - **Velocity** covariant components use unweighted averages; the contravariant
    vertical component on faces uses a mass-weighted average. There is no unique
    reconstruction of the contravariant vertical velocity onto centers.
  - **Vorticity** contravariant components are computed with a weak horizontal
    curl.
  - **Momentum advection** uses the vector-invariant form, with a strong
    horizontal gradient of kinetic energy and a weighted average of the vorticity
    term. In this form, momentum advection conserves kinetic energy and
    vorticity globally, and avoids the curvature terms that appear in advection
    terms in non-orthogonal coordinates.

Because total energy is separately conserved, any numerical conversion between
kinetic and non-kinetic energy comes from the discretized pressure-gradient term
and the physical sources, and not from the advection scheme.

## Writing a new tendency

Putting the two rules together, a new term added to `src/prognostic_equations/`
generally follows this pattern.

 1. Decide where the result lives. A tendency for a center variable must end on
    centers, one for `u₃` on faces.
 2. For a horizontal advective flux of a conserved scalar, use the split form
    `split_divₕ`; for other horizontal flux divergences, such as diffusive
    ones, use `wdivₕ`. For a gradient in the momentum equation, use `gradₕ`.
 3. For the vertical part, use `ᶜdivᵥ` on a face flux, and build that face flux
    with `ᶠwinterp` if it multiplies a mass flux, or with an upwind operator if it
    transports a scalar.
 4. If the term diffuses energy, diffuse the dry static energy and the effective
    total water separately rather than a lumped total enthalpy, so the
    decomposition stays energetically consistent. See
    [Thermodynamics and the Working Fluid](thermodynamics.md).
 5. If the term is vertical and fast, add it to the implicit tendency and to the
    Jacobian; see [Implicit Solver](implicit_solver.md).

The semi-discrete equations at the end of this page show these patterns applied
to each governing equation.

## Timestepping

Tendencies are split into an explicit part and an implicit part and advanced
with a horizontally explicit, vertically implicit (HEVI) additive Runge–Kutta
method [Ascher1997, Gardner2018](@cite). The implicit part takes the vertical
terms responsible for sound and gravity waves, falling and sedimenting
condensate, the Rayleigh damping of ``u_3``, and, when `implicit_diffusion` is
enabled, the vertical diffusion; the viscous sponge stays explicit. Because the
implicit part involves no horizontal derivatives, it can be solved independently
in each column, with no horizontal communication.

This lifts the timestep restriction from fast vertical dynamics and leaves the
horizontal propagation of sound waves as the limit, so the maximum timestep
scales as ``\delta t \sim (\delta x)_{\min} / c_s`` with the minimum horizontal
distance between nodal points and the speed of sound.

[Implicit Solver](implicit_solver.md) documents the Newton solve, the Jacobian
approximation, and the available Jacobian algorithms.
[Integer Time (ITime)](itime.md) explains how time itself is represented.

## Semi-discrete equations

The sections above define the operators. What follows applies them: the
semi-discrete form of each governing equation, that is, the equations discretized
in space but still continuous in time. The continuous equations these come from
are on the [Governing Equations](equations.md) page.

### Reconstructed velocities and kinetic energy

Two reconstructed velocities appear throughout, along with the kinetic energy
built from them. All follow from the staggering: the covariant vertical
component lives on faces and everything else on centers, so each flux term
needs one or the other interpolated.

  - ``\tilde{\boldsymbol{u}}`` is the mass-weighted reconstruction of velocity at the interfaces,
    carried out by weighted interpolation of the horizontal components (see `compute_ᶠuₕ³` in
    `src/cache/precomputed_quantities.jl`):

    ```math
    \tilde{\boldsymbol{u}} = WI^f(\rho J, \boldsymbol{u}_h) + \boldsymbol{u}_v
    ```

  - ``\bar{\boldsymbol{u}}`` is the reconstruction of velocity at cell-centers, carried out by linear interpolation of the covariant vertical component:

    ```math
    \bar{\boldsymbol{u}} = \boldsymbol{u}_h + I^c(\boldsymbol{u}_v)
    ```

  - ``K = \tfrac{1}{2} \|\boldsymbol{u}\|^2`` is the specific kinetic energy (J/kg), reconstructed at cell centers by

    ```math
    K = \tfrac{1}{2} (\boldsymbol{u}_{h} \cdot \boldsymbol{u}_{h} + 2 \boldsymbol{u}_{h} \cdot I^c (\boldsymbol{u}_{v}) + I^c(\boldsymbol{u}_{v} \cdot \boldsymbol{u}_{v})),
    ```

    where ``\boldsymbol{u}_{h}`` is defined on cell-centers, ``\boldsymbol{u}_{v}`` is defined on cell-faces, and ``I^c (\boldsymbol{u}_{v})`` is interpolated using covariant components.

  - No-flux boundary conditions are enforced by requiring the vertical contravariant component ``\tilde{u}^3`` of the face-valued velocity at the boundary to be zero, which fixes the boundary value of the covariant component:

    ```math
    u_3 = -\frac{g^{31} u_1 + g^{32} u_2}{g^{33}}.
    ```

### Mass

Follows the continuity equation

```math
\frac{\partial}{\partial t} \rho = - \nabla \cdot(\rho \boldsymbol{u}) + \rho \hat{S}_{q_t}.
```

This is discretized using the following

```math
\frac{\partial}{\partial t} \rho
= - \hat{\mathcal{D}}_h[ \rho \bar{\boldsymbol{u}}] - D^c \left[ \mathcal{M}^f(\rho) \right] + \rho \hat{S}_{q_t}
```

with the

```math
-D^c[\mathcal{M}^f(\rho)]
```

term treated implicitly (the full face velocity ``\tilde{\boldsymbol{u}}``, including
the topographic contribution of ``\boldsymbol{u}_h``, enters the implicit term).

### Momentum

Uses the vector-invariant form

```math
\frac{\partial}{\partial t} \boldsymbol{u}  = - (2 \boldsymbol{\Omega} + \nabla \times \boldsymbol{u}) \times \boldsymbol{u} - c_{pd} (\theta_v - \theta_{v, r}) \nabla \Pi  - \nabla [(\Phi - \Phi_r) + K].
```

The pressure-gradient force is expressed through the Exner function,
``-\nabla p / \rho = -c_{pd} \theta_v \nabla \Pi``, with a hydrostatically
balanced reference state ``(\theta_{v,r}, \Phi_r)`` subtracted; the reference
profile, its defining formulas, and its parameters are given on the
[Governing Equations](equations.md) page.

#### Horizontal momentum

By breaking the curl and cross product terms into horizontal and vertical contributions, and removing zero terms (e.g. ``\nabla_v \times \boldsymbol{u}_v = 0``), we obtain

```math
\frac{\partial}{\partial t} \boldsymbol{u}_h  =
  - (2 \boldsymbol{\Omega}^h + \nabla_v \times \boldsymbol{u}_h +  \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v
  - (2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h
  - c_{pd} (\theta_v - \theta_{v, r}) \nabla_h \Pi  - \nabla_h [(\Phi - \Phi_r) + K],
```

where ``\boldsymbol{u}^h`` and ``\boldsymbol{u}^v`` are the horizontal and vertical *contravariant* vectors.

Topography enters through the computation of the contravariant velocity components (projections from the covariant velocity representation) before the cross-product contributions.

This is stabilized by adding 4th-order vector hyperviscosity

```math
-\nu_u \, \nabla_h^2 (\nabla_h^2(\boldsymbol{\overline{u}})),
```

projected onto the first two covariant directions, where
``\boldsymbol{\overline{u}}`` is the center-valued velocity defined above and

```math
\nabla_h^2(\boldsymbol{v}) = \nabla_h(\nabla_{h} \cdot \boldsymbol{v}) - \nabla_{h} \times (\nabla_{h} \times \boldsymbol{v})
```

is the horizontal vector Laplacian.

The ``(2 \boldsymbol{\Omega}^h + \nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^v`` term is discretized as:

```math
\frac{I^c\{(2 \boldsymbol{\Omega}^h + C^f[\boldsymbol{u}_h] + \hat{\mathcal{C}}_h[\boldsymbol{u}_v]) \times (I^f(\rho J)\tilde{\boldsymbol{u}}^v)\}}{\rho J} ,
```

in which ``C^f[\boldsymbol{u}_h] + \hat{\mathcal{C}}_h[\boldsymbol{u}_v]`` is the discrete horizontal relative vorticity ``\boldsymbol{\omega}^h``.

The ``(2 \boldsymbol{\Omega}^v + \nabla_h \times \boldsymbol{u}_h) \times \boldsymbol{u}^h`` term is discretized as

```math
(2 \boldsymbol{\Omega}^v + \hat{\mathcal{C}}_h[\boldsymbol{u}_h]) \times \boldsymbol{u}^h
```

and the ``c_{pd} (\theta_v - \theta_{v,r}) \nabla_h \Pi + \nabla_h (\Phi - \Phi_r + K)`` term is discretized in the split form introduced above,

```math
\frac{c_{pd}}{2} \left\{ \theta_v' \, \mathcal{G}_h[\Pi]
  + \mathcal{G}_h[\theta_v' \Pi] - \Pi \, \mathcal{G}_h[\theta_v'] \right\}
+ \mathcal{G}_h[\Phi - \Phi_r + K] ,
```

with ``\theta_v' = \theta_v - \theta_{v,r}``. All these terms are treated
explicitly.

The hyperviscosity term is

```math
- \nu_u \left\{ \delta_{div} \, \hat{\mathcal{G}}_h ( \mathcal{D}_h(\boldsymbol{\psi}) ) - \hat{\mathcal{C}}_h( \mathcal{C}_h( \boldsymbol{\psi} )) \right\}
```

where

```math
\boldsymbol{\psi} = \mathcal{P} \left[ \hat{\mathcal{G}}_h ( \mathcal{D}_h(\boldsymbol{\overline{u}}) ) - \hat{\mathcal{C}}_h( \mathcal{C}_h( \boldsymbol{\overline{u}} )) \right]
```

and ``\delta_{div}`` is the divergence damping factor, which strengthens the
damping of divergent modes; see [Hyperdiffusion](hyperdiffusion.md).

#### Vertical momentum

Similarly, for vertical velocity

```math
\frac{\partial}{\partial t} \boldsymbol{u}_v  =
  - (2 \boldsymbol{\Omega}^h + \nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^h
  -c_{pd} (\theta_v - \theta_{v, r}) \nabla_v \Pi  - \nabla_v [(\Phi - \Phi_r) + K].
```

The ``(2 \boldsymbol{\Omega}^h + \nabla_v \times \boldsymbol{u}_h + \nabla_h \times \boldsymbol{u}_v) \times \boldsymbol{u}^h`` term is discretized as

```math
(2 \boldsymbol{\Omega}^h + C^f[\boldsymbol{u}_h] + \hat{\mathcal{C}}_h[\boldsymbol{u}_v]) \times I^f(\boldsymbol{u}^h) ,
```

The ``\nabla_v K`` term is discretized as

```math
G^f[K],
```

The ``c_{pd} (\theta_v - \theta_{v,r}) \nabla_v \Pi + \nabla_v (\Phi - \Phi_r)`` term is discretized as

```math
I^f[c_{pd} (\theta_v - \theta_{v, r} ) ] G^f[\Pi] + G^f[\Phi - \Phi_r],
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
\frac{\partial}{\partial t} \rho e_{tot} = - \nabla \cdot((\rho e_{tot} + p) \boldsymbol{u} + \boldsymbol{F}_R) + \rho S_{e},
```

which is stabilized by adding a 4th-order hyperdiffusion term on total enthalpy:

```math
- \nu_h \left[ \nabla \cdot \left( \rho \nabla^3 s_d \right)
  + \nabla \cdot \left( \rho \, h_{\mathrm{tot,cl}} \, \nabla^3 q_t^{\mathrm{eff}} \right) \right],
```

where the total enthalpy is decomposed into dry static energy ``s_d`` and a
single water-enthalpy piece, carrying the suspended-water enthalpy
``h_{\mathrm{tot,cl}}`` against the gradient of the effective total water
``q_t^{\mathrm{eff}} = q_t - q_{rai} - q_{sno}``, so that the ``\nabla^4``
operator never acts on a lumped total enthalpy. See
[Hyperdiffusion](hyperdiffusion.md) for the reference-state subtraction that
both Laplacians apply.

This is discretized using

```math
\frac{\partial}{\partial t} \rho e_{tot} \approx
- \mathcal{D}^{split}_h[ \rho \bar{\boldsymbol{u}}, \tfrac{\rho e_{tot} + p}{\rho} ]
- D^c \left[ \mathcal{M}^f(\rho) \, I^f \left(\frac{\rho e_{tot} + p}{\rho} \right) \right]
- D^c \left[ \boldsymbol{F}_R \right]
- \nu_h \left[ \hat{\mathcal{D}}_h( \rho \mathcal{G}_h(\psi_{s_d}) )
  + \hat{\mathcal{D}}_h( \rho \, h_{\mathrm{tot,cl}} \, \mathcal{G}_h(\psi_{q_t^{\mathrm{eff}}}) ) \right],
```

where

```math
\psi_x = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h (x - x_r) \right) \right],
```

with ``x_r`` the hydrostatic reference profile subtracted before the first
Laplacian, as described in [Hyperdiffusion](hyperdiffusion.md), and the
radiative flux divergence ``-D^c[\boldsymbol{F}_R]`` is
applied separately as an explicit tendency.

The central reconstruction

```math
- D^c \left[ \mathcal{M}^f(\rho) \, I^f \left(\frac{\rho e_{tot} + p}{\rho} \right) \right]
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
\frac{\partial}{\partial t} \rho \chi = - \nabla \cdot(\rho \chi \boldsymbol{u}) + \rho S_{\chi}.
```

This is stabilized by adding a 4th-order hyperdiffusion term

```math
- \nu_\chi \nabla \cdot(\rho \nabla^3(\chi))
```

This is discretized using

```math
\frac{\partial}{\partial t} \rho \chi \approx
- \mathcal{D}^{split}_h[ \rho \bar{\boldsymbol{u}}, \chi ]
- D^c \left[ \frac{I^f(\rho J)}{J^f} \, U^f\left( \tilde{\boldsymbol{u}},  \frac{\rho \chi}{\rho} \right) \right]
- \nu_\chi \hat{\mathcal{D}}_h ( \rho \, \mathcal{G}_h (\psi) )
```

where

```math
\psi = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h \left( \frac{\rho \chi}{\rho} \right)\right) \right]
```

For total water ``\rho q_\mathrm{tot}``, the central reconstruction

```math
- D^c \left[ \mathcal{M}^f(\rho) \, I^f\left( \frac{\rho \chi}{\rho} \right) \right]
```

is treated implicitly (as for total energy), with the upwind-central difference
applied after the Newton solve as a `T_post_imp!` correction. All other
grid-mean tracers are advected fully explicitly with the reconstruction
selected by `tracer_upwinding` (the van Leer limiter by default).

## Where this is implemented

| Concept                           | Source                                                                                                                                           |
|:--------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------ |
| Operator short names              | [Discrete operators](@ref), defined in [src/utils/abbreviations.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/utils/abbreviations.jl) |
| Horizontal and vertical advection | [src/prognostic_equations/advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl)                  |
| Sedimentation and water transport | [src/prognostic_equations/water_advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/water_advection.jl)      |
| Implicit/explicit tendency split  | [src/prognostic_equations/implicit/](https://github.com/CliMA/ClimaAtmos.jl/tree/main/src/prognostic_equations/implicit)                         |
| Grid construction                 | [src/simulation/grids.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/simulation/grids.jl)                                              |

The upwinding and limiter choices are exposed as configuration keys; see
[Configuration Options](configuration_options.md) for `energy_q_tot_upwinding`,
`tracer_upwinding`, and the corresponding PROPHET keys.
