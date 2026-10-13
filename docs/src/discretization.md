# Discretization and Operators

ClimaAtmos discretizes the [governing equations](equations.md) with a hybrid
scheme: a Galerkin spectral element method in the horizontal and finite
differences on a staggered grid in the vertical [Yatunin2026](@cite). The
horizontal method is continuous Galerkin (CG) today; ClimaCore also provides
discontinuous Galerkin (DG) on the same element structure, and the two differ
only in how element-local results are completed across element boundaries, so
this page is written to cover both and marks the one place they part ways.

The page explains the choice of scheme, states the two rules that decide which
operator each term uses, defines the operator symbols used throughout these
docs, and shows how the pieces assemble into a discretized equation. Readers who
need only the symbols can go straight to the [operator reference](@ref "Operator reference"); those adding a term to the model want
[Writing a new tendency](@ref).

The operators come from ClimaCore, whose
[Mathematical framework](@extref ClimaCore Mathematical-framework),
[Spectral elements: continuous and discontinuous Galerkin](@extref ClimaCore Spectral-elements:-continuous-and-discontinuous-Galerkin),
[Staggered vertical discretization](@extref ClimaCore Staggered-vertical-discretization),
and [DSS and numerical fluxes](@extref ClimaCore DSS-and-numerical-fluxes)
pages define the nodal basis, the strong and weak forms, the staggered vertical
operators, and the completion step (direct stiffness summation for CG,
interface fluxes for DG). This page states the choices ClimaAtmos makes among
those operators — which form each term uses, and which reconstruction — and
gives the symbol used for each one in the equations. For the short names the
tendencies are written in, see [Discrete operators](@ref) in the API.

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
rules, and the implicit solve are independent of how horizontal derivatives are
computed, and the [assembled equations](@ref "Assembling a discretized equation")
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
differentiating that expansion. The element-local operators, and the rule
below for choosing between their strong and weak forms, are the same for
continuous and discontinuous Galerkin; the two methods differ only in the
completion step at the end of this section. The vertical discretization does
not depend on either.

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

Mixing the two forms is therefore deliberate: using the
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

### Completing a tendency across elements

An element-local operator evaluates each element on its own, so at a node on
an element boundary the weak-form result is incomplete: it holds only that
element's share. A *completion* step, written ``\mathcal{P}`` in these docs,
supplies the rest, and it is the one place where continuous and discontinuous
Galerkin differ; ClimaCore's
[DSS and numerical fluxes](@extref ClimaCore DSS-and-numerical-fluxes) page
develops both.

With continuous elements, which ClimaAtmos uses, ``\mathcal{P}`` is direct
stiffness summation (DSS): the volume-weighted average over the copies of each
boundary node, which makes the field single-valued there. It preserves the
discrete inner product, so it leaves the conservation properties of the weak
operators intact, and it is the only operation that communicates horizontally
between elements. ClimaAtmos applies it several times per step — at stage
boundaries, around the implicit solve, and at the end of the step — which keeps
round-off errors from accumulating as discontinuities across element
boundaries.

With discontinuous elements, fields may jump at element boundaries, and
``\mathcal{P}`` instead adds an interface numerical flux (a central or Rusanov
flux, or a user-supplied one) to each conservative term and a lifting flux to
each gradient or curl. The element-local operators and the choice of form are
untouched; only the completion changes, and the discrete conservation
statements above carry over because the numerical flux is single-valued and
antisymmetric across each interface.

## The vertical discretization: interpolating between centers and faces

Staggering forces interpolation. The covariant vertical velocity ``u_3`` is
defined on faces and everything else on centers, so any flux term needs one or
the other moved. As with the strong and weak forms, the choice of average
decides whether a discrete conservation law holds.

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
    [`ClimaCore.Operators.Divergence`](@extref)`{WeakForm}`.

Each operator below has a short name in the ClimaAtmos source, documented under
[Discrete operators](@ref) in the API.

### Reconstruction between centers and faces

  - The face-to-center interpolation ``I^c`` is
    [`ClimaCore.Operators.InterpolateF2C`](@extref), an arithmetic mean.

  - The center-to-face interpolation ``I^f`` is
    [`ClimaCore.Operators.InterpolateC2F`](@extref), an arithmetic mean with
    constant extrapolation to the domain boundaries.

  - The center-to-face weighted interpolation ``WI^f`` is
    [`ClimaCore.Operators.WeightedInterpolateC2F`](@extref), with
    ``WI^f(w, x) = I^f(w x) / I^f(w)`` for a weight ``w``. It appears with the
    weight ``\rho J``, to reconstruct the horizontal velocity onto faces and in
    the hyperdiffusive vertical-momentum tendency.

  - The face mass flux ``\mathcal{M}^f(\rho)`` is

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

  - The center-to-face upwind product ``U^f``: first order
    [`ClimaCore.Operators.UpwindBiasedProductC2F`](@extref), third order
    [`ClimaCore.Operators.Upwind3rdOrderBiasedProductC2F`](@extref), or the van
    Leer limiter [`ClimaCore.Operators.LinVanLeerC2F`](@extref). The van Leer
    limiter is the default for grid-mean energy and tracer vertical transport,
    set by the `energy_q_tot_upwinding` and `tracer_upwinding` configuration
    keys.

### Horizontal differential operators

These are the element-local strong and weak forms of the previous section, and
they are the same operators for continuous and discontinuous elements; only
``\mathcal{P}`` below changes between the two.

  - The strong horizontal spectral divergence ``\mathcal{D}_h`` is
    [`ClimaCore.Operators.Divergence`](@extref).

  - The weak horizontal spectral divergence ``\hat{\mathcal{D}}_h`` is
    `Divergence{WeakForm}` (see [`ClimaCore.Operators.Divergence`](@extref)).

  - The split, skew-symmetric horizontal divergence ``\mathcal{D}^{split}_h`` is
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

  - The strong horizontal spectral gradient ``\mathcal{G}_h`` is
    [`ClimaCore.Operators.Gradient`](@extref).

  - The weak horizontal spectral gradient ``\hat{\mathcal{G}}_h`` is
    `Gradient{WeakForm}` (see [`ClimaCore.Operators.Gradient`](@extref)), the
    outer gradient of the vector Laplacian.

  - The curl ``\mathcal{C}_h`` acts on the components involving horizontal
    derivatives [`ClimaCore.Operators.Curl`](@extref). Applied to
    ``\boldsymbol{u}_h`` it returns a vector with only vertical contravariant
    components; applied to ``\boldsymbol{u}_v`` it returns a vector with only
    horizontal contravariant components.

  - The corresponding weak curl ``\hat{\mathcal{C}}_h`` is
    `Curl{WeakForm}` (see [`ClimaCore.Operators.Curl`](@extref)).

  - The completion step ``\mathcal{P}`` across element boundaries is direct
    stiffness summation on the continuous elements ClimaAtmos uses; see
    [Completing a tendency across elements](@ref).

### Vertical differential operators

  - The face-to-center vertical divergence ``D^c`` is
    [`ClimaCore.Operators.DivergenceF2C`](@extref). Separate variants impose the
    boundary conditions for advective, precipitation, and diffusive fluxes.

  - The center-to-face vertical gradient ``G^f`` is
    [`ClimaCore.Operators.GradientC2F`](@extref), set to zero at the top and
    bottom boundaries. The boundary values are placeholders: the vertical
    velocity at the boundaries is fixed instead by the impenetrability
    condition described above.

  - The face-to-center vertical gradient ``G^c`` is
    [`ClimaCore.Operators.GradientF2C`](@extref), used where a face-defined
    field such as the geopotential is differentiated onto centers.

  - The center-to-face curl ``C^f`` acts on the components involving
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

## Assembling a discretized equation

The operators above combine in the same way in every prognostic equation. Two
reconstructed velocities and the kinetic energy built from them appear
throughout; all follow from the staggering, since the covariant vertical
component lives on faces and everything else on centers:

```math
\tilde{\boldsymbol{u}} = WI^f(\rho J, \boldsymbol{u}_h) + \boldsymbol{u}_v ,
\qquad
\bar{\boldsymbol{u}} = \boldsymbol{u}_h + I^c(\boldsymbol{u}_v) ,
\qquad
K = \tfrac{1}{2} \left( \boldsymbol{u}_h \cdot \boldsymbol{u}_h
  + 2 \boldsymbol{u}_h \cdot I^c(\boldsymbol{u}_v)
  + I^c(\boldsymbol{u}_v \cdot \boldsymbol{u}_v) \right) .
```

``\tilde{\boldsymbol{u}}`` is the face velocity (see `compute_ᶠuₕ³` in
`src/cache/precomputed_quantities.jl`), ``\bar{\boldsymbol{u}}`` the
center velocity, and ``K`` the specific kinetic energy at centers. The no-flux
condition at the surface and the model top fixes the covariant vertical
component from the horizontal one, ``u_3 = -(g^{31} u_1 + g^{32} u_2)/g^{33}``,
so that the contravariant ``\tilde{u}^3`` vanishes there; see
[Topography](topography.md).

### One equation in full: total energy

Total energy shows every pattern the other equations reuse. Its continuous form
is

```math
\frac{\partial}{\partial t} \rho e_{tot}
  = - \nabla \cdot \left[ (\rho e_{tot} + p) \boldsymbol{u} + \boldsymbol{F}_R \right]
  + \rho S_e ,
```

stabilized by fourth-order hyperdiffusion of the total enthalpy, decomposed into
dry static energy and a water-enthalpy piece (see
[Hyperdiffusion](hyperdiffusion.md)). Discretized, it reads

```math
\frac{\partial}{\partial t} \rho e_{tot} \approx
- \mathcal{D}^{split}_h \left[ \rho \bar{\boldsymbol{u}},
    \tfrac{\rho e_{tot} + p}{\rho} \right]
- D^c \left[ \mathcal{M}^f(\rho) \, I^f \left( \tfrac{\rho e_{tot} + p}{\rho} \right) \right]
- D^c \left[ \boldsymbol{F}_R \right]
- \nu_h \left[ \hat{\mathcal{D}}_h \left( \rho \, \mathcal{G}_h(\psi_{s_d}) \right)
  + \hat{\mathcal{D}}_h \left( \rho \, h_{\mathrm{tot,cl}} \,
      \mathcal{G}_h(\psi_{q_t^{\mathrm{eff}}}) \right) \right],
\qquad
\psi_x = \mathcal{P} \left[ \hat{\mathcal{D}}_h \left( \mathcal{G}_h (x - x_r) \right) \right],
```

with ``x_r`` the hydrostatic reference profile subtracted before the first
Laplacian. Reading the terms in order:

  - The horizontal advective flux uses the split divergence
    ``\mathcal{D}^{split}_h`` of the center velocity, the entropy-stable form
    from the rule above.
  - The vertical advective flux is the face mass flux ``\mathcal{M}^f(\rho)``
    times the centrally interpolated enthalpy, differenced with ``D^c``. This
    term is **implicit**. When upwinding is enabled for energy (the van Leer
    limiter by default), the implicit solve still uses this central form, and
    the difference between the upwinded and central fluxes is applied after the
    Newton solve as a `T_post_imp!` correction
    (`correct_implicit_advection_tendency!`), so the upwind flux sees the
    Newton-solved velocity.
  - The radiative flux divergence is explicit and applied separately.
  - Hyperdiffusion is two Laplacians with the completion ``\mathcal{P}`` in
    between, each a weak divergence of a strong gradient.

### The other equations

The remaining equations follow the same template; the table gives the operator
each term uses and where it lives. Everything in the "Implicit" column is
solved column by column as described under [Timestepping](@ref); everything
else is explicit.

| Equation                                 | Horizontal                                                                                                                                        | Vertical                                                                                                                                                                             | Implicit part                                               | Source                                                                                                                          |
|:---------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |:----------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------- |
| Mass ``\rho``                            | ``\hat{\mathcal{D}}_h[\rho \bar{\boldsymbol{u}}]``, which is ``\mathcal{D}^{split}_h`` with unit scalar                                           | ``D^c[\mathcal{M}^f(\rho)]``                                                                                                                                                         | the vertical flux, with the full ``\tilde{\boldsymbol{u}}`` | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl)                          |
| Horizontal momentum ``\boldsymbol{u}_h`` | strong gradient ``\mathcal{G}_h`` of ``\Phi - \Phi_r + K`` and the split-form pressure gradient; weak curls ``\hat{\mathcal{C}}_h`` for vorticity | ``I^c\{\boldsymbol{\omega}^h \times I^f(\rho J)\tilde{\boldsymbol{u}}^v\} / \rho J``, with ``\boldsymbol{\omega}^h = C^f[\boldsymbol{u}_h] + \hat{\mathcal{C}}_h[\boldsymbol{u}_v]`` | none                                                        | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl)                          |
| Vertical momentum ``u_3``                | the same vorticity terms, ``\times I^f(\boldsymbol{u}^h)``                                                                                        | ``G^f[K]``, and ``I^f[c_{pd}\theta_v'] \, G^f[\Pi] + G^f[\Phi - \Phi_r]``                                                                                                            | the pressure-gradient and geopotential term                 | [implicit_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/implicit_tendency.jl) |
| Total water ``\rho q_t``                 | ``\mathcal{D}^{split}_h[\rho \bar{\boldsymbol{u}}, q_t]``                                                                                         | ``D^c[\mathcal{M}^f(\rho) I^f(q_t)]``, upwind correction after the solve as for energy                                                                                               | the central vertical flux                                   | [implicit_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/implicit_tendency.jl) |
| Other grid-mean tracers ``\rho \chi``    | ``\mathcal{D}^{split}_h[\rho \bar{\boldsymbol{u}}, \chi]``                                                                                        | ``D^c\left[\tfrac{I^f(\rho J)}{J^f} U^f(\tilde{\boldsymbol{u}}, \chi)\right]`` with `tracer_upwinding`                                                                               | none                                                        | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl)                          |

Two details of the momentum rows are worth spelling out. The pressure-gradient
and geopotential terms in the momentum equation are written as a departure from
a hydrostatic reference state ``(\theta_{v,r}, \Phi_r)``, with
``\theta_v' = \theta_v - \theta_{v,r}``; the horizontal part uses the split
form ``\tfrac{c_{pd}}{2}\{\theta_v' \mathcal{G}_h[\Pi] + \mathcal{G}_h[\theta_v' \Pi]

  - \Pi \mathcal{G}_h[\theta_v']\}``, and the reference profile itself is on the [Governing Equations](equations.md) page. And the momentum hyperviscosity is the vector Laplacian of the rule table applied twice,``-\nu_u \{ \delta_{div} \hat{\mathcal{G}}_h(\mathcal{D}_h \boldsymbol{\psi})
  - \hat{\mathcal{C}}_h(\mathcal{C}_h \boldsymbol{\psi}) \}``with``\boldsymbol{\psi} = \mathcal{P}[\hat{\mathcal{G}}_h(\mathcal{D}_h \bar{\boldsymbol{u}})
  - \hat{\mathcal{C}}_h(\mathcal{C}_h \bar{\boldsymbol{u}})]``, projected onto the horizontal covariant directions for``\boldsymbol{u}_h``and onto the vertical one, after a``\rho J``-weighted interpolation to faces, for``u_3``; the divergence damping factor``\delta_{div}`` is described under
    [Hyperdiffusion](hyperdiffusion.md).

The fully expanded forms of each tendency are in the source files linked above
and in Appendix B of [Yatunin2026](@cite).

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
