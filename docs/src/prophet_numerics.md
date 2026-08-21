# PROPHET: Discretization and Time Stepping

The [PROPHET equations](prophet.md) are continuous. This page gives their
discrete form: the spatial discretization, the implicit–explicit split, and the
two analytic implicit-stage solves that make the scheme run at the dynamical
core's timestep rather than at the updraft CFL limit. It also collects the
regularizations that keep the scheme well behaved where the drafts are nearly
empty or nearly fill the cell.

Two things make the draft equations harder to time-step than the resolved ones.
Updraft vertical velocities are large, ``O(10)`` m s⁻¹, while the vertical grid
spacing near the surface is small, ``O(10)`` m, so an explicit treatment of the
vertical draft advection would cap the timestep at a few seconds. The draft mass
variable ``\hat{\rho}^j`` can also approach zero, and the equations divide by it.
Both are handled at the time-discretization level.

## Spatial discretization

PROPHET uses the same discretization as the dynamical core: a horizontal
spectral-element method and vertical finite differences on a Lorenz staggered
grid, with the draft vertical velocity ``u_3^j`` on faces and everything else at
cell centers. The operators, the strong/weak/split-form choices, and the
notation used below are defined once in
[Discretization and Operators](discretization.md); this page uses them without
restating them.

Two reconstructed velocities appear, as in the resolved equations:

```math
\tilde{\boldsymbol{u}}^j = WI^f(\rho J, \boldsymbol{u}_h) + \boldsymbol{u}_v^j ,
\qquad
\bar{\boldsymbol{u}}^j = \boldsymbol{u}_h + I^c(\boldsymbol{u}_v^j) .
```

Note the weight in the face reconstruction. Mass conservation for the draft
would call for ``\hat{\rho}^j J``, but that weight vanishes with the draft area;
the code uses the grid-mean weight ``\rho J``, the same weighted interpolation
as for the resolved velocity. The horizontal part of the face velocity is
therefore shared with the grid mean, and only the vertical covariant component
is draft-specific.

The draft kinetic energy at centers is built the same way as the resolved one,

```math
\kappa^j = \tfrac{1}{2} \left( \boldsymbol{u}_h \cdot \boldsymbol{u}_h
  + 2 \boldsymbol{u}_h \cdot I^c(\boldsymbol{u}_v^j)
  + I^c(\boldsymbol{u}_v^j \cdot \boldsymbol{u}_v^j) \right) ,
```

and the purely vertical part

```math
\kappa_v^j = \tfrac{1}{2} \, \boldsymbol{u}_v^j \cdot \boldsymbol{u}_v^j
```

is kept separately at faces (`ᶠKᵥʲs`), because the vertical advection it
generates is treated implicitly and must be excluded from the explicit
kinetic-energy gradient.

No-flux boundary conditions are imposed exactly as for the resolved velocity.
The third contravariant component of the face velocity is required to vanish at
the surface and at the model top, which fixes the covariant component there:

```math
u_3^j = - \frac{g^{31} u_1 + g^{32} u_2}{g^{33}} .
```

### Draft mass

The flux-form continuity equation is discretized as

```math
\frac{\partial \hat{\rho}^j}{\partial t}
= - \hat{\mathcal{D}}_h \left[ \hat{\rho}^j \bar{\boldsymbol{u}}^j \right]
  - \mathcal{D}^c_v \left[ WI^f(J, \hat{\rho}^j) \, \tilde{\boldsymbol{u}}^j \right]
  + \text{sources} ,
```

with the horizontal term explicit (in the code, `split_divₕ(ρa ᶜuʲ, 1)`, which
for a unit scalar reduces to the weak divergence) and the vertical term folded
into the implicit-stage solve described below.

### Draft scalars

All draft scalars are advected in advective form. Horizontally, the code uses
the split, skew-symmetric divergence in the combination that makes the advective
form exact,

```math
- \mathcal{D}^{split}_h(\bar{\boldsymbol{u}}^j, \psi^j)
+ \psi^j \, \mathcal{D}^{split}_h(\bar{\boldsymbol{u}}^j, 1) ,
```

so that a uniform ``\psi^j`` produces no tendency regardless of the divergence
of the draft velocity. Vertically, the same construction is applied with the
vertical divergence,

```math
- \mathcal{D}^c_v \left[ U^f(\tilde{\boldsymbol{u}}^j, \psi^j) \right]
+ \psi^j \, \mathcal{D}^c_v \left[ \tilde{\boldsymbol{u}}^j \right] ,
```

so that the reconstruction and the divergence operator are the same ones the
flux form would use. The reconstruction ``U^f`` is selected by the
configuration:
`edmfx_mse_q_tot_upwinding` for ``h_s^j`` and ``q_t^j`` (default
`first_order`), `edmfx_tracer_upwinding` for the remaining draft tracers
(default `first_order`), and `none` for the central reconstruction. The
resolvability argument of [the high-resolution limit](prophet.md#The-high-resolution-limit)
bears on this choice: upwinding adds numerical diffusion to the draft variables,
which acts like a residual subgrid flux and so competes with the physical
closure as the grid is refined.

The buoyancy source in the energy equation is evaluated at centers from the
face-interpolated draft velocity and the normalized density excess,

```math
\left[ \frac{\partial h_s^j}{\partial t} \right]_{\mathrm{buoy}}
= I^c(\boldsymbol{u}_v^j) \cdot \frac{\rho^j - \rho}{\rho^j} \, \mathcal{G}^f_v[\Phi] .
```

### Draft vertical momentum

Breaking the curl and cross-product terms into horizontal and vertical
contributions and dropping the terms that vanish
(``\nabla_v \times \boldsymbol{u}_v = 0``) leaves, for the explicit part,

```math
\frac{\partial \boldsymbol{u}_v^j}{\partial t}
= - \left( \mathcal{C}^f_v[\boldsymbol{u}_h]
    + \mathcal{C}_h[\boldsymbol{u}_v^j] \right) \times I^f(\boldsymbol{u}_h)
  - \mathcal{G}^f_v \left[ \kappa^j - I^c(\kappa_v^j) \right]
  + \text{hyperdiffusion} .
```

Subtracting ``I^c(\kappa_v^j)`` removes the purely vertical kinetic-energy
gradient, the ``\partial_z(w^2/2)`` advection, which the implicit stage solve
below carries instead. Buoyancy, the pressure drag, and the entrainment sink are
likewise in the implicit solve.

### Subgrid-scale fluxes onto the grid mean

The mass-flux contribution is discretized in the difference form of
[Coupling to the resolved equations](prophet.md#Coupling-to-the-resolved-equations):
for each subdomain ``k`` (drafts and environment), the tendency added to a
grid-mean scalar is the vertical transport of ``a^k (\chi^k - \chi)`` by the
velocity difference,

```math
- \mathcal{D}^c_v \left[ WI^f(J, \rho^k) \, U^f \!\left(
    \tilde{\boldsymbol{u}}^k - \tilde{\boldsymbol{u}}, \;
    a^k (\chi^k - \chi) \right) \right] ,
```

reconstructed with `edmfx_sgsflux_upwinding` (default `none`, i.e. central).
Writing the flux with both differences taken explicitly is what makes it vanish
identically when ``\chi^k = \chi``, whatever the area gradients. For energy, the
transported scalar is ``h_s^j + \kappa^j`` for the drafts and the grid-mean
enthalpy is subtracted; the ``q_t`` flux also increments ``\rho``,
because subgrid moisture transport moves moist air mass.

The diffusive fluxes are evaluated at faces from the face-native diffusivities
``K_h``, ``K_u``, ``K_e`` (see
[Closures](prophet_closures.md#Capping-inversions-as-unresolved-interfaces)).
Two structural details matter for reading the code:

  - The turbulent-mixing component ``K_h`` acts on
    ``q_{t,\mathrm{eff}} = q_t - q_r - q_s``, the suspended water, distributed
    to the cloud species by tendency scaling; rain and snow receive no turbulent
    transport, so that rain shafts dominated by sedimentation are not smeared.
    Energy diffuses as ``-K_h [\nabla s_d + (h_{\mathrm{eff}} + \Phi) \nabla q_{t,\mathrm{eff}}]``,
    the dry-static-energy-plus-water-enthalpy decomposition.
  - The interfacial-entrainment component ``K_e`` acts per species on each
    species' own gradient, and on ``\nabla h_{\mathrm{tot}}`` for energy, because
    interfacial entrainment transports every constituent bodily with the parcel.

When `edmfx_vertical_diffusion` is enabled, each draft receives the same
*specific* tendency the grid mean receives, which is the discrete form of the
grid-mean-divergence requirement of [Thuburn2022](@cite).

### Hyperdiffusion

Draft variables are hyperdiffused for numerical stability, along
terrain-following coordinate surfaces, using the decomposed and unweighted
subdomain fields: each
subdomain inherits the grid-mean specific hyperdiffusive tendency, with the
energy and moisture contributions split into a dry-static-energy term and a
water term as in the grid mean. See [Hyperdiffusion](hyperdiffusion.md) for the
operator and the reference-state subtraction.

## The implicit–explicit split

PROPHET terms are distributed between the two halves of the IMEX splitting (see
[Implicit Solver](implicit_solver.md)) like this.

**Implicit** (`implicit_tendency!`):

  - vertical advection of the draft scalars, the buoyancy source in the draft
    energy equation, and draft sedimentation
    (`edmfx_sgs_vertical_advection_tendency!`);
  - entrainment relaxation of the draft scalars
    (`edmfx_entr_detr_tendency!`);
  - the subgrid-scale mass fluxes onto the grid mean
    (`edmfx_sgs_mass_flux_tendency!`);
  - the surface mass-source tendency on the draft scalars
    (`edmfx_boundary_condition_tendency!`);
  - the cached ``u_3^j`` and ``\hat{\rho}^j`` stage tendencies (below);
  - the diffusive fluxes, when `implicit_diffusion` is set.

**Explicit** (`remaining_tendency!`):

  - horizontal advection of all draft variables
    (`horizontal_dynamics_tendency!` for ``\hat{\rho}^j`` and ``h_s^j``,
    `horizontal_tracer_advection_tendency!` for ``q_t^j`` and the draft
    tracers);
  - the explicit part of the draft vertical momentum equation (vorticity terms
    and the horizontal/cross kinetic-energy gradient), in
    `explicit_vertical_advection_tendency!`;
  - TKE shear and buoyancy production (`edmfx_tke_tendency!`);
  - hyperdiffusion, the Rayleigh sponge relaxation of the draft scalars toward
    the grid mean, and the optional horizontal diffusive fluxes;
  - the diffusive fluxes, when `implicit_diffusion` is not set.

Configurations that use PROPHET normally set `implicit_diffusion: true` together
with `approximate_linear_solve_iters: 2`, because the diffusive coupling between
the grid mean and the drafts is stiff at the timesteps of interest.

## Analytic implicit-stage solves

The draft mass ``\hat{\rho}^j`` and vertical velocity ``u_3^j`` are *not*
advanced by the Newton solve. At the start of every implicit stage, before the
solve, `initialize_implicit_stage_problem!` overwrites them in the state with
values obtained from closed-form column solves, and caches the implied
tendencies

```math
\frac{u_3^{j,\mathrm{stage}} - u_3^{j,\mathrm{old}}}{\gamma \Delta t} ,
\qquad
\frac{\hat{\rho}^{j,\mathrm{stage}} - \hat{\rho}^{j,\mathrm{old}}}{\gamma \Delta t} ,
```

which `sgs_u₃_implicit_tendency!` and `sgs_ρa_implicit_tendency!` then return as
the implicit tendencies of those variables. Because the tendencies are
*assigned* rather than accumulated, and the corresponding Jacobian rows are
identity blocks, the Newton solve reproduces the analytic stage values exactly
and leaves them alone. The result is less general than a Newton solve but
unconditionally robust: each solve is exact, needs one sweep through the column,
and is constructed so that it cannot produce a negative area or a downward
updraft.

### Vertical velocity

The evolution equation for the physical vertical velocity of a draft, retaining
the terms treated implicitly, is

```math
\frac{\partial w^j}{\partial t}
  + \frac{\partial}{\partial z} \frac{(w^j)^2}{2}
= (1 - \alpha_b) \, b^j + E^j (w^0 - w^j) - d^j ,
```

with ``\alpha_b`` the virtual-mass coefficient and ``d^j`` the form drag of
[Pressure closure](prophet_closures.md#Pressure-closure), quadratic in
``w^j - w^0``. At an IMEX stage, this becomes an algebraic equation in the new
stage value. The environment velocity is eliminated using the mass-flux
constraint with the resolved vertical velocity neglected,

```math
w^0 - w^j \approx - \frac{\rho}{\hat{\rho}^0} \, w^j ,
```

which is exact for a single draft and is applied unchanged when there are
several. The velocity-independent entrainment rates (background, buoyancy-driven,
area-bounding, and turbulent) then contribute a *linear* sink in ``w^j``, the
velocity-proportional entrainment contributes a *quadratic* sink, and the form
drag is purely quadratic. The prognostic variable is the covariant component
``u_3 = w \Delta z``, so the whole equation carries one factor of ``\Delta z``
relative to the equation for ``w``. At face ``i`` the stage equation is then a
quadratic, coupled to the face below through the advection term:

```math
\mathcal{A} \, u_3^2 + \mathcal{B} \, u_3 + \mathcal{C}
  - \frac{1}{2}\left(\frac{u_3[i-1]}{\Delta z}\right)^2 = 0 .
```

The coefficients ``(\mathcal{A}, \mathcal{B}, \mathcal{C})`` (`ᶠa`, `ᶠb`, `ᶠc`
in the code) collect the stage term ``1/(\gamma \Delta t)``, the entrainment
rates, the form drag (only when `edmfx_nh_pressure` is enabled), the Rayleigh
sponge damping rate (only when a sponge is configured), the reduced buoyancy,
and the local part of the vertical kinetic-energy advection. The equation is
swept upward with `Operators.column_accumulate!`, each face taking the ``+``
root. Clamping the constant term with ``\min(0, \cdot)`` keeps the discriminant
non-negative and the root non-negative, so the solve can never return a downward
draft velocity.

### Draft mass

The flux-form stage equation
``\partial \hat{\rho}^j / \partial t + \partial_z(\hat{\rho}^j w^j) = (E^j - \Delta^j) \hat{\rho}^j``
under first-order upwinding (for upward ``u_3^j``) is a forward recurrence in
the column,

```math
\hat{\rho}^j[i] = \frac{n[i] + \alpha_{\mathrm{bot}}[i] \, \hat{\rho}^j[i-1]}{d[i]} ,
\qquad
n[i] = \frac{\hat{\rho}^{j,\mathrm{old}}[i]}{\gamma \Delta t} ,
```

```math
d[i] = \max\!\left( \frac{0.1}{\gamma \Delta t}, \;
  \frac{1}{\gamma \Delta t} + \alpha_{\mathrm{top}}[i]
  - (E^j - \Delta^j)[i] \right) ,
\qquad
\alpha_{\mathrm{face}} = \frac{I^f(\rho^j J)}{J^f}
  \frac{u_3^j / \Delta z^f}{\rho^j_{\mathrm{upwind}} \, \Delta z} ,
```

swept bottom to top, with the upwind density ``\rho^j[i]`` at the top face and
``\rho^j[i-1]`` at the bottom face. The floor on the denominator caps the
per-step growth at roughly a factor of ten.

The recurrence routes detrainment, the surface source, and the area bound in
ways that are easy to misread:

  - **Detrainment enters in two ways.** The area-bounding, velocity-scale
    entrainment, and buoyancy-detrainment pieces appear in ``(E^j - \Delta^j)``.
    The *mass-flux-divergence* component of detrainment does not; it becomes a
    multiplicative prefactor on ``\alpha_{\mathrm{top}}`` and
    ``\alpha_{\mathrm{bot}}``, so that it is treated implicitly together with the
    flux divergence it derives from. Where the draft is at ``a_{\max}``, that
    prefactor is one and all converging mass is detrained.
  - **The surface source enters the first cell's numerator.** In cell 1 the
    bottom coefficient is zeroed, since the physical flux is zero there
    (``u_3 = 0`` at the surface), and the capped surface mass source
    ``F_{\mathrm{sfc}} / \Delta z`` is added to the numerator as an
    area-independent constant. That is the discrete form of the free lower
    boundary condition on the draft area.
  - **The area is clamped, asymmetrically.** The limiters in
    ``(E^j - \Delta^j)`` are evaluated at the previous iterate, so they
    cannot guarantee the bound at the stage value. The sweep therefore clamps
    ``\hat{\rho}^j`` to ``[0, \rho^j a_{\max}]``. Clipping from *above* is
    mass-conserving: the excess is absorbed by the environment automatically,
    acting like instantaneous detrainment. Clipping from below at
    ``\rho^j a_{\min}`` would not be, since it would create draft mass out of
    nothing, so the lower bound is zero.

## Jacobian

The implicit residual is linearized by the Jacobian algorithms of
[Implicit Solver](implicit_solver.md). For `ManualSparseJacobian`, the PROPHET
blocks are assembled by five update functions that mirror the tendency
structure, in a fixed order because they accumulate into shared blocks:

| Update                                     | Covers                                                                                                      |
|:------------------------------------------ |:----------------------------------------------------------------------------------------------------------- |
| `update_sgs_advection_jacobian!`           | vertical advection of the draft scalars (assigns the diagonals, including the ``-I`` residual term) and draft sedimentation with its lateral-mixing correction |
| `update_sgs_diffusion_jacobian!`           | the diffusive coupling between the grid mean and the drafts, including ``K_e``                               |
| `update_sgs_entr_detr_jacobian!`           | the entrainment relaxation, including the feedback through the environment residual                          |
| `update_sgs_boundary_condition_jacobian!`  | the surface mass-source relaxation in the first cell                                                         |
| `update_sgs_massflux_jacobian!`            | the mass-flux contributions to the grid-mean scalars                                                         |

The ``\hat{\rho}^j`` and ``u_3^j`` rows receive fallback identity blocks,
since their stage values come from the analytic solves. The block solver
eliminates the sedimenting draft tracers first, then the draft ``q_t`` and
``h_s``, then the grid-scale condensate masses, then ``\rho`` and the surface
state, then ``\rho q_t``, and finally the remaining scalars including
``\rho e_{tot}``; the two velocity groups (``\boldsymbol{u}_h``, then
``u_3^j``) are solved with a block lower-triangular solve. The
automatic-differentiation algorithms
([`AutoSparseJacobian`](@ref ClimaAtmos.AutoSparseJacobian),
[`AutoDenseJacobian`](@ref ClimaAtmos.AutoDenseJacobian)) cover the same terms
without a hand-written derivative, and the `*_sparse_autodiff` and
`*_dense_autodiff` configurations use them to check the manual one.

## Regularizations and state filters

Several devices keep the scheme well behaved at the edges of its validity. Each
introduces a small inconsistency in order to stay robust, and each shows up in
the code and in the diagnostics, so it helps to know which is which.

### Environment reconstruction

Recovering a specific environment scalar from the residual requires dividing by
``\hat{\rho}^0``, which vanishes as the drafts fill the cell. `specific` blends
the residual quotient with the grid-mean quotient,

```math
\psi^0 = \mathcal{W} \, \frac{\rho \psi
    - \sum_{j} \hat{\rho}^j \psi^j}{\hat{\rho}^0}
  + (1 - \mathcal{W}) \, \psi ,
```

and returns the grid-mean value outright when ``\hat{\rho}^0`` falls below
machine epsilon, where even a zero weight would leave a ``0/0`` that automatic
differentiation turns into `NaN`s. The weight ``\mathcal{W}`` is a sigmoid in
the environment area fraction ``\hat{\rho}^0/\rho`` that is zero at zero area,
one half at ``a_{1/2}``, one above ``42 a_{1/2}``, and continuously
differentiable with vanishing endpoint derivatives, so the blend introduces no
kinks. Where the area fraction is small, the subdomains then no longer sum
exactly to the grid mean. ``a_{1/2}`` is the `a_half` field of
[`PrognosticEDMFX`](@ref ClimaAtmos.PrognosticEDMFX), which
`get_turbconv_model` sets to the minimum draft area `EDMF_min_area`.

### State filters

With `edmfx_filter: true`, `enforce_edmf_updraft_constraints!` runs after every
timestepper stage (from `constrain_state!`) and, for each draft:

  - clips ``\hat{\rho}^j`` to ``[0, \rho^j]`` and ``u_3^j`` to non-negative
    values, and zeroes ``u_3^j`` where the face-interpolated ``\hat{\rho}^j`` is
    negligible;
  - relaxes ``h_s^j`` and ``q_t^j`` to the grid-mean values where
    ``\hat{\rho}^j`` is negligible, and otherwise bounds each draft scalar so
    that ``\hat{\rho}^j \chi^j \le \rho \chi``;
  - rescales the draft condensate species so that their sum does not exceed
    ``q_t^j``, mirroring the grid-mean constraint.

These are *filters*, not conservative corrections; the clipping of
``\hat{\rho}^j`` from above and the scalar bounds are mass-conserving in the
sense that the environment absorbs the difference, but the relaxation toward the
grid mean is not. `edmfx_filter` is enabled in every PROPHET configuration in
`config/model_configs/` except the advection test.

### Other limiters

  - The entrainment and detrainment area limiters and the additive area-bounding
    rate, described in [Closures](prophet_closures.md#Entrainment-and-detrainment).
  - A floor of 1 m on the mixing length, which prevents division by zero in the
    TKE dissipation.
  - Negative TKE is relaxed to zero within one timestep rather than being
    clipped, in `edmfx_sgs_diffusive_flux_tendency!`.
  - Draft entrainment prefactors return zero at or below the surface, where
    ``1/(z - z_s)`` is singular.

## Update order within a timestep

The precomputed quantities the tendencies read are set in two passes, and the
order matters because the closures depend on each other:

1. `set_implicit_precomputed_quantities!`: grid-mean velocities, thermodynamics
   and pressure, then the draft diagnosed quantities
   (`set_prognostic_edmf_precomputed_quantities_draft!`: draft velocities,
   kinetic energies, temperatures, densities) and the environment residuals
   (`..._environment!`). Called before every implicit tendency evaluation, so
   these are consistent with the current Newton iterate.
2. `set_explicit_precomputed_quantities!`: surface conditions, then the
   PROPHET explicit closures (entrainment and detrainment rates, the surface
   mass-source payload), then the coupled covariance/cloud-fraction Picard
   iteration, then the face diffusivities (which need the final cloud fraction),
   then the center mixing length, terminal velocities, and the microphysics
   tendency cache. Called before every explicit tendency evaluation.

The face diffusivities are computed *after* the cloud fraction because the
buoyancy gradient that enters ``N_{e,\mathrm{eff}}^2`` depends on it; the
mixing length is materialized inside the Picard iteration when the configuration
uses covariances, and separately otherwise, with `uses_covariances` as the
shared predicate that keeps the two paths from disagreeing.

## Where this is implemented

| Component                                        | Source                                                                                                                                                                                            |
|:------------------------------------------------ |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implicit/explicit split                          | [implicit/implicit_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/implicit_tendency.jl), [remaining_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/remaining_tendency.jl) |
| Analytic stage solves for ``u_3^j`` and ``\hat{\rho}^j`` | [implicit/initialize_implicit_problem.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/initialize_implicit_problem.jl)                                |
| Horizontal advection of draft variables          | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl)                                                                                             |
| Vertical advection, buoyancy, sedimentation      | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl), [water_advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/water_advection.jl) |
| Subgrid-scale flux discretization                | [edmfx_sgs_flux.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_sgs_flux.jl)                                                                                   |
| Draft hyperdiffusion                             | [hyperdiffusion.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/hyperdiffusion.jl)                                                                                   |
| Jacobian blocks                                  | [implicit/manual_sparse_jacobian.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/manual_sparse_jacobian.jl)                                                 |
| State filters                                    | [mass_flux_closures.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/mass_flux_closures.jl), [constrain_state.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/constrain_state.jl) |
| Environment reconstruction and the blend weight  | [utils/variable_manipulations.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/utils/variable_manipulations.jl)                                                                            |
| Precomputed-quantity ordering                    | [cache/precomputed_quantities.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cache/precomputed_quantities.jl), [cache/prognostic_edmf_precomputed_quantities.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cache/prognostic_edmf_precomputed_quantities.jl) |
