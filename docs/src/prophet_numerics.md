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
variable ``\hat{\rho}^j`` can also approach zero, and the equations divide by
it. Both are handled at the time-discretization level.

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
  - D^c \left[ \frac{I^f(\hat{\rho}^j J)}{J^f} \, \tilde{\boldsymbol{u}}^j \right]
  + \text{sources} ,
```

with the horizontal term explicit (in the code, `split_divₕ(ρa ᶜuʲ, 1)`, which
for a unit scalar reduces to the weak divergence) and the vertical term
integrated in the post-ARK two-stream column sweep described below. Over
terrain, the vertical flux of ``\hat{\rho}^j`` and of every draft scalar uses the
full contravariant draft velocity ``u^{3j} = u_h^3 + g^{33} u_3^j``, with the
same face value of ``u_h^3`` as the grid-mean ``u^3``, so the draft velocity
relative to the grid mean is ``u^{3j} - u^3 = g^{33} (u_3^j - u_3)``.

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
- D^c \left[ U^f(\tilde{\boldsymbol{u}}^j, \psi^j) \right]
+ \psi^j \, D^c \left[ \tilde{\boldsymbol{u}}^j \right] ,
```

so that the reconstruction and the divergence operator are the same ones the
flux form would use. The reconstruction ``U^f`` is selected by the
configuration: `edmfx_mse_humidity_upwinding` for ``h_s^j``, ``q_t^j``, and the
sedimenting draft microphysics species (default `first_order`), so that the
draft energy and water cross each face with one reconstruction;
`edmfx_tracer_upwinding` for the remaining draft tracers (default
`first_order`); and `none` for the central reconstruction. The
resolvability argument of
[the high-resolution limit](prophet.md#The-high-resolution-limit) bears on this
choice: upwinding adds numerical diffusion to the draft variables, which acts
like a residual subgrid flux and so competes with the physical closure as the
grid is refined.

The buoyancy source in the energy equation is evaluated at centers from the
face-interpolated draft velocity and the normalized density excess,

```math
\left[ \frac{\partial h_s^j}{\partial t} \right]_{\mathrm{buoy}}
= I^c(\boldsymbol{u}_v^j) \cdot \frac{\rho^j - \rho}{\rho^j} \, G^c[\Phi] .
```

Under non-equilibrium microphysics, `step_coupled_microphysics_sedimentation!`
integrates draft condensate sedimentation and microphysical phase changes
top-down per column. For each sedimenting draft condensate or precipitation
species ``q_\mu^{\sigma j}``, sedimentation exchanges mass with the
environment where ``\partial_z a^j \ne 0``, and each intensive draft scalar
``\chi^j`` receives ``(S_\chi - \chi^j S) / \hat{\rho}^j``, with ``S = \sum_\mu S_\mu``
the draft-mass sedimentation source, ``S_\chi`` its own sedimentation tendency
(``S_\mu`` for ``q_\mu^{\sigma j}``, ``S`` for ``q_t^j``, and the condensate
enthalpy sedimentation source for ``h_s^j``), and zero for the other draft
tracers.

### Draft vertical momentum

Breaking the curl and cross-product terms into horizontal and vertical
contributions and dropping the terms that vanish
(``\nabla_v \times \boldsymbol{u}_v = 0``) leaves, for the explicit part,

```math
\frac{\partial \boldsymbol{u}_v^j}{\partial t}
= - \left( C^f[\boldsymbol{u}_h]
    + \mathcal{C}_h[\boldsymbol{u}_v^j] \right) \times I^f(\boldsymbol{u}_h)
  - G^f \left[ \kappa^j - I^c(\kappa_v^j) \right]
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
- D^c \left[ \frac{I^f(\rho^k J)}{J^f} \, U^f \!\left(
    \tilde{\boldsymbol{u}}^k - \tilde{\boldsymbol{u}}, \;
    a^k (\chi^k - \chi) \right) \right] ,
```

reconstructed with `edmfx_sgsflux_upwinding` (default `none`, i.e. central).
Writing the flux with both differences taken explicitly makes it vanish
identically when ``\chi^k = \chi``, whatever the area gradients. For energy, the
transported scalar is ``h_s^j + \kappa^j`` for the drafts and the grid-mean
enthalpy is subtracted; the ``q_t`` flux also increments ``\rho``,
because subgrid moisture transport moves moist air mass.

The diffusive fluxes are evaluated at faces from the face-native diffusivities
``K_h``, ``K_u`` (see
[Closures](prophet_closures.md#Unresolved-inversions)):

  - The turbulent-mixing component ``K_h`` acts on the suspended water
    ``q_{t,\mathrm{eff}} = q_t - q_{\mathrm{rai}} - q_{\mathrm{sno}}`` and on
    the suspended cloud species (`ρq_lcl`, `ρq_icl`, `ρn_lcl`, `ρn_ice`), which
    diffuse down their own gradients (species the drafts do not carry, such as
    the P3 ice number and rime quantities, down their grid-mean gradients); rain and snow receive no turbulent
    transport, so that rain shafts dominated by sedimentation are not smeared.
    Energy diffuses through the bracket of [Diffusion](diffusion.md),
    ``\mathcal{B} = c_p \nabla T + \nabla \Phi + (h_v + \Phi) \nabla q_{t,\mathrm{eff}} + (h_l - h_v) \nabla q_l^{\mathrm{cl}} + (h_i - h_v) \nabla q_i^{\mathrm{cl}}``,
    with ``c_p`` the heat capacity of the diffusing mixture and each
    thermodynamic coefficient evaluated at centers and interpolated to faces.
  - For `PrognosticEDMFX`, the grid-mean diffusive flux sums the environment and
    draft contributions with face weights ``w^0 = I^f(\hat{\rho}^0) K_h`` and
    ``w^j = f_j K_h / I^f(1/\max(\hat{\rho}^j, \epsilon))``
    (``f_j =`` `edmfx_updraft_kh_diffusion_factor`), and when
    `edmfx_vertical_diffusion` is enabled each draft scalar
    ``\chi^j \in \{h_s^j, q_t^j, q_l^{\mathrm{cl},j}, q_i^{\mathrm{cl},j}, n_l^{\mathrm{cl},j}, n_i^{\mathrm{cl},j}\}``
    receives ``-\max(\hat{\rho}^j, \epsilon)^{-1} D^c[-w^j G^f[\chi_{\mathrm{eff}}^j]]``,
    where ``q_{t,\mathrm{eff}}^j = q_t^j - q_r^j - q_s^j`` and
    ``h_{s,\mathrm{eff}}^j`` excludes non-diffusing rain and snow enthalpy. The
    harmonic face mean in ``w^j`` is the series conductance of the adjacent
    cells, so ``w^j / \hat{\rho}^j_c \le 2 f_j K_h`` and the intensive tendency
    is a bounded three-point diffusion of ``\chi^j`` for every value of
    ``\hat{\rho}^j_c`` (see [PROPHET](prophet.md)).
    When `edmfx_vertical_diffusion` is `false`, `PrognosticEDMFX` reduces to the
    environment diffusive flux ``-D^c[-w^0 G^f[\chi^0]]``.

### Hyperdiffusion

Draft variables are hyperdiffused for numerical stability, along
terrain-following coordinate surfaces, using the decomposed and unweighted
subdomain fields, with the energy and moisture contributions split into
sensible-heat, geopotential, and water terms as in the grid mean. See
[Hyperdiffusion](hyperdiffusion.md) for the operator and the reference-state
subtraction.

## The implicit–explicit split

PROPHET terms are distributed between the two halves of the IMEX splitting (see
[Implicit Solver](implicit_solver.md)) like this.

**Post-ARK vertical transport and column physics** (`post_step_callback!`):

  - resolved vertical transport of ``\hat{\rho}^j``
    (`step_vertical_scalar_transport!`);
  - vertical advection of the draft velocity and intensive scalars, buoyancy,
    entrainment/detrainment, subgrid-scale mass fluxes onto the grid mean,
    surface mass-source relaxation, vertical diffusion, and prognostic TKE,
    subcycled at ``\Delta t_{\mathrm{sub}} \le 60\,\mathrm{s}``
    (`step_edmf_and_env_turbulence!` and `step_prognostic_tke!` inside
    `step_column_physics!`);
  - microphysics and grid-mean and draft sedimentation
    (`step_coupled_microphysics_sedimentation!`).

**Explicit ARK stage** (`remaining_tendency!`):

  - horizontal advection of all draft variables
    (`horizontal_dynamics_tendency!` for ``\hat{\rho}^j`` and ``h_s^j``,
    `horizontal_tracer_advection_tendency!` for ``q_t^j`` and the draft
    tracers);
  - the explicit part of the draft vertical momentum equation (vorticity terms
    and the horizontal/cross kinetic-energy gradient), in
    `explicit_vertical_advection_tendency!`;
  - hyperdiffusion, the Rayleigh sponge relaxation of the draft scalars toward
    the grid mean, and the optional horizontal diffusive fluxes.

## Split column integration

PROPHET column physics — vertical diffusion, updraft momentum and area,
entrainment, detrainment, relative two-stream mass-flux transport, and
prognostic TKE — runs per column in `step_column_physics!`
(`column_physics_step.jl`), after `step_vertical_scalar_transport!` advances
``\hat{\rho}^j`` with the same M-matrix factorization as the grid-mean scalars.
When ``\Delta t > 60\,\mathrm{s}``, `step_column_physics!` subcycles
``n_{\mathrm{sub}} = \lceil \Delta t / 60\,\mathrm{s} \rceil`` times with
``\Delta t_{\mathrm{sub}} = \Delta t / n_{\mathrm{sub}}``. Each substep runs:

 1. `step_edmf_and_env_turbulence!`: solves the draft vertical velocity,
    advances ``\hat{\rho}^j`` with entrainment, detrainment, and the surface
    area source (clamped to ``[0, \rho^j a_{\max}]``), and solves the relative
    two-stream system as successive tridiagonal solves: the intensive draft
    scalars first, then the conservative grid-mean transport and diffusion with
    the environment as the unknown.
 2. `step_prognostic_tke!`: updates ``\rho e_{\mathrm{tke}}`` with vertical
    diffusion, shear and buoyancy production, and dissipation.
 3. `_step_slab_ocean_turbulent!`, `repair_column_water_conservatively!`,
    `enforce_edmf_updraft_constraints!`, thermo/cloud-fraction refresh,
    `step_coupled_microphysics_sedimentation!`, and `_step_slab_ocean_precip!`.

See [Implicit Solver](implicit_solver.md#Post-ARK-Lie-split-column-physics) for
the full post-ARK step sequence.

## Jacobian

In the ARK wave-core stage, `ManualSparseJacobian` assigns `-I` diagonal blocks
to every `sgsʲs` and `ρtke` variable (`implicit_tendency!` leaves them zero),
so `BlockArrowheadSolve` eliminates only the resolved wave-core blocks. The
automatic-differentiation algorithms
([`AutoSparseJacobian`](@ref ClimaAtmos.AutoSparseJacobian),
[`AutoDenseJacobian`](@ref ClimaAtmos.AutoDenseJacobian)) remain available for
verification against finite differences (`*_sparse_autodiff` and
`*_dense_autodiff` configurations).

## Regularizations and state filters

Several devices keep the scheme well behaved at the edges of its validity. Each
introduces a small inconsistency so that the scheme does not break down there,
and each shows up in the code and in the diagnostics, so it helps to know which
is which.

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
kinks. Where the area fraction is small, the subdomains then sum to the grid
mean only approximately. ``a_{1/2}`` is the `a_half` field of
[`PrognosticEDMFX`](@ref ClimaAtmos.PrognosticEDMFX), which `get_turbconv_model`
sets to the minimum draft area `EDMF_min_area`.

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
grid mean is not. `edmfx_filter` is enabled in all PROPHET configurations in
`config/model_configs/` except the advection test.

### Other limiters

  - The entrainment and detrainment area limiters and the additive area-bounding
    rate, described in
    [Closures](prophet_closures.md#Entrainment-and-detrainment).
  - A floor of 1 m on the mixing length, which prevents division by zero in the
    TKE dissipation.
  - Negative TKE is relaxed to zero within one timestep rather than being
    clipped, in `step_edmf_and_env_turbulence!` (`column_physics_step.jl`).
  - The subgrid total-water standard deviation is bounded by
    ``\sigma_q \le r_{\max} q_t`` (`sgs_variance_max_rel_std`, default 0.5),
    with ``q_t`` the mean of the subdomain on which the quadrature is centred,
    before the quadrature, in `set_covariance_cache!`; see
    [Closures](prophet_closures.md#Variances-and-cloud-fraction).
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
    PROPHET explicit closures (the entrainment rates and the signed
    area-bounding rate, the surface mass-source terms; detrainment itself is
    assembled at each call site), then the coupled covariance/cloud-fraction
    Picard iteration, then the face diffusivities (which need the final cloud
    fraction), then the center mixing length, terminal velocities, and the
    microphysics tendency cache. Called before every explicit tendency evaluation.

The face diffusivities are computed *after* the cloud fraction because the
buoyancy gradient that enters ``N_e^2`` depends on it; the
mixing length is materialized inside the Picard iteration when the configuration
uses covariances, and separately otherwise, with `uses_covariances` as the
shared predicate that keeps the two paths from disagreeing.

## Where this is implemented

| Component                                                | Source                                                                                                                                                                                                                                                                           |
|:-------------------------------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implicit/explicit split                                  | [implicit/implicit_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/implicit_tendency.jl), [remaining_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/remaining_tendency.jl)               |
| Analytic stage solves for ``u_3^j`` and ``\hat{\rho}^j`` | [implicit/initialize_implicit_problem.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/initialize_implicit_problem.jl)                                                                                                                     |
| Horizontal advection of draft variables                  | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl)                                                                                                                                                                           |
| Vertical advection, buoyancy, sedimentation              | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl), [water_advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/water_advection.jl)                                                       |
| Subgrid-scale flux discretization                        | [edmfx_sgs_flux.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_sgs_flux.jl)                                                                                                                                                                 |
| Draft hyperdiffusion                                     | [hyperdiffusion.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/hyperdiffusion.jl)                                                                                                                                                                 |
| Jacobian blocks                                          | [implicit/manual_sparse_jacobian.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/manual_sparse_jacobian.jl)                                                                                                                               |
| State filters                                            | [mass_flux_closures.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/mass_flux_closures.jl), [constrain_state.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/constrain_state.jl)                                     |
| Environment reconstruction and the blend weight          | [utils/variable_manipulations.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/utils/variable_manipulations.jl)                                                                                                                                                          |
| Precomputed-quantity ordering                            | [cache/precomputed_quantities.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cache/precomputed_quantities.jl), [cache/prognostic_edmf_precomputed_quantities.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cache/prognostic_edmf_precomputed_quantities.jl) |
