# PROPHET: Overview and Equations

PROPHET (Prognostic Representation Of Physics for Eddy Transport) is
ClimaAtmos's representation of turbulence, convection, and the clouds they
produce. It is an extended eddy-diffusivity mass-flux (EDMF) scheme in the
lineage of
[Siebesma2007, Tan2018, Cohen2020, Lopez2020, Christopoulos2024](@cite),
generalized along the lines of the conditional-filtering framework of
[Thuburn2018, Thuburn2018b, Weller2019](@cite). The formulation is described in
full in the PROPHET paper (Azimi et al., in preparation); this page states the
formulation the code implements and maps it onto the source.

The scheme is still called `EDMFX` throughout the code: the configuration key
is `turbconv: "prognostic_edmfx"`, the types are
[`PrognosticEDMFX`](@ref ClimaAtmos.PrognosticEDMFX) and
[`EDOnlyEDMFX`](@ref ClimaAtmos.EDOnlyEDMFX), and the source files are
`src/prognostic_equations/edmfx_*.jl`. The rename to PROPHET has not been
carried into the code.

Three companion pages complete the description: [Closures](prophet_closures.md)
for the parameterized rates and diffusivities,
[Discretization and Time Stepping](prophet_numerics.md) for the semi-discrete
forms and the implicit solves, and
[Horizontal Diffusion](prophet_horizontal_diffusion.md) for the optional
horizontal component of the diffusive fluxes. The how-to guide
[Configuring and Tuning PROPHET](prophet_howto.md) covers the configuration
surface.

## What PROPHET adds to a dynamical core

The [governing equations](equations.md) carry subgrid-scale flux terms: the
momentum flux tensor ``\boldsymbol{\mathcal{T}}`` and the scalar fluxes
``\boldsymbol{\mathcal{F}}_h``, ``\boldsymbol{\mathcal{F}}_{q_t}``,
``\boldsymbol{\mathcal{F}}_{q_\mu^\sigma}``, and
``\boldsymbol{\mathcal{F}}_\chi``. They do not say how those terms are
computed. PROPHET closes them.

The distinguishing choice is that it does so with *prognostic* equations. A
conventional convection parameterization diagnoses a mass flux from the
instantaneous resolved state, which discards the memory of the convecting
population and presumes a separation between the timescale of the parameterized
process and the model timestep. PROPHET instead carries subgrid-scale state
forward in time: each grid cell holds a small set of subdomains with their own
density, vertical velocity, energy, water, and tracer content, advanced by
equations that mirror the resolved ones. Convective structures therefore have
memory, are advected horizontally by the resolved flow (so mesoscale
organization can develop), and conserve mass, water, and energy in the same
sense the resolved equations do.

The second consequence of the prognostic formulation is resolution adaptivity.
Nothing in the equations assumes that convection is unresolved. As the
horizontal grid is refined and the resolved flow captures more of the convective
variability, the subdomain-to-grid-mean contrasts shrink, the mass-flux part of
the subgrid-scale fluxes vanishes smoothly with them, and what remains is a
Deardorff-type 1.5-order large-eddy closure. There is no mode switch between the
two limits; see [The high-resolution limit](@ref) below.

## Subdomain decomposition

Each grid cell is partitioned into ``M + 1`` subdomains, indexed
``m = 0, \dots, M``. Subdomain ``m`` occupies a volume fraction ``a^m`` of the
cell, and

```math
\sum_{m=0}^{M} a^m = 1 .
```

Physically, the subdomains represent coherent structures (updrafts,
downdrafts) and the enveloping, more isotropically turbulent remainder;
formally, they are the result of a conditional averaging operation on the
subgrid flow [Thuburn2018](@cite). Subdomain ``m = 0`` is called the
*environment*. Unlike in earlier EDMF formulations, it is not structurally
distinct: it is simply the
subdomain whose prognostic equations are eliminated. The remaining subdomains
``m \ge 1``, collectively the *drafts*, are the ones the model integrates. The
number of drafts is set by `updraft_number` and is 1 by default; ``M = 1``
suffices for shallow convection and stratocumulus
[Lopez2020, Christopoulos2024](@cite).

Define the *effective density* of a subdomain as the product of its volume
fraction and its density,

```math
\hat{\rho}^m = a^m \rho^m ,
```

which is the model's prognostic mass variable for each draft (the field `ρa`).
Consistency between the subdomains and the grid mean is imposed by three
constraints: the effective densities sum to the grid-mean density,

```math
\sum_{m=0}^{M} \hat{\rho}^m = \rho ,
```

the effective-density-weighted sum of any specific scalar recovers the grid-mean
(Favre-averaged) scalar,

```math
\sum_{m=0}^{M} \hat{\rho}^m \psi^m = \rho \psi ,
```

and the subdomain mass fluxes sum to the grid-mean mass flux,

```math
\sum_{m=0}^{M} \hat{\rho}^m \boldsymbol{u}^m = \rho \boldsymbol{u} .
```

These constraints are what make the decomposition a decomposition rather than an
additional, independent model. They also imply that only ``M`` of the ``M+1``
sets of subgrid-scale equations are independent of the resolved equations, which
is why one set can be dropped.

!!! note "Why the environment is the residual"

    Eliminating the environment rather than the resolved equations, which is
    the choice made by the multifluid framework of
    [Thuburn2018, Weller2019](@cite), keeps the resolved equations untouched, so
    PROPHET slots into an existing dynamical core and stays accurate at
    resolutions where the subdomain decomposition is unnecessary. The drawback
    is that environment quantities have to be reconstructed as residuals, which
    is delicate when the drafts fill most of the cell; see
    [Environment as residual](@ref).

All subdomains share the resolved pressure ``p``, on the grounds that acoustic
adjustment is fast compared with the dynamics of interest
[Thuburn2018, Weller2019, Cohen2020](@cite). This removes the extra acoustic
modes the decomposition would otherwise introduce. The effect of subdomain
pressure differences is retained as a parameterized pressure (form) drag; see
[Pressure closure](prophet_closures.md#Pressure-closure). Subdomain densities
are *not* assumed equal outside the buoyancy terms, unlike in
[Cohen2020](@cite): an anelastic approximation within the subgrid is unnecessary
in a fully compressible model and complicates exact energy consistency.

Horizontal velocities are shared across subdomains, and only the vertical
velocity differs:

```math
\boldsymbol{u}^m = \boldsymbol{u}_h + \boldsymbol{w}^m .
```

This reduces the three-dimensional draft momentum equation to a single equation
for the vertical component, and eliminates vertical transport of horizontal
momentum by the drafts [Tan2018, Cohen2020](@cite). In generalized coordinates,
``\boldsymbol{u}_h = (u_1, u_2, 0)`` and ``\boldsymbol{w}^m = (0, 0, u_3^m)`` in
covariant components, so over sloping terrain the shared horizontal component
still contributes a nonzero contravariant vertical velocity
``u_h^3 = g^{31} u_1 + g^{32} u_2``.

## Draft equations

The drafts (``m \ge 1``) are advanced by the following equations. Continuity is
in flux form; everything else is in advective form, obtained by combining the
flux-form scalar equation with continuity. Advective form is used because the
effective densities ``\hat{\rho}^m`` can be very small, and dividing by them
inside a flux divergence is numerically fragile [Weller2019](@cite).

**Mass continuity:**

```math
\frac{\partial \hat{\rho}^m}{\partial t}
  + \nabla \cdot \left[ \hat{\rho}^m
      (\boldsymbol{u}^m - W_{q_t}^m \hat{\boldsymbol{k}}) \right]
  = \sum_{n \ne m} \hat{\rho}^m (E^{mn} - \Delta^{mn})
    + \hat{\rho}^m \hat{S}_{q_t}^m
```

**Momentum.** The shared horizontal velocity reduces this to a single equation
for ``u_3^m``:

```math
\frac{\partial \boldsymbol{u}^m}{\partial t}
  + (2 \boldsymbol{\Omega} + \boldsymbol{\omega}^m) \times \boldsymbol{u}^m
  = \boldsymbol{b}_{\mathrm{eff}}^m - \nabla \kappa^m + \boldsymbol{S}_u^m
    - \frac{1}{\rho} \nabla \cdot
      \left[ \rho (\boldsymbol{\mathcal{T}}^{\mathrm{diff}}
        + \boldsymbol{\mathcal{H}}_u) \right]
    + \sum_{n \ne m} E^{mn} (\boldsymbol{u}^n - \boldsymbol{u}^m)
```

**Moist static energy:**

```math
\frac{\partial h_s^m}{\partial t}
  + \boldsymbol{u}^m \cdot \nabla h_s^m
  = S_{e,\mathrm{eff}}^m
    - \boldsymbol{u}^m \cdot \left( - \frac{1}{\rho^m} \nabla p^\dagger
      + \boldsymbol{b}^m + \boldsymbol{S}_u^m \right)
    + Q_R^m
    - \frac{1}{\rho} \nabla \cdot
      \left[ \rho (\boldsymbol{\mathcal{F}}_h^{\mathrm{diff}}
        + \boldsymbol{\mathcal{H}}_h) \right]
    + \sum_{n \ne m} E^{mn} (h_s^n - h_s^m)
    + D_{h_s}^m
```

**Total water:**

```math
\frac{\partial q_t^m}{\partial t}
  + \boldsymbol{u}^m \cdot \nabla q_t^m
  = (1 - q_t^m) S_{q_t}^m
    - \frac{1}{\rho} \nabla \cdot
      \left[ \rho (\boldsymbol{\mathcal{F}}_{q_t}^{\mathrm{diff}}
        + \boldsymbol{\mathcal{H}}_{q_t}) \right]
    + \sum_{n \ne m} E^{mn} (q_t^n - q_t^m)
    + D_{q_t}^m
```

**Condensate and precipitation**, for ``\mu \in \{l, i\}`` and
``\sigma \in \{\cdot, \mathrm{cl}, \mathrm{pr}\}`` as the microphysics model
requires:

```math
\frac{\partial q_\mu^{\sigma m}}{\partial t}
  + \boldsymbol{u}^m \cdot \nabla q_\mu^{\sigma m}
  = S_{q_\mu^\sigma,\mathrm{eff}}^m
    - \frac{1}{\rho} \nabla \cdot
      \left[ \rho (\boldsymbol{\mathcal{F}}_{q_\mu^\sigma}^{\mathrm{diff}}
        + \boldsymbol{\mathcal{H}}_{q_\mu^\sigma}) \right]
    + \sum_{n \ne m} E^{mn} (q_\mu^{\sigma n} - q_\mu^{\sigma m})
    + D_{q_\mu^\sigma}^m
```

**Tracers:**

```math
\frac{\partial \chi^m}{\partial t}
  + \boldsymbol{u}^m \cdot \nabla \chi^m
  = S_{\chi,\mathrm{eff}}^m
    - \frac{1}{\rho} \nabla \cdot
      \left[ \rho (\boldsymbol{\mathcal{F}}_\chi^{\mathrm{diff}}
        + \boldsymbol{\mathcal{H}}_\chi) \right]
    + \sum_{n \ne m} E^{mn} (\chi^n - \chi^m)
    + D_\chi^m
```

**Equation of state:** the shared pressure and the subdomain temperature give
the subdomain density,

```math
\rho^m = \frac{p}{R_m^m T^m} .
```

Note the direction: in the resolved equations, the equation of state returns
pressure; here, it returns density, because pressure is shared.

Which water species a draft carries follows the grid-mean microphysics model.
In the equilibrium case, only ``q_t^m`` is prognostic and the condensate
partition comes from saturation adjustment; in the non-equilibrium cases, each
draft also carries cloud and precipitation species (and, for two-moment
microphysics, their number concentrations), each with its own sedimentation flux
and sources.

### Definitions and assumptions

**Perturbation pressure.** A hydrostatic reference pressure ``p_h`` satisfying
``\nabla p_h = -\rho_h \nabla \Phi`` is subtracted to define
``p^\dagger = p - p_h``. The reference is the same one the dynamical core uses
for accurate evaluation of the pressure-gradient force; the choice affects
numerical accuracy, not the dynamics.

**Buoyancy.** The buoyancy of a subdomain is defined relative to a reference
density,

```math
\boldsymbol{b}^m = - \frac{\rho^m - \rho_{\mathrm{ref}}}{\rho^m} \nabla \Phi ,
```

and the *effective* buoyancy driving the subdomain momentum adds the
nonhydrostatic pressure gradient and subtracts the pressure drag
``\boldsymbol{d}^m``,

```math
\boldsymbol{b}_{\mathrm{eff}}^m
  = - \frac{1}{\rho^m} \nabla p^\dagger + \boldsymbol{b}^m
    - \boldsymbol{d}^m .
```

Because the effective densities sum to ``\rho``, the effective-density-weighted
sum of the buoyancies is a grid-mean buoyancy,

```math
\sum_{m=0}^{M} \hat{\rho}^m \boldsymbol{b}^m
  = -(\rho - \rho_{\mathrm{ref}}) \nabla \Phi \equiv \rho \boldsymbol{b} ,
```

so the subdomains exert no net force on the grid mean. Newton's third law
imposes the same requirement on the drag,
``\sum_m \hat{\rho}^m \boldsymbol{d}^m = 0``, which is why the pressure drag
does not appear in the resolved momentum equation.

!!! note "Reference density in the code"

    The formulation takes ``\rho_{\mathrm{ref}} = \rho_h``, the hydrostatic
    reference density. The code uses the grid-mean density,
    ``\rho_{\mathrm{ref}} = \rho``, computing the normalized density excess
    `ᶜρ_diffʲs = (ρʲ - ρ)/ρʲ` once and reusing it in the momentum and energy
    tendencies and in the implicit solve. The two agree wherever the grid mean
    is close to hydrostatic, and differ by the grid-mean hydrostatic imbalance
    at gray-zone resolutions. Reconciling them is tracked as a code-side task.

**Effective sources.** For a scalar ``\psi \ne q_t``, the source that appears in
the advective-form equation carries the dilution effect of the mass source,

```math
S_{\psi,\mathrm{eff}}^m = S_\psi^m - \psi^m S_{q_t}^m ,
```

and the total-water equation uses the equivalent form
``(1 - q_t^m) S_{q_t}^m``. The effective mass source in continuity is

```math
\hat{S}_{q_t}^m = S_{q_t}^m
  - \frac{1}{\rho} \nabla \cdot
    \left[ \rho (\boldsymbol{\mathcal{F}}_{q_t}^{\mathrm{diff}}
      + \boldsymbol{\mathcal{H}}_{q_t}) \right] ,
```

the subdomain counterpart of the grid-mean ``\hat{S}_{q_t}``: subgrid-scale
moisture transport moves moist air mass, so it appears in the mass budget as
well as the water budget.

**Sedimentation-divergence term.** The terms ``D_\psi^m`` arise from converting
the flux-form scalar equations to advective form in the presence of
sedimentation,

```math
D_\psi^m = \frac{1}{\hat{\rho}^m} \left[
    \nabla \cdot (\hat{\rho}^m W_\psi^m \hat{\boldsymbol{k}})
    - \psi^m \nabla \cdot (\hat{\rho}^m W_{q_t}^m \hat{\boldsymbol{k}})
  \right] ,
```

with ``W_\psi^m`` the mass-weighted sedimentation flux of ``\psi`` in subdomain
``m``. The physical content beyond ordinary within-draft fallout is geometric:
where a draft's volume fraction varies with height, its boundary is not
vertical, and sedimenting condensate crosses it. A draft that narrows downward
loses falling condensate to its surroundings; one that widens downward gains it.
This transfer acts only on condensate species and carries pure condensate rather
than a parcel at the source subdomain's composition, which is what makes it
structurally distinct from the dynamical entrainment and detrainment below. The
code implements it two-directionally, for a single draft exchanging with the
environment, in `updraft_sedimentation!`; the second (``\psi^m``-weighted) term
of ``D_\psi^m`` is neglected as small compared with the leading-order dynamics.

**Diffusive fluxes are grid-mean.** The diffusive and hyperdiffusive flux
divergences in the draft equations are the *grid-mean* divergences, applied
uniformly to every subdomain, not per-subdomain divergences. This is the
conservative-exchange consistency condition of [Thuburn2022](@cite): taking the
divergence of ``\hat{\rho}^m``-weighted fluxes inside a material tendency would
make gradients of the volume fraction act as spurious sources, so that two
materially identical drafts with different area gradients would evolve
differently. With the grid-mean form, the ``\hat{\rho}^m`` factors cancel
algebraically and the subdomain sum collapses to the grid-mean flux divergence.
Consequently, when `edmfx_vertical_diffusion` is enabled, each draft receives
the same *specific* diffusive tendency as the grid mean.

**Energy.** The prognostic thermodynamic variable of a draft is the specific
moist static energy

```math
h_s^m = c_{pm}(q^m)(T^m - T_0) + (q_t^m - q_c^m) L_{v,0} - q_i^m L_{f,0} + \Phi ,
```

not the total energy the resolved equations carry. The energy equation neglects
the explicit ``\partial p^\dagger / \partial t`` but retains the work term
``-\boldsymbol{u}^m \cdot (-(\rho^m)^{-1} \nabla p^\dagger + \boldsymbol{b}^m + \boldsymbol{S}_u^m)``,
the reversible conversion between kinetic and moist static energy.
[Peters2021](@cite) showed that keeping the perturbation-pressure part of this
work matters for updraft temperature and buoyancy in deep convection. The
pressure drag is deliberately *excluded* from the work term: form drag moves
macroscopic kinetic energy into turbulence in the surroundings rather than
changing the parcel's thermodynamic state, so energy consistency requires it to
reappear as a return-to-isotropy source in the turbulence kinetic energy budget.

The subgrid-scale turbulent and hyperdiffusive fluxes of moist static energy are
approximated by the grid-mean fluxes of specific total enthalpy,
``\boldsymbol{\mathcal{F}}_{h_s}^{\mathrm{diff}} \approx \boldsymbol{\mathcal{F}}_h^{\mathrm{diff}}``
and ``\boldsymbol{\mathcal{H}}_{h_s} \approx \boldsymbol{\mathcal{H}}_h``. The
two differ by the kinetic energy, whose gradients are smaller than those of
thermal and potential energy by two orders of magnitude or more, so the
diffusive transport of kinetic energy is negligible. This lets the draft
equations reuse the grid-mean enthalpy fluxes unchanged, including their
decomposition into a dry-static-energy term and a water-enthalpy term, which
[Yatunin2026](@cite) adopt to avoid a spurious enthalpy flux carried by dry-air
diffusion. See [Governing Equations](equations.md) and
[Hyperdiffusion](hyperdiffusion.md).

!!! note "TODO: not yet implemented"

    Two pieces of the formulation above are not in the code:

      - the perturbation-pressure work
        ``\boldsymbol{u}^m \cdot (\rho^m)^{-1} \nabla p^\dagger`` in the energy
        equation. `pressure_work.jl` is an explicit, documented no-op dispatch
        point that reserves the place for it;
      - the return-to-isotropy source that receives the kinetic energy the
        pressure drag removes. The drag itself is applied (in the implicit
        vertical-velocity solve), but the energy it extracts is currently lost
        rather than transferred to the turbulence kinetic energy budget.

## Coupling to the resolved equations

The subgrid-scale fluxes in the resolved equations are fixed, not chosen: they
are whatever makes the sum of the ``M+1`` subdomain equations reproduce the
resolved equations. Carrying out that sum splits each flux into two parts. For a
generic resolved scalar ``\psi``,

```math
\rho \boldsymbol{\mathcal{F}}_\psi
  = \underbrace{\sum_{m=0}^{M} \hat{\rho}^m
      (\boldsymbol{w}^m - \boldsymbol{w})(\psi^m - \psi)}_{\text{mass flux}}
  + \underbrace{\rho \boldsymbol{\mathcal{F}}_\psi^{\mathrm{diff}}}_{\text{diffusive}} .
```

The first term is the coherent transport by the inter-subdomain circulation, the
*mass-flux* contribution. Only vertical-velocity differences appear, because the
horizontal velocity is shared. The second is the transport by intra-subdomain
isotropic turbulence, closed by the grid-mean eddy diffusivity of
[Closures](prophet_closures.md).

The consistency constraints make three algebraically equivalent forms of the
mass-flux term available,

```math
\sum_m \hat{\rho}^m (\boldsymbol{w}^m - \boldsymbol{w})(\psi^m - \psi)
= \sum_m \hat{\rho}^m \boldsymbol{w}^m (\psi^m - \psi)
= \sum_m \hat{\rho}^m (\boldsymbol{w}^m - \boldsymbol{w}) \psi^m .
```

The code uses the symmetric first form, in which the flux vanishes identically
when the subdomains are materially identical, regardless of volume-fraction
gradients. That is the discrete counterpart of the property
[Thuburn2022](@cite) require of the diffusive terms.

The specific fluxes follow. For momentum, the mass-flux part is a tensor
product, and because horizontal velocities are shared, it populates only the
vertical–vertical entry:

```math
\rho \boldsymbol{\mathcal{T}}
  = \sum_{m=0}^{M} \hat{\rho}^m (\boldsymbol{w}^m - \boldsymbol{w})
      \otimes (\boldsymbol{w}^m - \boldsymbol{w})
  + \rho \boldsymbol{\mathcal{T}}^{\mathrm{diff}} .
```

For energy, the transported quantity is the specific total enthalpy, since that
is what the resolved advective flux carries:

```math
\rho \boldsymbol{\mathcal{F}}_h
  = \sum_{m=0}^{M} \hat{\rho}^m (\boldsymbol{w}^m - \boldsymbol{w})
      (h_{\mathrm{tot}}^m - h_{\mathrm{tot}})
  + \rho \boldsymbol{\mathcal{F}}_h^{\mathrm{diff}} ,
  \qquad h_{\mathrm{tot}}^m = h_s^m + \kappa^m + \kappa_{\mathrm{iso}} .
```

The mass-flux part carries ``h_{\mathrm{tot}}^m`` directly rather than the
dry-static-energy-plus-water-enthalpy decomposition used for the diffusive part:
it represents bodily motion of whole moist parcels, each at its own composition,
so the dry-air-diffusion artifact that motivates the decomposition does not
arise. Adding the grid-mean ``\kappa_{\mathrm{iso}}`` uniformly to every
subdomain keeps
``\sum_m \hat{\rho}^m h_{\mathrm{tot}}^m = \rho h_{\mathrm{tot}}``; it cancels
from the deviation that drives the flux. Water species and tracers take the
generic form above.

!!! note "TODO: not yet implemented"

    The vertical mass-flux contribution to the momentum flux, the
    ``(\boldsymbol{w}^m - \boldsymbol{w}) \otimes (\boldsymbol{w}^m - \boldsymbol{w})``
    term in ``\boldsymbol{\mathcal{T}}`` above, is a stub in
    `edmfx_sgs_flux.jl`. The mass-flux transport of energy, water, and tracers
    is applied; the momentum part is not. It is leading-order once the drafts
    are partially resolved.

The scalar mass fluxes are applied by `edmfx_sgs_mass_flux_tendency!` and the
diffusive fluxes by `edmfx_sgs_diffusive_flux_tendency!`, both in
`edmfx_sgs_flux.jl`, gated on the configuration flags
`edmfx_sgs_mass_flux` and `edmfx_sgs_diffusive_flux` respectively. Because the
total-water flux moves moist air mass, both tendencies also increment the
resolved density, consistent with ``\hat{S}_{q_t}`` in the continuity equation.

### Environment as residual

With the environment equations eliminated, the environment quantities that the
flux sums above require are reconstructed from the resolved state and the draft
variables:

```math
\hat{\rho}^0 = \rho - \sum_{m \ge 1} \hat{\rho}^m ,
\qquad
\psi^0 = \frac{\rho \psi - \sum_{m \ge 1} \hat{\rho}^m \psi^m}{\hat{\rho}^0} ,
\qquad
\hat{\rho}^0 \boldsymbol{w}^0
  = \rho \boldsymbol{w} - \sum_{m \ge 1} \hat{\rho}^m \boldsymbol{w}^m .
```

The reconstruction of a specific environment scalar divides by ``\hat{\rho}^0``,
which is small wherever the drafts nearly fill the cell. `specific` and the
`ᶜspecific_env_*` helpers in `src/utils/variable_manipulations.jl` therefore
blend the residual toward the grid-mean value below an area threshold `a_half`;
see
[Environment reconstruction](prophet_numerics.md#Environment-reconstruction).

### The high-resolution limit

The per-subdomain mass-flux term is a product of three factors,
``a^m``, ``\rho^m (\boldsymbol{w}^m - \boldsymbol{w})``, and
``\psi^m - \psi``, each of which responds to refinement. The two contrasts are
themselves measures of the subgrid variance among subdomains, and vanish once
the resolved flow captures the convective structures. The draft volume
fractions ``a^m`` can adapt downward as well, through the
entrainment/detrainment closures and the surface closure that seeds the drafts,
since at fine enough resolution the most buoyant fraction of surface-layer
turbulence is resolved rather than represented by a draft. A product vanishes if
any factor does, so vanishing contrasts alone are sufficient.

What survives is the diffusive flux,

```math
\rho \boldsymbol{\mathcal{F}}_\psi^{\mathrm{diff}} = -\rho K_\psi \nabla \psi ,
\qquad K_\psi = \frac{K_m}{\mathrm{Pr}_t(\mathrm{Ri})} ,
\qquad K_m = c_m \, l \sqrt{\kappa_{\mathrm{iso}}} ,
```

together with the deviatoric momentum flux and the prognostic equation for
``\kappa_{\mathrm{iso}}``. That is a Deardorff-type 1.5-order large-eddy closure
[Deardorff1980, Wyngaard2010](@cite): a prognostic subgrid kinetic energy
supplies the velocity scale, the mixing length supplies the length scale, and
the trace of the subgrid Reynolds stress is closed by
``\kappa_{\mathrm{iso}}`` itself. The mixing length is grid-aware through the
resolvability filter scale ``l \le \max(\Delta x_h, \Delta z)``, which is inert
at global-model horizontal resolutions and reduces to the Deardorff grid-scale
bound on isotropic grids.

For this smooth transition to hold *discretely*, the advection schemes applied
to the subdomain variables must not introduce numerical diffusion that mimics a
residual subgrid flux; otherwise the scheme approaches the large-eddy limit with
a resolution-dependent effective diffusivity. This is one reason the upwinding
options for the subdomain variables are configurable; see
[Discretization and Time Stepping](prophet_numerics.md).

## Symbols

| Symbol                                               | Meaning                                                                           | Units          |
|:---------------------------------------------------- |:--------------------------------------------------------------------------------- |:-------------- |
| ``a^m``                                              | Volume fraction of subdomain ``m``                                                |                |
| ``\rho^m``, ``\hat{\rho}^m``                         | Subdomain density and effective density ``a^m \rho^m``                            | kg m⁻³         |
| ``\boldsymbol{u}^m``, ``\boldsymbol{w}^m``           | Subdomain velocity and its vertical part                                          | m s⁻¹          |
| ``h_s^m``                                            | Subdomain specific moist static energy                                            | J kg⁻¹         |
| ``q_t^m``, ``q_\mu^{\sigma m}``, ``\chi^m``          | Subdomain total water, condensate/precipitation, tracer                           | kg kg⁻¹        |
| ``T^m``, ``R_m^m``                                   | Subdomain temperature and gas constant of moist air                               | K, J kg⁻¹ K⁻¹  |
| ``\boldsymbol{b}^m``, ``b^m``                        | Subdomain buoyancy and its vertical component                                     | m s⁻²          |
| ``\boldsymbol{b}_{\mathrm{eff}}^m``                  | Effective buoyancy driving the subdomain momentum                                 | m s⁻²          |
| ``\boldsymbol{d}^m``                                 | Pressure (form) drag deceleration                                                 | m s⁻²          |
| ``p_h``, ``p^\dagger``, ``\rho_h``                   | Hydrostatic reference pressure, perturbation, reference density                   | Pa, Pa, kg m⁻³ |
| ``\kappa^m``, ``\boldsymbol{\omega}^m``              | Subdomain specific kinetic energy and relative vorticity                          | J kg⁻¹, s⁻¹    |
| ``E^{mn}``, ``\Delta^{mn}``                          | Dynamical entrainment (``n \to m``) and detrainment (``m \to n``)                 | s⁻¹            |
| ``S_\psi^m``, ``S_{\psi,\mathrm{eff}}^m``            | Subdomain source of ``\psi``, and its dilution-corrected form                     | s⁻¹, varies    |
| ``\hat{S}_{q_t}^m``                                  | Effective subdomain mass source                                                   | s⁻¹            |
| ``W_\psi^m``, ``D_\psi^m``                           | Sedimentation flux of ``\psi`` and sedimentation-divergence term                  | varies         |
| ``Q_R^m``                                            | Subdomain radiative heating rate                                                  | W kg⁻¹         |
| ``\kappa_{\mathrm{iso}}``, ``\kappa_{\mathrm{coh}}`` | Isotropic (intra-subdomain) and coherent (inter-subdomain) subgrid kinetic energy | J kg⁻¹         |
| ``K_m``, ``K_\psi``                                  | Eddy viscosity and eddy diffusivity                                               | m² s⁻¹         |
| ``l``                                                | Mixing and dissipation length scale                                               | m              |

The remaining closure symbols are defined on the [Closures](prophet_closures.md)
page. [Notation and Symbols](notation.md) maps these onto the field names in the
code.

## Radiation

Radiation is computed once per column, not per subdomain. The formulation
partitions the heating rate at the heating-rate level: a clear-sky and an
overcast calculation give ``Q_R^{\mathrm{clr}}`` and ``Q_R^{\mathrm{cld}}``, and
each subdomain receives the linear combination weighted by *its own* cloud
fraction,

```math
Q_R^m = (1 - f_c^m) Q_R^{\mathrm{clr}} + f_c^m Q_R^{\mathrm{cld}} ,
```

which sums to the grid-mean heating rate because
``\sum_m a^m f_c^m = f_c^{\mathrm{tot}}``. A clear subdomain then sees no
cloud-top longwave cooling and a fully cloudy one sees all of it.

!!! note "TODO: not yet implemented"

    The code applies the *grid-mean* heating rate to every draft
    (`radiation_tendency!` in `radiation.jl`), which is a reasonable
    approximation mainly because drafts are usually absent in the stratosphere,
    where radiative heating is largest. The subdomain partition above is not yet
    in place. Of the two profiles it needs, the clear-sky one is available in
    the `AllSkyRadiationWithClearSkyDiagnostics` mode
    (`rad: allskywithclear`), which exposes RRTMGP's `clear_*` fluxes; the
    overcast profile is not computed by any mode.

## Where this is implemented

| Component                                  | Source                                                                                                                                                                                                                     |
|:------------------------------------------ |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model types and configuration dispatch     | [types.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/types.jl), [config/model_getters.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/config/model_getters.jl)                                         |
| Draft state variables                      | [setups/common/prognostic_variables.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/setups/common/prognostic_variables.jl)                                                                                        |
| Draft and environment diagnosed quantities | [cache/prognostic_edmf_precomputed_quantities.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/cache/prognostic_edmf_precomputed_quantities.jl)                                                                    |
| Draft advection, buoyancy, sedimentation   | [advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/advection.jl), [water_advection.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/water_advection.jl) |
| Entrainment and detrainment tendencies     | [edmfx_entr_detr.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_entr_detr.jl)                                                                                                         |
| Subgrid-scale fluxes onto the grid mean    | [edmfx_sgs_flux.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_sgs_flux.jl)                                                                                                           |
| Buoyancy, drag, and area helpers           | [mass_flux_closures.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/mass_flux_closures.jl)                                                                                                   |
| Environment reconstruction from residuals  | [utils/variable_manipulations.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/utils/variable_manipulations.jl)                                                                                                    |
| Diagnostics of subdomain quantities        | [diagnostics/edmfx_diagnostics.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/diagnostics/edmfx_diagnostics.jl)                                                                                                  |
