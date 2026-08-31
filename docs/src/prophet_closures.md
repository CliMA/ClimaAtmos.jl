# PROPHET: Closures

The [PROPHET equations](prophet.md) leave several terms unspecified: the rates
at which subdomains exchange fluid, the pressure drag between them, the eddy
diffusivity that closes the intra-subdomain fluxes, the subgrid variances that
the cloud and microphysics closures integrate over, and the surface boundary
conditions. This page states the closures the code uses for each, and names the
parameters that control them.

These closures are the parts of the scheme most likely to change. They are
deliberately separated from the equations so that they can be replaced
(physics-based, calibrated, or machine-learned) without touching the
conservation structure. Where the implementation differs from the formulation in
the PROPHET paper (Azimi et al., in preparation), that is stated.

## Entrainment and detrainment

Entrainment and detrainment parameterize the exchange of mass and of every
property the exchanged fluid carries. PROPHET treats them as *relaxation rates*
``E^{mn}`` and ``\Delta^{mn}`` with units of inverse time, rather than as the
fractional rates per unit length ``\epsilon^{mn} = E^{mn}/w^m`` conventional in
mass-flux schemes. Rates carry no coordinate dependence through an explicit
velocity. They also make plain that entrainment and detrainment relax subdomain
properties toward each other, on timescales ``1/E^{mn}`` and
``1/\Delta^{mn}``, which is what makes their numerical treatment and the
positivity of ``\hat{\rho}^m`` tractable.

Mass and scalar conservation require the symmetry

```math
\hat{\rho}^m E^{mn} = \hat{\rho}^n \Delta^{nm} ,
```

so that what subdomain ``m`` gains from ``n`` is what ``n`` loses to ``m``, and
the exchange terms cancel in the sum over subdomains that recovers the resolved
equations.

In the code, the total entrainment rate of draft ``j`` is assembled from three
pieces (`compute_entrainment`):

```math
E^j = \varepsilon_w \, |w^j| + \varepsilon_r + \max(0, \Lambda_a) ,
```

a velocity-proportional part with an inverse-length prefactor
``\varepsilon_w``, a velocity-independent rate ``\varepsilon_r``, and the
positive branch of a signed area-bounding rate ``\Lambda_a`` (below). Two models
supply ``\varepsilon_w`` and ``\varepsilon_r``, selected by
`edmfx_entr_model`:

  - **`"Generalized"`** ([`InvZEntrainment`](@ref ClimaAtmos.InvZEntrainment),
    the default). The velocity-proportional prefactor is a fixed inverse length
    plus an inverse
    height above the surface, and the velocity-independent rate combines a
    buoyancy timescale with a constant relaxation rate:

    ```math
    \varepsilon_w = \Lambda_{\max} \left( \frac{1}{L_\varepsilon}
      + \frac{c_\varepsilon}{z - z_s} \right) ,
    \qquad
    \varepsilon_r = \Lambda_{\max} \left( c_{\varepsilon b} \,
      \tau_{\mathrm{buoy}}^{-1} + \tau_\varepsilon^{-1} \right) ,
    ```

    with ``\tau_{\mathrm{buoy}}^{-1} = \min(\tau_{\max}^{-1}, (\Delta b^j)_+ / |\Delta w^j|)``
    the clipped inverse buoyancy timescale built from the draft–environment
    buoyancy and velocity contrasts, and ``\Lambda_{\max}`` the upper-area
    limiter.

  - **`"PiGroups"`** ([`PiGroupsEntrainment`](@ref
    ClimaAtmos.PiGroupsEntrainment)). The prefactor is a linear function of five
    nondimensional groups formed from the local subdomain state, in the manner
    of [Cohen2020, Christopoulos2024](@cite):

    ```math
    \varepsilon_w = \frac{\Lambda_{\max}}{z - z_s}
      \max\!\left(0, \; \sum_{i=1}^{5} c_i |\Pi_i| + c_6 \right) ,
    ```

    with
    ``\Pi_1 \propto z \, \Delta b^j / (\Delta w^j)^2``,
    ``\Pi_2 \propto \kappa_{\mathrm{iso}} / (\Delta w^j)^2``,
    ``\Pi_3 = \sqrt{a^j}``,
    ``\Pi_4 = \mathrm{RH}^j - \mathrm{RH}^0``, and
    ``\Pi_5 = z / H`` with ``H`` the pressure scale height. ``\Pi_1`` and
    ``\Pi_2`` are rescaled to ``O(1)`` and clipped to ``[-1, 1]``. The
    coefficients are `entr_param_vec` and are the natural target of calibration
    or of a learned closure.

A separate *turbulent* entrainment rate, which does not scale with the draft
velocity and decays as the draft fills the cell,

```math
E_{\mathrm{turb}}^j = \varepsilon_{t} \exp(-c_t a^j) ,
```

is added to the dynamical rate in the scalar tendencies
(`turbulent_entrainment`; ``(\varepsilon_t, c_t)`` are the two entries of
`turb_entr_param_vec`).

Detrainment (`edmfx_detr_model: "Generalized"`,
[`BuoyancyVelocityDetrainment`](@ref ClimaAtmos.BuoyancyVelocityDetrainment))
combines a negative-buoyancy timescale with a converging mass flux:

```math
\Delta^j = \max\!\left(0, \; \Lambda_{\min} \left[
    c_\delta \, \tau_{\mathrm{buoy}}^{-1}
    - c_{\delta \nabla} \frac{\min(0, \nabla \cdot (\hat{\rho}^j \boldsymbol{w}^j))}{\hat{\rho}^j}
  \right] \right) + \max(0, -\Lambda_a) ,
```

with ``\tau_{\mathrm{buoy}}^{-1}`` now built from the *negative* part of the
buoyancy contrast, and ``\Lambda_{\min}`` the lower-area limiter. Only a
converging (negative) mass-flux divergence detrains.

**Area limiters.** Two mechanisms keep the draft area fraction inside
``[a_{\min}, a_{\max}]``. Multiplicative limiters damp the rate that would push
the area toward the bound it is approaching,

```math
\Lambda_{\max} = \left[ \left(1 - \frac{a}{a_{\max}}\right)_+ \right]^{p_{\max}}
\quad \text{on entrainment,} \qquad
\Lambda_{\min} = \left[ \left(1 - \frac{a_{\min}}{a}\right)_+ \right]^{p_{\min}}
\quad \text{on detrainment,}
```

and a signed additive rate ``\Lambda_a`` (`area_bounding_entr_detr`) relaxes the
area back if it leaves the interval at all:

```math
\Lambda_a = s_{\min} \left[ \frac{(a_{\min} - a)_+}{a_{\min}} \right]^{p_{\min}}
  - s_{\max} \left[ \frac{(a - a_{\max})_+}{1 - a_{\max}} \right]^{p_{\max}} .
```

The two ranges are disjoint, so at most one term is nonzero. The small positive
``a_{\min}`` keeps the equations from becoming singular and maintains a seed
draft across convective gaps; ``a_{\max}`` keeps the environment from being
displaced entirely.

!!! note "Departure from the formulation"

    In the paper the pairwise mass flux
    ``\hat{\rho}^m E^{mn} = \hat{\rho}^n \Delta^{nm}`` carries a *single* shared
    limiter, so the symmetry relation holds by construction. The code instead
    treats entrainment and detrainment as unrelated closures with independent
    one-sided limiters, and evaluates the entrainment relaxation against the
    environment value with ``w^0`` set to zero at the call site rather than
    using the true velocity difference. That substitution is deliberate: the
    true difference spuriously grows trivial drafts where ``w^j \approx 0``. A
    symmetric redesign has to address that failure mode, and is tracked as a
    code-side task.

Detrainment does not appear in the explicit scalar tendencies. It is absorbed
into the analytic implicit solve for ``\hat{\rho}^j``, and scalars are detrained
implicitly through the area divergence of the mass flux; see
[Discretization and Time Stepping](prophet_numerics.md).

## Pressure closure

The subgrid pressure perturbation exerts two forces on a draft: a virtual-mass
effect that reduces its effective buoyancy, because accelerating displaces the
surrounding fluid [Simpson1969, Bretherton2004](@cite), and an aerodynamic form
drag [Romps2015, Morrison2016](@cite). The formulation writes

```math
\boldsymbol{d}^m = C_v (\boldsymbol{b}^m - \boldsymbol{b})
  + C_d \sum_{n \ne m} \frac{\hat{\rho}^n}{\rho}
    \frac{\|\boldsymbol{u}^m - \boldsymbol{u}^n\|}{r^{mn}}
    (\boldsymbol{u}^m - \boldsymbol{u}^n) ,
```

with ``r^{mn}`` the harmonic mean of subdomain length scales. Both terms satisfy
the no-net-drag constraint ``\sum_m \hat{\rho}^m \boldsymbol{d}^m = 0``: the
virtual-mass term because the buoyancy anomalies sum to zero, the form-drag term
pairwise by symmetry. The length scale shrinks with the area fraction
(``r^m \propto \sqrt{a^m}``), which is necessary for well-posedness: if a
draft's mass vanishes while its drag length stays finite, its relative velocity
diverges [Weller2019](@cite).

The code implements this in the single-draft, environment-only setting, inside
the implicit vertical-velocity solve. The virtual-mass effect appears as a
buoyancy reduction factor ``(1 - \alpha_b)`` with
``\alpha_b`` = `pressure_normalmode_buoy_coeff1`, applied unconditionally. The
form drag,

```math
d^j = \frac{\alpha_d}{2 H}
  \left( \frac{1}{\sqrt{\max(a^j, a_{\min})}}
       + \frac{1}{\sqrt{\max(a^0, a_{\min})}} \right)
  (w^j - w^0) |w^j - w^0| ,
```

with ``\alpha_d`` = `pressure_normalmode_drag_coeff` and
``H = R_d T_{s,\mathrm{ref}} / g`` a fixed reference scale height, is gated on
`edmfx_nh_pressure` and contributes a quadratic sink to the stage equation for
``w^j``. This is the pairwise form above for the draft–environment pair with
``r^m = H \sqrt{a^m}``, so the harmonic mean gives the sum of ``1/\sqrt{a}``
terms and ``r^{mn} \to 0`` as either area vanishes. The
``\hat{\rho}^n / \rho`` weighting is not included; the environment velocity is
instead eliminated in favor of ``w^j`` in the implicit solve, which brings in
factors of ``\rho / \hat{\rho}^0`` (see
[Discretization and Time Stepping](prophet_numerics.md)).

## Turbulence kinetic energy and eddy diffusivity

The grid-mean subgrid kinetic energy splits into an isotropic
intra-subdomain part and a coherent inter-subdomain part,

```math
\kappa_{\mathrm{SGS}} = \kappa_{\mathrm{iso}} + \kappa_{\mathrm{coh}} ,
\qquad
\kappa_{\mathrm{coh}} = \frac{1}{2\rho}
  \sum_{m} \hat{\rho}^m \|\boldsymbol{u}^m - \boldsymbol{u}\|^2 .
```

Only ``\kappa_{\mathrm{iso}}`` does the mixing, so the diffusivity and
dissipation closures are functions of it alone; ``\kappa_{\mathrm{coh}}`` is
diagnosable from the subdomain velocities the model already carries. It is
``\kappa_{\mathrm{iso}}`` that the model prognoses, as `Y.c.ρtke` (enabled by
`prognostic_tke`).

Because the diffusive fluxes are grid-mean quantities applied uniformly across
subdomains, a single grid-mean ``\kappa_{\mathrm{iso}}`` suffices. That is also
a considerable simplification: the inter-subdomain transfers of isotropic TKE by
entrainment, detrainment, and return-to-isotropy drag redistribute energy within
the cell and sum to zero over it [Tan2018, Lopez2020](@cite), so none of them
has to be computed.

The budget is the compressible TKE equation [Pope2000, Wyngaard2010](@cite):

```math
\frac{\partial (\rho \kappa_{\mathrm{iso}})}{\partial t}
  + \nabla \cdot (\rho \kappa_{\mathrm{iso}} \boldsymbol{u})
= \underbrace{\nabla \cdot (\rho K_m \nabla \kappa_{\mathrm{iso}})}_{\text{transport}}
+ \underbrace{2 \rho K_m \|\boldsymbol{\mathcal{E}}_D\|_F^2}_{\text{shear production}}
- \underbrace{\tfrac{2}{3} \rho \kappa_{\mathrm{iso}} \nabla \cdot \boldsymbol{u}}_{\text{dilatation}}
- \underbrace{\rho K_h \frac{\partial b}{\partial z}}_{\text{buoyancy production}}
+ \underbrace{\rho R_{\mathrm{coh} \to \mathrm{iso}}}_{\text{drag return-to-isotropy}}
- \underbrace{\rho D_\kappa}_{\text{dissipation}} ,
```

with ``\boldsymbol{\mathcal{E}}_D`` the trace-free deviatoric strain rate of the
resolved velocity. Writing the shear production as the Frobenius norm of
``\boldsymbol{\mathcal{E}}_D`` makes it manifestly non-negative, as the
irreversible cascade must be, and separates it from the dilatation term, which
is reversible work done by macroscopic compression and expansion. The dilatation
term is usually dropped under anelastic or Boussinesq assumptions; it belongs
here because the dynamical core is fully compressible, and keeping it is what
makes the conversion between resolved kinetic energy, internal energy, and
subgrid TKE consistent.

Only the *diffusive* part of the buoyancy flux is a source of
``\kappa_{\mathrm{iso}}``. The full Favre-averaged buoyancy flux is

```math
\rho \overline{w' b'}^*
  = \underbrace{\sum_m \hat{\rho}^m (w^m - w) b^m}_{\text{coherent}}
    \underbrace{- \rho K_h \frac{\partial b}{\partial z}}_{\text{diffusive}} ,
```

and the coherent part drives the inter-subdomain circulation through the
buoyancy term of the draft momentum equations, so it is already accounted for in
``\kappa_{\mathrm{coh}}``. Including it here would double-count buoyancy
production and spuriously inflate the diffusivity near cloud tops with active
drafts.

Dissipation uses Taylor's surrogate,

```math
D_\kappa = c_d \frac{\kappa_{\mathrm{iso}}^{3/2}}{l} ,
```

and the eddy viscosity and diffusivities are

```math
K_m = c_m \, l \sqrt{\kappa_{\mathrm{iso}}} ,
\qquad
K_\psi = \frac{K_m}{\mathrm{Pr}_t(\mathrm{Ri})} .
```

The turbulent Prandtl number follows [Li2019](@cite) (with an algebraic error in
the published expression corrected), as the positive root

```math
\mathrm{Pr}_t = \frac{X + \sqrt{\max(X^2 - 4 \mathrm{Pr}_n \mathrm{Ri}, 0)}}{2},
\qquad X = \mathrm{Pr}_n + \omega_{\mathrm{pr}} \mathrm{Ri} ,
```

of the gradient Richardson number
``\mathrm{Ri} = N_{e,\mathrm{eff}}^2 / (2 \|\boldsymbol{\mathcal{E}}_D\|_F^2)``,
limited above by `Pr_max`. It approaches ``\mathrm{Pr}_n`` in neutral conditions
and grows with stability, and the stability dependence applies throughout the
column rather than only in the surface layer.

!!! note "TODO: not yet implemented"

    Three terms of the budget above are absent from `edmfx_tke.jl`: the
    dilatation term, the return-to-isotropy source
    ``R_{\mathrm{coh} \to \mathrm{iso}}`` that should receive the kinetic energy
    the pressure drag removes, and the
    ``\tfrac{2}{3} \kappa_{\mathrm{iso}} \boldsymbol{I}`` trace of the
    diffusive momentum flux. Together, these are what would close the
    resolved-kinetic-energy ↔ ``\kappa_{\mathrm{iso}}`` ↔ ``e_{\mathrm{tot}}``
    chain exactly. ``\kappa_{\mathrm{coh}}`` is also not currently diagnosed.

    The strain rate itself is not the deviatoric one either, and on the default
    path it is built from vertical gradients only
    (`compute_strain_rate_center_vertical`), so
    ``\|\boldsymbol{\mathcal{E}}_D\|_F^2`` above is in practice the vertical
    shear. Horizontal shear production is added only under
    `edmfx_sgs_horizontal_diffusive_flux`; see
    [Horizontal Diffusion](prophet_horizontal_diffusion.md).

    What *is* implemented, in `edmfx_tke_sources!`, is shear and buoyancy
    production evaluated from the same face diffusivities and face buoyancy
    gradient as the fluxes they parameterize, so the discrete conversions mirror
    the discrete fluxes. Transport and dissipation are applied in
    `edmfx_sgs_diffusive_flux_tendency!`.

### Mixing-length scales

The mixing length is a smooth minimum (`edmfx_scale_blending`, default
`SmoothMinimum`) of three physical scales, following [Lopez2020](@cite):

  - a **wall** scale ``l_W = \kappa_* (z - z_s) u_* / (c_m \sqrt{e_{\mathrm{sfc}}} \, \phi_m(\zeta))``,
    which reproduces Monin–Obukhov similarity in the surface layer. Here
    ``e_{\mathrm{sfc}}`` is ``\kappa_{\mathrm{iso}}`` in the first interior
    cell and ``\phi_m`` is the Businger–Dyer momentum stability function of
    ``\zeta = (z - z_s)/L``;
  - a **TKE-balance** scale
    ``l_{\mathrm{TKE}} = \sqrt{c_d \kappa_{\mathrm{iso}}^{3/2} / a_{pd}}`` with
    ``a_{pd} = c_m (2\|\boldsymbol{\mathcal{E}}_D\|_F^2 - N_e^2/\mathrm{Pr}_t) \sqrt{\kappa_{\mathrm{iso}}}``,
    the scale at which production balances dissipation (dropped from the blend
    where net production is non-positive). This balance uses the *un-augmented*
    ``N_e^2``, not the interface-aware ``N_{e,\mathrm{eff}}^2`` below, so it
    stays consistent with the TKE budget it parameterizes; the code carries the
    two as separate arguments;
  - a **buoyancy** scale
    ``l_N = \sqrt{c_b \kappa_{\mathrm{iso}}} / N_{e,\mathrm{eff}}``, used only
    in stably stratified air.

The blend is then capped by the wall distance and by the *resolvability filter
scale*

```math
\Delta_f = \max(\Delta x_h, \Delta z) ,
```

and floored at 1 m. The filter scale expresses that an eddy can be handed to the
resolved dynamics only if it is resolvable in every direction, so the coarsest
grid direction sets the cap. In single columns (``\Delta x_h \to \infty``) and
at global-model horizontal resolutions, the cap is inert and the mixing length
is purely physical, and therefore convergent under vertical refinement. In the
gray zone it binds at ``\Delta x_h``, and on isotropic grids it reduces to the
Deardorff bound ``l \le \Delta``.

The buoyancy frequency is the *moist effective* one,
``N_e^2 = \partial b / \partial z``, evaluated by the chain rule on the
prognostic state through the saturated and unsaturated branches
[OGorman2011, Marquet2011](@cite), so that latent heat release across phase
changes is accounted for. It is the same quantity that drives the buoyancy
production in the TKE budget and enters the Richardson number, which is what
keeps the length scale, the flux, and the budget mutually consistent.

**The dissipation coefficient is not independent.** In stably stratified air
with a buoyancy-limited mixing length, the local TKE balance is linear in
``\kappa_{\mathrm{iso}}``: there is no equilibrium TKE, only a threshold, and
turbulence grows or decays according to the sign of
``1/\mathrm{Ri} - 1/\mathrm{Pr}_t - c_d/(c_m c_b)``. As ``\mathrm{Pr}_t`` grows
in strong stability, that threshold approaches a critical Richardson number, so
the calibratable parameter is ``\mathrm{Ri}_c`` and

```math
c_d = \frac{c_m c_b}{\mathrm{Ri}_c}
```

is derived from it (`tke_dissipation_coefficient`). The resulting basis
``(c_m, c_b, \mathrm{Ri}_c)`` is nearly orthogonal: ``c_m`` scales the flux
magnitudes, ``c_b`` partitions TKE amplitude against buoyancy length, and
``\mathrm{Ri}_c`` sets the stable-regime cutoff.

### Capping inversions as unresolved interfaces

The subdomain decomposition addresses the gray zone of *horizontal* resolution.
The vertical direction has its own gray zone, and its canonical case is the
capping inversion of a cloud-topped boundary layer: the entrainment interfacial
layer is meters to tens of meters thick [Mellado2016](@cite), far below the
``\Delta z \gtrsim 100`` m typical of global models near cloud top. A centered
vertical gradient across such an interface averages the jump over a grid cell,
biasing ``N_e^2`` low and hence ``l_N`` and ``K_h`` high exactly where the
stability closure should shut mixing off. The bias weakens as ``\Delta z``
shrinks, so it breaks the resolution adaptivity PROPHET otherwise maintains, and
its integrated effect is a slow, spurious erosion of sharp inversions and the
stratocumulus decks beneath them.

PROPHET makes the stability closure interface-aware with two face-local terms.
First, a face buoyancy jump ``\Delta b = N_e^2 \Delta z`` is compatible with any
subgrid profile between a uniform gradient over ``\Delta z`` and a sheet
interface at the face. The discrete data cannot distinguish the two, but the
energetics of eddy excursions across the face can. Crossing a sheet requires
work ``\Delta b \, \ell``, which caps excursions at a penetration depth
``\ell_p \propto \kappa_{\mathrm{iso}} / \Delta b``. Traversing the same jump
spread over ``\Delta z`` requires only ``N_e^2 \ell^2 / 2``, and gives the
standard ``l_N``. A single effective frequency captures both limits:

```math
N_{e,\mathrm{eff}}^2 = N_e^2 + \frac{[(\Delta b)_+]^2}{c_b \kappa_{\mathrm{iso}}} ,
```

which replaces ``N_e^2`` in ``l_N`` and in the Richardson number that controls
``\mathrm{Pr}_t``. In the jump-dominated limit, ``l_N \to \ell_p``, independent
of ``\Delta z`` and of the unknown interface thickness. In smooth regions, the
correction is relatively ``O((\Delta z / l_N)^2)``, so ``N_{e,\mathrm{eff}}^2``
is a consistent, second-order-accurate discretization of the same continuum
stability, and it acts as a smooth interface indicator with no mode switch. The
positive part restricts it to stable jumps.

Second, collapsing the down-gradient diffusivity at a sheet interface is correct
for turbulent mixing but leaves interfacial *entrainment*, which proceeds at a
finite velocity, unrepresented. Mixed-layer theory expresses that velocity
through an efficiency law [Lilly1968](@cite); its local, TKE-based analog is

```math
w_e = A \frac{\sqrt{\kappa_{\mathrm{iso}}}}{\max(\mathrm{Ri}_b, 1)} ,
\qquad
\mathrm{Ri}_b = \frac{\ell_e \Delta b}{\kappa_{\mathrm{iso}}} ,
```

with ``\ell_e`` the energy-containing eddy scale, taken as the minimum of
``l_W`` and ``l_{\mathrm{TKE}}``, which unlike ``l_N`` are not suppressed by the
interface itself. The floor on ``\mathrm{Ri}_b`` restricts the efficiency law
to the strong interfaces it describes and bounds ``w_e`` by the turbulent
velocity scale. To
express the entrainment flux ``-w_e \Delta \psi`` in the down-gradient form the
model discretizes, it suffices to add a face diffusivity

```math
K_e = \gamma \, w_e \, \Delta z ,
\qquad
\gamma = 1 - \frac{N_e^2}{N_{e,\mathrm{eff}}^2} \in [0, 1] ,
```

since the discrete face flux is then
``K_e \Delta \psi / \Delta z = \gamma w_e \Delta \psi``. The gate ``\gamma``,
the fraction of the effective stability carried by the jump term, approaches one
at sheet interfaces and vanishes as ``(\Delta z / l_N)^2`` where the
stratification is resolved, so ``K_e \to 0`` doubly as ``\Delta z \to 0`` and
the standard local closure is recovered. At coarse ``\Delta z`` over a sharp
inversion, the entrainment flux is resolution-independent by construction.
Setting the single new constant ``A`` (`EDMF_interface_entr_efficiency`) to zero
recovers the pure stability closure.

``K_e`` is added to the face diffusivities for all scalars and for momentum, so
energy, water, and momentum transport stay conservative and mutually consistent.
The TKE buoyancy term is evaluated from the same face diffusivities and the same
face gradient, so the interfacial sink ``-\gamma w_e \Delta b`` per face is
carried without a separate term. That sink is bounded by
``A \kappa_{\mathrm{iso}}^{3/2} / \ell_e``, a fixed multiple of the dissipation,
which is how the efficiency inherits the classical energetic bound on
entrainment.

**Face-native evaluation.** The whole stability pipeline is evaluated at the
faces where the diffusive fluxes live. Vertical differences of cell means
(``\Delta h_s``, ``\Delta q_t``, and hence ``\Delta b``) come from the two-point
face stencil with no interpolation and no loss of sharpness; pointwise factors
(thermodynamic chain-rule coefficients, ``\kappa_{\mathrm{iso}}``, the strain
rate) vary smoothly between adjacent cells and are interpolated arithmetically.
Evaluating the closure at centers and interpolating the resulting diffusivity
instead cannot confine the collapse to the interface. A cell that registers the
jump carries its collapsed diffusivity to *both* of its faces, so it throttles
mixing on the boundary-layer side too. Meanwhile the interpolation across the
inversion hands about half the boundary-layer diffusivity to the face where the
flux should vanish.

!!! warning "Validity domain"

    This closure targets strong, mixed-layer-capping inversions, where the
    buoyancy jump dominates the interface. In that regime, which covers
    stratocumulus and shallow cumulus under a capping inversion, cloud cover
    and inversion height converge under vertical refinement with the closure
    active. It does not extend to weak, moisture-dominated interfaces: at a
    trade-wind inversion with a small buoyancy jump but a large humidity jump,
    the down-gradient form transports moisture up the jump faster than it warms
    and dries the layer, and on a coarse grid the thick inversion-base cell can
    saturate. Coarse-grid
    trade-cumulus cloud cover therefore remains resolution-dependent for any
    ``A``. Entrainment at cumulus tops is localized to penetrating plumes and
    belongs to the entrainment closures above, not to a grid-mean face
    diffusivity. Calibrate ``A`` against equilibrium (a day or longer) targets;
    spin-up snapshots reward values that fail at equilibrium.

## Variances and cloud fraction

Cloud fraction and microphysical rates are strongly nonlinear in temperature and
humidity, so evaluating them at the subdomain mean state biases them. PROPHET
instead integrates them over an assumed subgrid distribution of ``(T, q_t)``
within each subdomain, by Gauss–Hermite quadrature.

The distribution's *spread* comes from a diagnostic closure. Earlier EDMF
formulations carried the environmental covariance prognostically
[Cohen2020](@cite), with its own entrainment, detrainment, and source terms;
consistent with the unified treatment of TKE and of the diffusive fluxes,
PROPHET uses the classical production–dissipation balance of
[Sommeria1977, Mellor1977](@cite) instead,

```math
\sigma_\psi^2 = c_\sigma \, l^2 \, |\nabla \psi|^2 ,
```

a single isotropic intra-subdomain variance shared across subdomains, driven by
grid-mean gradients with the grid-mean mixing length. The cross-covariance is
set through a prescribed correlation,

```math
\sigma_{\psi \phi} = r_{\psi \phi} \, \sigma_\psi \, \sigma_\phi ,
```

rather than the gradient-product form
``c_\sigma l^2 \nabla \psi \cdot \nabla \phi``, which would give a singular
covariance matrix whenever the two gradients are collinear, as they generically
are at coarse horizontal resolution, where vertical gradients of opposite sign
dominate. The implementation evaluates the gradient closure for
``(\theta_{li}, q_t)``, converts to a temperature variance through the
thermodynamic Jacobian ``\partial T / \partial \theta_{li}``, and prescribes the
``T``–``q_t`` correlation as a constant (`Tq_correlation_coefficient`).

The total grid-mean subgrid covariance adds the inter-subdomain spread to this
intra-subdomain part [Lappen2001, Siebesma2007](@cite),

```math
\Sigma_{\psi \phi}^{\mathrm{tot}} = \Sigma_{\psi \phi}^{\mathrm{iso}}
  + \sum_{m} \frac{\hat{\rho}^m}{\rho} (\psi^m - \psi)(\phi^m - \phi) ,
```

so that draft–environment asymmetry is carried by the spread while the
isotropic piece supplies a uniform background. Because the coherent contrast is
assigned to the spread, ``c_\sigma`` is the *intra*-subdomain coefficient and is
correspondingly smaller than an effective grid-scale value.

Given the mean state and this covariance, cloud fraction and microphysics are
driven from a single reconstruction of the local condensate over the quadrature
points, with a Lagrange multiplier chosen so that the quadrature mean reproduces
the *prognostic* mean condensate exactly. The prognostic equations carry
non-equilibrium condensate (supersaturation not yet condensed at the resolved
scale, persistent supercooled liquid), while fluctuations within a subdomain are
assumed to relax onto the saturation curve
[Sommeria1977, Mellor1977, Bechtold1995](@cite). The cloud fraction is then the
fraction of the distribution carrying positive local condensate, evaluated with
a variance augmented by a non-equilibrium floor that scales with the saturation
specific humidity and is released as the subdomain mean saturates. That floor
keeps the cloud fraction from being driven to one by a vanishing equilibrium
variance, and loosely parallels the critical-relative-humidity scaling of
statistical cloud schemes [Quaas2012](@cite). Microphysical rates are evaluated
at each quadrature point and aggregated linearly, which preserves total water
and total energy at the subdomain level because the bulk rates preserve them
pointwise.

Because the variance depends on the mixing length, the mixing length on the
buoyancy gradient, and the buoyancy gradient on the cloud fraction, the closure
is solved as a fixed point at each timestep. Two Picard iterations are followed
by a guarded Aitken ``\Delta^2`` extrapolation, which is applied only when
successive increments change sign, so that the accelerated value lies between
computed iterates.

The grid-mean cloud fraction is the volume-fraction-weighted sum
``f_c^{\mathrm{tot}} = \sum_m a^m f_c^m``. In the current implementation, the
quadrature closure is evaluated for the environment, which carries the bulk of
the intra-subdomain variability; the drafts, comparatively homogeneous and
near-saturated where cloudy, use a condensate-presence indicator.

!!! note "This part of the scheme is in flux"

    The cloud-fraction and SGS-microphysics closures are under active
    development; the phase partition, the treatment of skewness, and the floor
    parameters in particular. This page deliberately states only their
    structure. The current forms live in
    [cloud_fraction.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/cloud_fraction.jl)
    and
    [sgs_quadrature.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/sgs_quadrature.jl),
    with the microphysics coupling in
    [Microphysics](microphysics.md); the configuration keys are `cloud_model`,
    `use_sgs_quadrature`, `sgs_distribution`, and `quadrature_order`.

## Boundary conditions

**Vertical velocity.** The rigid lower boundary gives every subdomain the same
Dirichlet condition as the grid mean, ``w^m(z_s) = w(z_s)``, which vanishes over
flat terrain. Draft velocities at interior faces then develop from buoyancy in
the lowest cell. The model top is likewise rigid for every subdomain.

**Draft area.** No Dirichlet condition is imposed. Because the vertical velocity
vanishes at the surface, the advective mass flux through the bottom face is
identically zero, and the lowest cell is treated as an interior cell whose area
evolves prognostically, following [Christopoulos2024](@cite). Drafts are instead
generated by a volumetric mass source in that cell.

**Surface mass source.** The source injects the most buoyant fraction of
surface-layer air into the draft. Its magnitude is

```math
F_{\mathrm{sfc}} = a_s \, \rho \, w_* ,
\qquad
a_s = a_{s,\max} \frac{w_*^3}{w_*^3 + c_u u_*^3} ,
\qquad
w_* = \left[ \left( z_i \overline{w'b'}_s \right)_+ \right]^{1/3} ,
```

with ``w_*`` the Deardorff convective velocity scale, ``z_i`` a prescribed
boundary-layer depth (`convective_zi`), and ``a_s`` a diagnostic estimate of the
surface-layer area occupied by active thermals that interpolates between free
convection (``a_s \to a_{s,\max}``) and shear-dominated conditions
(``a_s \to 0``). It vanishes in stable boundary layers. This closure is
resolution-aware in the same sense as the mass-flux fluxes: once the grid
resolves boundary-layer thermals, the resolved state carries the anomalies and
the seeding shuts off.

The injected air carries the conditional mean of the upper ``a_s`` tail of the
surface-layer distribution,

```math
\psi^{\mathrm{entr}}(z_0) = \psi(z_0) + D_{\mathcal{P}}(a_s) \, \sigma_{\psi,s} ,
```

where ``\sigma_{\psi,s}`` is the surface-layer standard deviation from
Monin–Obukhov similarity theory [Tan2018, Cohen2020, LopezGomez2022](@cite) and
``D_{\mathcal{P}}(a_s)`` is the conditional mean of the distribution
``\mathcal{P}`` over its top ``a_s`` fraction; for a standard Gaussian,
``D_{\mathcal{P}} = \phi(z_*)/a_s`` with ``z_* = \Phi^{-1}(1 - a_s)``. Surface
buoyancy fluctuations are observed to be positively skewed in the convective
boundary layer [Wyngaard1974](@cite), and the form above accommodates any
``\mathcal{P}``; the implementation is Gaussian-only. Together with the flux
boundary condition below, this reproduces the phenomenology that organized
drafts are seeded by anomalously buoyant near-surface parcels while the surface
flux supplies the bulk forcing.

The mass source is capped so that the drafts cannot transport more than a
fraction ``\alpha`` = `sfc_mass_flux_cap_fraction` of any surface scalar flux,
leaving at least ``1 - \alpha`` to the environment:

```math
F_{\mathrm{sfc}} \le \alpha \,
  \frac{\rho \overline{w' \chi'}|_s}{\chi^{\mathrm{entr}} - \chi^0}
  \quad \text{for } \chi \in \{h_s, q_t\} .
```

**Scalars.** Surface forcing on the draft scalars is a *flux* condition, not a
prescribed value: the same Monin–Obukhov surface flux that forces the grid-mean
scalar forces every subdomain scalar,

```math
\rho \, \hat{\boldsymbol{n}} \cdot
  \boldsymbol{\mathcal{F}}_\psi^{\mathrm{diff}} \big|_{z_s}
= \rho \overline{w' \psi'} \big|_s ,
\qquad \psi \in \{h_s, q_t\} ,
```

with ``\rho \overline{w' q_t'}|_s`` the evaporation rate and
``\rho \overline{w' h_s'}|_s`` the sum of the sensible and latent heat fluxes
(under ``\boldsymbol{\mathcal{F}}_{h_s}^{\mathrm{diff}} \approx \boldsymbol{\mathcal{F}}_h^{\mathrm{diff}}``).
Draft–environment thermodynamic contrasts at the surface are therefore not
prescribed; they develop from the seeding and the buoyant production of vertical
velocity. Diffusive surface fluxes of condensate and precipitation species
vanish, mirroring the grid-mean treatment: surface-layer turbulence is not
assumed to produce systematic phase imbalances between subdomains. See
[Surface Conditions](surface_conditions.md) for the bulk exchange laws.

**Turbulence kinetic energy.** A flux condition motivated by surface-layer
scaling of the TKE budget,

```math
\rho \overline{w' \kappa_{\mathrm{iso}}'} \big|_{z_s} = c_k \rho u_*^3 ,
```

with ``c_k`` = `tke_surf_flux_coeff` of order one. This is the surface-layer
limit of the more elaborate condition of [Lopez2020, Christopoulos2024](@cite),
and requires the first interior level to lie within the surface layer.

## Parameters

Closure parameters come from ClimaParams and are overridden through the `toml:`
key of a configuration file; `toml/prognostic_edmfx*.toml` hold the tuned sets.
The table maps the symbols above onto the accessor in
[`ClimaAtmos.Parameters`](@ref ClimaAtmos.Parameters) and the ClimaParams name.
The full list of fields is in the docstring of
[`ClimaAtmos.Parameters.TurbulenceConvectionParameters`](@ref).

| Symbol                                                              | Accessor                                                                                         | ClimaParams name                                                                                        |
|:------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------ |:------------------------------------------------------------------------------------------------------- |
| ``a_{\min}``, ``a_{\max}``                                          | `min_area`, `max_area`                                                                           | `EDMF_min_area`, `EDMF_max_area`                                                                        |
| ``a_{s,\max}``                                                      | `max_surface_area`                                                                               | `EDMF_max_surface_area`                                                                                 |
| ``c_u``, ``z_i``, ``\alpha``                                        | `sfc_mass_flux_ustar_coeff`, `convective_zi`, `sfc_mass_flux_cap_fraction`                       | `EDMF_sfc_mass_flux_ustar_coeff`, `EDMF_convective_zi`, `EDMF_sfc_mass_flux_cap_fraction`               |
| ``c_\varepsilon``, ``1/L_\varepsilon``                              | `entr_coeff`, `entr_inv_length`                                                                  | `entr_coeff`, `entr_inv_length`                                                                         |
| ``c_{\varepsilon b}``, ``\tau_\varepsilon^{-1}``                    | `entr_buoy_coeff`, `entr_inv_tau`                                                                | `entr_buoy_coeff`, `entr_inv_tau`                                                                       |
| ``\tau_{\max}^{-1}``                                                | `entr_detr_buoy_inv_tau_max`                                                                     | `entr_detr_buoy_inv_tau_max`                                                                            |
| ``c_1 \dots c_6`` (Π groups)                                        | `entr_param_vec`                                                                                 | `entr_param_vec`                                                                                        |
| ``\varepsilon_t``, ``c_t``                                          | `turb_entr_param_vec`                                                                            | `turb_entr_param_vec`                                                                                   |
| ``c_\delta``, ``c_{\delta \nabla}``                                 | `detr_buoy_coeff`, `detr_massflux_vertdiv_coeff`                                                 | `detr_buoy_coeff`, `detr_massflux_vertdiv_coeff`                                                        |
| ``s_{\min}``, ``p_{\min}``                                          | `min_area_limiter_scale`, `min_area_limiter_power`                                               | `min_area_limiter_scale`, `min_area_limiter_power`                                                      |
| ``s_{\max}``, ``p_{\max}``                                          | `max_area_limiter_scale`, `max_area_limiter_power`                                               | `max_area_limiter_scale`, `max_area_limiter_power`                                                      |
| ``\alpha_b``, ``\alpha_d``                                          | `pressure_normalmode_buoy_coeff1`, `pressure_normalmode_drag_coeff`                              | same                                                                                                    |
| ``c_m``                                                             | `tke_ed_coeff`                                                                                   | `mixing_length_eddy_viscosity_coefficient`                                                              |
| ``c_b``                                                             | `static_stab_coeff`                                                                              | `mixing_length_static_stab_coeff`                                                                       |
| ``\mathrm{Ri}_c``                                                   | `Ri_crit`                                                                                        | `mixing_length_Ri_crit`                                                                                 |
| ``c_d``                                                             | `tke_dissipation_coefficient` (derived)                                                          | —                                                                                                       |
| ``\mathrm{Pr}_n``, ``\omega_{\mathrm{pr}}``, ``\mathrm{Pr}_{\max}`` | `Prandtl_number_0`, `Prandtl_number_scale`, `Pr_max`                                             | `mixing_length_Prandtl_number_0`, `mixing_length_Prandtl_number_scale`, `mixing_length_Prandtl_maximum` |
| ``c_k``                                                             | `tke_surf_flux_coeff`                                                                            | `mixing_length_tke_surf_flux_coeff`                                                                     |
| ``c_\sigma``                                                        | `diagnostic_covariance_coeff`, which is ``c_\sigma / 2`` (the closure carries the factor of two) | `diagnostic_covariance_coeff`                                                                           |
| ``r_{T,q_t}``                                                       | `Tq_correlation_coefficient`                                                                     | `Tq_correlation_coefficient`                                                                            |
| ``A``                                                               | `interface_entr_efficiency`                                                                      | `EDMF_interface_entr_efficiency`                                                                        |

The generated [Configuration Options](configuration_options.md) table lists the
YAML keys; [Configuring and Tuning PROPHET](prophet_howto.md) explains which of
these to change first.

## Where this is implemented

| Closure                                                                                 | Source                                                                                                                                         |
|:--------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------- |
| Entrainment, detrainment, area limiters                                                 | [edmfx_entr_detr.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_entr_detr.jl)                             |
| Buoyancy, pressure drag, surface mass flux                                              | [mass_flux_closures.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/mass_flux_closures.jl)                       |
| Mixing length, ``\mathrm{Pr}_t``, ``N^2_{e,\mathrm{eff}}``, ``K_e``, face diffusivities | [eddy_diffusion_closures.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/eddy_diffusion_closures.jl)             |
| Grid-mean mixing length for non-EDMF paths                                              | [gm_sgs_closures.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/gm_sgs_closures.jl)                             |
| TKE production, dissipation                                                             | [edmfx_tke.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_tke.jl)                                         |
| Variances, cloud fraction, Picard iteration                                             | [microphysics/cloud_fraction.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/cloud_fraction.jl) |
| SGS quadrature over ``(T, q_t)``                                                        | [microphysics/sgs_quadrature.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/sgs_quadrature.jl) |
| Surface conditions for the drafts                                                       | [edmfx_boundary_condition.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_boundary_condition.jl)           |
| Closure parameter set                                                                   | [parameters/Parameters.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameters/Parameters.jl)                                      |
