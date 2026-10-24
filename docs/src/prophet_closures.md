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

Entrainment and detrainment parameterize the exchange of mass and of the
properties the exchanged fluid carries. PROPHET treats them as *relaxation rates*
``E^{mn}`` and ``\Delta^{mn}`` with units of inverse time, rather than as the
fractional rates per unit length ``\epsilon^{mn} = E^{mn}/w^m`` conventional in
mass-flux schemes. Rates introduce no coordinate dependence through an explicit
velocity. They also make plain that entrainment and detrainment relax subdomain
properties toward each other, on timescales ``1/E^{mn}`` and
``1/\Delta^{mn}``, which makes their numerical treatment and the
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

    In the paper, the pairwise mass flux
    ``\hat{\rho}^m E^{mn} = \hat{\rho}^n \Delta^{nm}`` has a *single* shared
    limiter, so the symmetry relation holds by construction. The code instead
    treats entrainment and detrainment as unrelated closures with independent
    one-sided limiters, and evaluates the entrainment relaxation with ``w^0``
    set to zero at the call site in place of the velocity difference. Setting
    ``w^0 = 0`` bounds the entrainment of a draft with ``w^j \approx 0``, whose
    area would otherwise grow without limit. The analytic area solve holds the
    draft area fixed through the Newton iteration, which closes that path, so
    restoring the velocity difference is a code-side task, as is a symmetric
    redesign of the limiters.

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
d^j = a^0 \, d_c^j \, (w^j - w^0) |w^j - w^0| , \qquad
d_c^j = \frac{\alpha_d}{2 H}
  \left( \frac{1}{\sqrt{\mathrm{clamp}(a^j, a_{\min}, a_{\max})}}
       + \frac{1}{\sqrt{\mathrm{clamp}(a^0, 1 - a_{\max}, 1)}} \right) ,
```

with ``\alpha_d`` = `pressure_normalmode_drag_coeff` and
``H = R_d T_{s,\mathrm{ref}} / g`` a fixed reference scale height, is gated on
`edmfx_nh_pressure` and contributes a quadratic sink to the stage equation for
``w^j``. This is the pairwise form above for the draft–environment pair with
``r^m = H \sqrt{a^m}``, so the harmonic mean gives the sum of ``1/\sqrt{a}``
terms. The two subdomains are floored differently: the draft radius may shrink
to ``a_{\min}``, so ``r^j \to 0`` and the drag on a vanishing draft diverges,
but the environment is clamped from below at ``1 - a_{\max}``. Because the
vertical-velocity solve runs before the area solve, it can be handed an
extrapolated ``a^0`` outside ``[0, 1]``, and the clamp keeps the drag finite
there; with the default ``a_{\max} = 0.7`` the environment factor never exceeds
``1/\sqrt{0.3}``. The coefficient ``d_c^j`` is computed by the function
`pressure_drag_coefficient` for the momentum equation and for the TKE
return-to-isotropy source, so the kinetic energy the drag removes from the draft
is the energy the TKE budget receives. The leading ``a^0`` is the pairwise
weighting ``\hat{\rho}^0 / \rho`` for the environment; the environment velocity
is eliminated in favor of ``w^j`` in the implicit solve, which brings in factors
of ``\rho / \hat{\rho}^0`` (see
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
diagnosable from the subdomain velocities already in the state. It is
``\kappa_{\mathrm{iso}}`` that the model prognoses, as `Y.c.ρtke` (enabled by
`prognostic_tke`).

The eddy diffusivity is closed from a single grid-mean ``\kappa_{\mathrm{iso}}``
(scaled by `edmfx_updraft_kh_diffusion_factor` in the drafts), because the
inter-subdomain transfers of isotropic TKE redistribute energy within the cell
and sum to zero over it [Tan2018, Lopez2020](@cite). Two of those transfers are
nonetheless sources of grid-mean TKE, because they convert *coherent* subdomain
kinetic energy into isotropic turbulence: the return-to-isotropy drag and the
shear mixing that detrainment produces.

The budget is the compressible TKE equation [Pope2000, Wyngaard2010](@cite):

```math
\frac{\partial (\rho \kappa_{\mathrm{iso}})}{\partial t}
  + \nabla \cdot (\rho \kappa_{\mathrm{iso}} \boldsymbol{u})
= \underbrace{\nabla \cdot (\rho K_m \nabla \kappa_{\mathrm{iso}})}_{\text{transport}}
+ \underbrace{2 \rho K_m \|\boldsymbol{\mathcal{E}}_D\|_F^2}_{\text{shear production}}
- \underbrace{\tfrac{2}{3} \rho \kappa_{\mathrm{iso}} \nabla \cdot \boldsymbol{u}}_{\text{dilatation}}
- \underbrace{\rho K_h \frac{\partial b}{\partial z}}_{\text{buoyancy production}}
+ \underbrace{\rho R_{\mathrm{coh} \to \mathrm{iso}}}_{\text{drag return-to-isotropy}}
+ \underbrace{\rho R_\Delta}_{\text{detrainment shear mixing}}
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

The two coherent-to-isotropic transfers convert kinetic energy that the drafts
carry as an organized vertical motion into turbulence the drafts themselves do
not resolve. The pressure drag decelerates a draft relative to its environment,
and the energy it removes reappears as isotropic TKE at the same rate,

```math
\rho R_{\mathrm{coh} \to \mathrm{iso}} = \sum_j \hat{\rho}^j a^0 d_c^j
  \left| w^j - w^0 \right|^3 ,
```

with ``d_c^j`` the drag coefficient of the pressure-drag closure above.
Detrainment mixes air leaving a draft into the environment at a velocity
contrast ``w^j - w^0``, and half of the corresponding kinetic energy goes into
isotropic turbulence,

```math
\rho R_\Delta = \sum_j \tfrac{1}{2} \Delta^j \hat{\rho}^j
  \left( w^j - w^0 \right)^2 .
```

The first term is applied only when `edmfx_nh_pressure` is on, since it is the
drag closure's counterpart; the second is always applied.

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

The turbulent Prandtl number follows [Li2015](@cite) (with an algebraic error in
the published expression corrected), as the positive root

```math
\mathrm{Pr}_t = \frac{X + \sqrt{\max(X^2 - 4 \mathrm{Pr}_n \mathrm{Ri}, 0)}}{2},
\qquad X = \mathrm{Pr}_n + \omega_{\mathrm{pr}} \mathrm{Ri} ,
```

of the gradient Richardson number
``\mathrm{Ri} = N_e^2 / (2 \|\boldsymbol{\mathcal{E}}_D\|_F^2)``,
limited above by `Pr_max`. It approaches ``\mathrm{Pr}_n`` in neutral conditions
and grows with stability, and the stability dependence applies throughout the
column rather than only in the surface layer.

!!! note "TODO: not yet implemented"

    Two terms of the budget above are absent from `edmfx_tke.jl`: the dilatation
    term and the ``\tfrac{2}{3} \kappa_{\mathrm{iso}} \boldsymbol{I}`` trace of
    the diffusive momentum flux. Together with the terms that are present, these
    are what would close the resolved-kinetic-energy ↔
    ``\kappa_{\mathrm{iso}}`` ↔ ``e_{\mathrm{tot}}`` chain exactly.
    ``\kappa_{\mathrm{coh}}`` is also not currently diagnosed.

    The strain rate itself is not the deviatoric one either. On the default
    path, the shear production is the work of the vertical momentum flux, which
    acts on the horizontal wind only, so
    ``\|\boldsymbol{\mathcal{E}}_D\|_F^2`` above (and in ``\mathrm{Ri}``) is
    the vertical shear ``\tfrac{1}{2} |\partial_z \boldsymbol{u}_h|^2``
    (`vertical_shear_norm_sqr`). Horizontal shear production is added only under
    `edmfx_sgs_horizontal_diffusive_flux`; see
    [Horizontal Diffusion](prophet_horizontal_diffusion.md).

    What *is* implemented, in `edmfx_tke_sources!`, is shear and buoyancy
    production evaluated at the faces from the same face diffusivities, face
    strain rate, and face buoyancy gradient as the fluxes they parameterize,
    and then interpolated to the centers, so the discrete conversions mirror
    the discrete fluxes: summed over a column, the shear production equals the
    kinetic energy the momentum flux removes from the horizontal wind, up to
    the interpolation of ``\rho``. Transport and dissipation are applied in
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
    where net production is non-positive);
  - a **buoyancy** scale
    ``l_N = \sqrt{c_b \kappa_{\mathrm{iso}}} / N_e``, used only
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
gray zone, it binds at ``\Delta x_h``, and on isotropic grids it reduces to the
Deardorff bound ``l \le \Delta``.

The buoyancy frequency is the *moist effective* one,
``N_e^2 = \partial b / \partial z``, evaluated by the chain rule on the
prognostic state through the saturated and unsaturated branches
[OGorman2011, Marquet2011](@cite), so that latent heat release across phase
changes is accounted for. It is the same quantity that drives the buoyancy
production in the TKE budget and enters the Richardson number, which keeps the
length scale, the flux, and the budget mutually consistent.

**The dissipation coefficient is derived.** In stably stratified air
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

### Unresolved inversions

The subdomain decomposition addresses the gray zone of *horizontal* resolution.
The vertical direction has its own gray zone, and its canonical case is the
capping inversion of a cloud-topped boundary layer: the entrainment interfacial
layer is meters to tens of meters thick [Mellado2016](@cite), far below the
``\Delta z \gtrsim 100`` m typical of global models near cloud top. A centered
vertical gradient across such an interface averages the jump over a grid cell,
biasing ``N_e^2`` low and hence ``l_N`` and ``K_h`` high exactly where the
stability closure should shut mixing off. The bias weakens as ``\Delta z``
shrinks, so it breaks the resolution adaptivity PROPHET otherwise maintains, and
its integrated effect is a slow erosion of sharp inversions and the
stratocumulus decks beneath them.

**Face-native evaluation.** The stability pipeline is evaluated at the faces
where the diffusive fluxes live. Vertical differences of cell means
(``\Delta h_s``, ``\Delta q_t``, and hence ``\Delta b``) come from the two-point
face stencil with no interpolation and no loss of sharpness; pointwise factors
(thermodynamic chain-rule coefficients, ``\kappa_{\mathrm{iso}}``, the strain
rate) vary smoothly between adjacent cells and are interpolated arithmetically.
Evaluating the closure at centers and interpolating the resulting diffusivity
instead cannot confine the collapse to the interface. A cell that registers the
jump carries its collapsed diffusivity to *both* of its faces, so it damps
mixing on the boundary-layer side too. Meanwhile, the interpolation across the
inversion hands about half the boundary-layer diffusivity to the face where the
flux should vanish.

!!! note "TODO: interface-aware stability"

    The face-native evaluation confines the collapse of the diffusivity to the
    interface face but leaves the centered estimate of ``N_e^2`` across it, so
    a coarse grid still over-mixes a sharp inversion. An effective stability
    that distinguishes a sheet interface from a uniform gradient over
    ``\Delta z``, with an explicit interfacial entrainment velocity, is the
    natural remedy; applied wherever the face buoyancy jump is positive it
    over-mixes smoothly stratified stable boundary layers, so it needs an
    indicator that separates sheet interfaces from resolved stratification.

## Variances and cloud fraction

Cloud fraction and microphysical rates are strongly nonlinear in temperature and
humidity, so evaluating them at the subdomain mean state biases them. PROPHET
instead integrates them over an assumed subgrid distribution of ``(T, q_t)``
within each subdomain, by Gauss–Hermite quadrature.

The distribution's *spread* comes from a diagnostic closure. Earlier EDMF
formulations carried the environmental covariance prognostically
[Cohen2020](@cite), with its own entrainment, detrainment, and source terms;
PROPHET instead diagnoses it from the resolved gradients. A subgrid parcel is
displaced by a Gaussian ``\boldsymbol{\delta}_\psi`` with covariance
``\mathbf{D}_\psi = \mathrm{diag}(\ell_{h,\psi}^2, \ell_{h,\psi}^2, \ell_{v,\psi}^2)`` and keeps its
``\theta_{li}`` and ``q_t``, so the variances and cross-covariance of the two scalars at a point are

```math
\Sigma_{\psi \psi} = \ell_{h,\psi}^2 \, |\nabla_h \psi|^2
  + \ell_{v,\psi}^2 \, (\partial_z \psi)^2 ,
\qquad
\Sigma_{\theta_{li} q_t} = r_{h,\theta q} \, \ell_{h,\theta} \ell_{h,q} \, \nabla_h \theta_{li} \cdot \nabla_h q_t
  + \ell_{v,\theta} \ell_{v,q} \, \partial_z \theta_{li} \, \partial_z q_t ,
```

with ``\psi \in \{\theta_{li}, q_t\}``. The moisture horizontal
scale is set by the grid, ``\ell_{h,q} = \ell_{h,0} = \sqrt{c_g} \, c_{\Delta x} \Delta x_h``,
with ``c_g = 1/12`` the variance of a linear field over a uniform cell and
``c_{\Delta x}`` (`sgs_variance_horizontal_scale_factor`) the scale factor, and
the potential-temperature horizontal scale is ``\ell_{h,\theta} = r_{\Delta x,\theta} \ell_{h,0}``
(`sgs_variance_theta_horizontal_scale_ratio`), with horizontal displacement
cross-correlation ``r_{h,\theta q}``
(`sgs_variance_horizontal_cross_correlation`), which lies in ``[-1, 1]``. It
is the variance a cell holds because the resolved field varies across it, and
it vanishes as the horizontal grid is refined.

The moisture vertical scale is a displacement length,
``\ell_{v,q}^2 = \ell_{v,0}^2 = 2 c_\sigma l_d^2``, with potential-temperature vertical scale
``\ell_{v,\theta} = r_{v,\theta} \ell_{v,0}``
(`sgs_variance_theta_vertical_scale_ratio`). Here, ``l_d = \min(\max(l_{\mathrm{disp}}, l_b), z - z_s)`` takes the larger of the
buoyancy-bounded [mixing length](@ref "Mixing-length scales")
``l_{\mathrm{disp}} = l / \sqrt{1 + 2 c_\sigma \max(N_{\mathrm{disp}}^2, 0) \, l^2 / (c_b \kappa_{\mathrm{iso}})}``
(``l_{\mathrm{disp}} = l`` without TKE) and the floor displacement scale

```math
l_b = \min\!\left( \sqrt{c_b \kappa_{\min}} \,
  \min\!\left(\tau_N, \dfrac{l_{\mathrm{disp}}}{\sqrt{c_b \kappa_{\mathrm{iso}} / (2 c_\sigma)}}, \tau_0\right), \;
  z - z_s, \; \Delta z \right) ,
\qquad
\tau_N = \begin{cases} 1/N_{\mathrm{disp}} , & N_{\mathrm{disp}}^2 > 0 , \\[4pt] \infty , & N_{\mathrm{disp}}^2 \le 0 , \end{cases}
\qquad \kappa_{\min} = \frac{w_{\min}^2}{2 c_\sigma c_b} ,
```

capped by the wall distance, with ``N_{\mathrm{disp}}^2`` the
larger of the dry buoyancy gradients on the cell's lower and upper bounding
faces. Evaluating ``N_{\mathrm{disp}}^2`` on the bounding faces rather than on
the Van Leer harmonic-mean center gradients keeps the buoyancy barrier active in
a cell immediately below a sharp capping inversion, and using the dry buoyancy
coefficients keeps the moist latent-heat reduction at cloud base from weakening
the stratification barrier. In neutral turbulent layers, ``l_{\mathrm{disp}} = l``
sets ``l_d`` and the vertical displacement variance ``\ell_{v,0}^2 = 2 c_\sigma l^2``
follows the production–dissipation balance of [Sommeria1977, Mellor1977](@cite),
while in stably stratified layers ``\ell_{v,0}^2 = 2 c_\sigma l_{\mathrm{disp}}^2``
is smoothly bounded by the adiabatic buoyancy excursion ``c_b \kappa_{\mathrm{iso}} / N_{\mathrm{disp}}^2``. The floor is the
displacement of unresolved vertical motions with velocity scale ``w_{\min}``:
gravity waves, mesoscale circulations and
convective remnants. The
displacement lasts until the parcel's vertical velocity decorrelates.
In stably stratified air buoyancy stops it after a time ``1/N_{\mathrm{disp}}``, giving
``\ell_{v,0} = w_{\min}/N_{\mathrm{disp}}``. In turbulent air the turbulence scrambles it after
its Lagrangian time scale ``l_{\mathrm{disp}}/\sqrt{c_b \kappa_{\mathrm{iso}} / (2 c_\sigma)}``; this approaches
``1/N_{\mathrm{disp}}`` where the mixing length exceeds the buoyancy scale and reduces to the
turnover time of the wall or TKE-balance scale where those are smaller, so it matches the
``1/N_{\mathrm{disp}}`` limit in strongly stratified air and shortens the displacement where the mixing length is
constrained below the buoyancy scale. In weakly stratified quiescent air, where ``1/N_{\mathrm{disp}}`` exceeds ``\tau_0``
(`sgs_variance_floor_time_scale`) and turbulence is weak, the displacement
persists for ``\tau_0``, giving ``\ell_{v,0} = w_{\min} \tau_0``; where the air
is neutral or unstable (``N_{\mathrm{disp}}^2 \le 0``) buoyancy imposes no
limit and the same ``\min(\tau_{\mathrm{turb}}, \tau_0)`` applies, so ``l_b``
is continuous across ``N_{\mathrm{disp}}^2 = 0``. This is the
vertical dispersion of a stochastic Lagrangian model in stratified flow. The cap by
the cell depth applies to the floor scale only, and the wall distance bounds
``l_d`` as a whole; ``l_d``
exceeds ``\Delta z`` wherever ``l_{\mathrm{disp}}`` does. A reversible displacement creates a fluctuation at a
fixed point, and the fluctuation condenses if it crosses saturation, so the
floor enters the variance and not the diffusivity, which keeps the unfloored
mixing length.

The velocity scale grows with the horizontal grid scale,

```math
w_{\min} = w_{\mathrm{ref}} \left( \frac{\Delta x_h}{\Delta x_{\mathrm{ref}}} \right)^p ,
```

(`sgs_variance_vertical_velocity_floor`,
`sgs_variance_floor_reference_resolution`,
`sgs_variance_floor_resolution_exponent`; a single column has no horizontal
grid scale and uses ``\Delta x_{\mathrm{ref}}``). The exponent follows from
the spectral slope of the displaced fields. For a spectrum ``E(k) \propto k^{-s}``, the variance below the grid scale is
``\int_{k > 2\pi/\Delta} E \, dk \propto \Delta^{s-1}``. The horizontal term
``\ell_h^2 |\nabla_h \psi|^2`` scales this way by itself: the variance of the
coarse-grained gradient is ``\int_{k < 2\pi/\Delta} k^2 E \, dk \propto \Delta^{s-3}``, and ``\ell_h^2 \propto \Delta^2``, so ``c_{\Delta x}`` is
independent of resolution for any ``s < 3``. The vertical term
``\ell_v^2 (\partial_z \psi)^2`` does not: the vertical gradient is a vertical
coarse-graining and does not change with ``\Delta x_h``, so the growth of
the sub-cell variance has to be carried by the amplitude,
``\ell_v \propto w_{\min} \propto \Delta x_h^{(s-1)/2}``. A ``-5/3`` spectrum
gives ``p = 1/3``. The code defaults ``c_{\Delta x} = 1.24``,
``w_{\mathrm{ref}} = 1.08`` m s⁻¹, ``r_{\Delta x,\theta} = 0.88``,
``r_{v,\theta} = 0.38``, ``r_{h,\theta q} = 0.74`` and ``\tau_0 = 300`` s
are a fit to the within-cell moments (``\overline{q_t'^2}``,
``\overline{\theta_{li}'^2}`` and their covariance) of 3.75 km global
simulations (MPAS, DYAMOND) aggregated onto the model's own cubed-sphere
nodes, with every closure input computed by the model's operators (the
spectral-element horizontal gradient invariant, the Van Leer vertical
gradient and ``N^2_{\mathrm{disp}}``) on the MPAS native levels, and with
the turbulent term from the model's own TKE climatology. ``c_{\Delta x}``
multiplies a specific discrete gradient, so it is only meaningful with that
operator. One set fits ne16 to ne60 and both the
130- and 63-level vertical grids within 1 % in cost of the per-grid fits.
``\tau_0`` is bounded only from below (``\gtrsim 100`` s): in the stratified
free troposphere ``1/N_{\mathrm{disp}}`` binds first. The exponent is held
at ``p = 2/3``, which joint fits across 60, 120 and 240 km give (the two
resolution pairs give the same exponent): an effective slope near ``-7/3``,
between the mesoscale ``-5/3`` and the synoptic ``-3``, which is where the
spectra of temperature and humidity lie at those scales.
In the large-eddy limit ``w_{\min}`` vanishes and the TKE closure carries the
variance.

The cross covariance follows from the same pair of displacements: the
horizontal displacements of ``\theta_{li}`` and ``q_t`` are correlated with
``r_{h,\theta q}`` and their vertical displacements are perfectly correlated,
so the joint displacement covariance is positive semi-definite for
``|r_{h,\theta q}| \le 1`` and the sampled correlation is

```math
r = \frac{\Sigma_{T q_t}}{\sqrt{\Sigma_{TT} \Sigma_{q_t q_t}}}
\qquad (\texttt{tq\_correlation\_model: diagnosed}, \text{default}),
```

after the thermodynamic Jacobian transform to the ``T`` basis. It lies in
``[-1, 1]`` by Cauchy–Schwarz, and its sign is set by the gradients: a parcel
displaced upward in stably stratified air that is moist below arrives colder
and moister, so ``r < 0`` through most of the free troposphere. Where the
variances vanish the diagnosed value falls back to the prescribed
`Tq_correlation_coefficient`, which `tq_correlation_model: constant` uses
everywhere.

The total-water standard deviation is then bounded by ``\sigma_q \le r_{\max} q_t`` (``r_{\max} = 0.5``),
because a wider Gaussian places a quadrature node at negative total water, and
condensing the clamped excess can drive the grid-mean vapor negative. The σ_q
bound rescales ``\Sigma_{T q_t}`` by the same factor so the diagnosed
correlation is invariant under the clamp.

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
[Sommeria1977, Mellor1977, Bechtold1995](@cite). Each species' share of the
reconstructed excess can be blended with its subdomain mean through the
parameters `sgs_liquid_uniform_fraction` and `sgs_ice_uniform_fraction`.
The equilibrium condensate the reconstruction relaxes toward is the expected
saturation excess ``\mathbb{E}[\max(0, S)]`` of the Gaussian, or, with
`sgs_variance_curvature: true`, of the vertical displacement acting on the
curved profile of the saturation excess. To second order in the displacement
``\delta = \ell_v x``,

```math
S = \mu_S + \ell_v \, \partial_z \tilde S \; x
  + \tfrac{1}{2} \ell_v^2 \, \partial_z^2 \tilde S \; (x^2 - 1) + h ,
```

with ``\partial_z \tilde S = \partial_z q_t - r_{v,\theta}\, (\partial q_{\mathrm{sat}}/\partial T) (\partial T/\partial \theta_{li})\, \partial_z \theta_{li}`` (and likewise
``\partial_z^2 \tilde S``), i.e. the saturation excess of a displaced parcel with
the ``\theta_{li}`` displacement scaled by ``r_{v,\theta}`` as in the covariance,
``x`` standard normal and ``h`` the Gaussian horizontal part.
The second derivative comes from the two bounding face gradients of ``q_t``
and ``\theta_{li}``, so the closure adds no parameter. Where the mean
saturation excess has a maximum in the column, as at cloud base, the curvature
is negative and the quadratic bounds ``S`` by the largest excess the displaced
profile reaches; the moist tail is then thinner than Gaussian and the
condensate stops growing with ``\sigma_S``. The expectation is the
truncated-Gaussian value plus the change the curvature makes to it, with the
vertical model in closed form (`displacement_condensate`).
The cloud fraction is then the fraction of the distribution carrying positive
local condensate, evaluated with a variance augmented by a non-equilibrium floor
that scales with the saturation specific humidity and is released as the
subdomain mean saturates. That floor keeps the cloud fraction from being driven
to one by a vanishing equilibrium variance, and loosely parallels the
critical-relative-humidity scaling of
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
``f_c^{\mathrm{tot}} = \sum_m a^m f_c^m``. In the current implementation the
cloud fraction is a single-domain quantity: the environment's PDF
inversion is applied to the grid-mean cloud condensate,
``f_c = f_c^0\!\left(a^0 q_c^0 + \sum_j a^j q_c^j\right)``, with no area weighting and
no separate draft term.

!!! note "Under active development"

    The cloud-fraction and SGS-microphysics closures are under active
    development; the phase partition, the treatment of skewness, and the floor
    parameters in particular. This page deliberately states only their
    structure. The current forms live in
    [cloud_fraction.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/cloud_fraction.jl)
    and
    [sgs_quadrature.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/sgs_quadrature.jl),
    with the microphysics coupling in
    [Microphysics](microphysics.md); the configuration keys are `cloud_model`,
    `use_sgs_quadrature`, `sgs_distribution`, `quadrature_order`,
    `tq_correlation_model` (default `diagnosed`), and `sgs_variance_curvature`
    (default `false`).

## Boundary conditions

**Vertical velocity.** The rigid lower boundary gives every subdomain the same
Dirichlet condition as the grid mean, ``w^m(z_s) = w(z_s)``, which vanishes over
flat terrain. Draft velocities at interior faces then develop from buoyancy in
the lowest cell. The model top is likewise rigid for all subdomains.

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
The full list of fields is in the docstrings of
[`ClimaAtmos.Parameters.TurbulenceConvectionParameters`](@ref) and, for the SGS
covariance, quadrature and cloud-fraction closures,
[`ClimaAtmos.Parameters.SGSQuadratureParameters`](@ref).

| Symbol                                                                                          | Accessor                                                                                                                                                                                                                             | ClimaParams name                                                                                        |
|:----------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |:------------------------------------------------------------------------------------------------------- |
| ``a_{\min}``, ``a_{\max}``                                                                      | `min_area`, `max_area`                                                                                                                                                                                                               | `EDMF_min_area`, `EDMF_max_area`                                                                        |
| ``a_{s,\max}``                                                                                  | `max_surface_area`                                                                                                                                                                                                                   | `EDMF_max_surface_area`                                                                                 |
| ``c_u``, ``z_i``, ``\alpha``                                                                    | `sfc_mass_flux_ustar_coeff`, `convective_zi`, `sfc_mass_flux_cap_fraction`                                                                                                                                                           | `EDMF_sfc_mass_flux_ustar_coeff`, `EDMF_convective_zi`, `EDMF_sfc_mass_flux_cap_fraction`               |
| ``c_\varepsilon``, ``1/L_\varepsilon``                                                          | `entr_coeff`, `entr_inv_length`                                                                                                                                                                                                      | `entr_coeff`, `entr_inv_length`                                                                         |
| ``c_{\varepsilon b}``, ``\tau_\varepsilon^{-1}``                                                | `entr_buoy_coeff`, `entr_inv_tau`                                                                                                                                                                                                    | `entr_buoy_coeff`, `entr_inv_tau`                                                                       |
| ``\tau_{\max}^{-1}``                                                                            | `entr_detr_buoy_inv_tau_max`                                                                                                                                                                                                         | `entr_detr_buoy_inv_tau_max`                                                                            |
| ``c_1 \dots c_6`` (Π groups)                                                                    | `entr_param_vec`                                                                                                                                                                                                                     | `entr_param_vec`                                                                                        |
| ``\varepsilon_t``, ``c_t``                                                                      | `turb_entr_param_vec`                                                                                                                                                                                                                | `turb_entr_param_vec`                                                                                   |
| ``c_\delta``, ``c_{\delta \nabla}``                                                             | `detr_buoy_coeff`, `detr_massflux_vertdiv_coeff`                                                                                                                                                                                     | `detr_buoy_coeff`, `detr_massflux_vertdiv_coeff`                                                        |
| ``s_{\min}``, ``p_{\min}``                                                                      | `min_area_limiter_scale`, `min_area_limiter_power`                                                                                                                                                                                   | `min_area_limiter_scale`, `min_area_limiter_power`                                                      |
| ``s_{\max}``, ``p_{\max}``                                                                      | `max_area_limiter_scale`, `max_area_limiter_power`                                                                                                                                                                                   | `max_area_limiter_scale`, `max_area_limiter_power`                                                      |
| ``\alpha_b``, ``\alpha_d``                                                                      | `pressure_normalmode_buoy_coeff1`, `pressure_normalmode_drag_coeff`                                                                                                                                                                  | same                                                                                                    |
| ``c_m``                                                                                         | `tke_ed_coeff`                                                                                                                                                                                                                       | `mixing_length_eddy_viscosity_coefficient`                                                              |
| ``c_b``                                                                                         | `static_stab_coeff`                                                                                                                                                                                                                  | `mixing_length_static_stab_coeff`                                                                       |
| ``\mathrm{Ri}_c``                                                                               | `Ri_crit`                                                                                                                                                                                                                            | `mixing_length_Ri_crit`                                                                                 |
| ``c_d``                                                                                         | `tke_dissipation_coefficient` (derived)                                                                                                                                                                                              | —                                                                                                       |
| ``\mathrm{Pr}_n``, ``\omega_{\mathrm{pr}}``, ``\mathrm{Pr}_{\max}``                             | `Prandtl_number_0`, `Prandtl_number_scale`, `Pr_max`                                                                                                                                                                                 | `mixing_length_Prandtl_number_0`, `mixing_length_Prandtl_number_scale`, `mixing_length_Prandtl_maximum` |
| ``c_k``                                                                                         | `tke_surf_flux_coeff`                                                                                                                                                                                                                | `mixing_length_tke_surf_flux_coeff`                                                                     |
| ``f_m``                                                                                         | `edmfx_updraft_kh_diffusion_factor`                                                                                                                                                                                                  | `edmfx_updraft_kh_diffusion_factor`                                                                     |
| ``c_\sigma``                                                                                    | `diagnostic_covariance_coeff` (``\ell_v^2 = 2 c_\sigma l_d^2``)                                                                                                                                                                      | `diagnostic_covariance_coeff`                                                                           |
| ``r_{T,q_t}``                                                                                   | `Tq_correlation_coefficient`                                                                                                                                                                                                         | `Tq_correlation_coefficient`                                                                            |
| ``c_g``, ``c_{\Delta x}``, ``w_{\mathrm{ref}}``, ``\Delta x_{\mathrm{ref}}``, ``p``, ``\tau_0`` | `sgs_variance_geometric_coeff`, `sgs_variance_horizontal_scale_factor`, `sgs_variance_vertical_velocity_floor`, `sgs_variance_floor_reference_resolution`, `sgs_variance_floor_resolution_exponent`, `sgs_variance_floor_time_scale` | `sgs_variance_geometric_coeff`, `sgs_variance_horizontal_scale_factor`, —, —, —, —                      |
| ``r_{\Delta x,\theta}``, ``r_{v,\theta}``, ``r_{h,\theta q}``                                   | `sgs_variance_theta_horizontal_scale_ratio`, `sgs_variance_theta_vertical_scale_ratio`, `sgs_variance_horizontal_cross_correlation`                                                                                                  | —, —, —                                                                                                 |
| ``r_{\max}``                                                                                    | `sgs_variance_max_rel_std` (bound ``\sigma_q \le r_{\max} q_t``)                                                                                                                                                                     | `sgs_variance_max_rel_std`                                                                              |

The generated [Configuration Options](configuration_options.md) table lists the
YAML keys; [Configuring and Tuning PROPHET](prophet_howto.md) explains which of
these to change first.

## Where this is implemented

| Closure                                              | Source                                                                                                                                         |
|:---------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------- |
| Entrainment, detrainment, area limiters              | [edmfx_entr_detr.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_entr_detr.jl)                             |
| Buoyancy, pressure drag, surface mass flux           | [mass_flux_closures.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/mass_flux_closures.jl)                       |
| Mixing length, ``\mathrm{Pr}_t``, face diffusivities | [eddy_diffusion_closures.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/eddy_diffusion_closures.jl)             |
| Grid-mean mixing length for non-EDMF paths           | [gm_sgs_closures.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/gm_sgs_closures.jl)                             |
| TKE production, dissipation                          | [edmfx_tke.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_tke.jl)                                         |
| Variances, cloud fraction, Picard iteration          | [microphysics/cloud_fraction.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/cloud_fraction.jl) |
| SGS quadrature over ``(T, q_t)``                     | [microphysics/sgs_quadrature.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/microphysics/sgs_quadrature.jl) |
| Surface conditions for the drafts                    | [edmfx_boundary_condition.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/edmfx_boundary_condition.jl)           |
| Closure parameter set                                | [parameters/Parameters.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameters/Parameters.jl)                                      |
