# PROPHET: Horizontal Diffusion

The PROPHET (`EDMFX` in code) subgrid-scale (SGS) diffusive closure is, by
default, vertical only: it parameterizes turbulent transport along the column
with an eddy-diffusivity closure and applies the resulting tendency to the grid
mean (and, under `edmfx_vertical_diffusion`, diffuses the prognostic updraft
scalars down their own vertical gradients). Two opt-in configuration options add
the analogous horizontal down-gradient terms, both defaulting to `false`:

  - `edmfx_sgs_horizontal_diffusive_flux` adds the horizontal component of the
    grid-mean SGS diffusive flux;
  - `edmfx_horizontal_diffusion` diffuses the prognostic EDMFX updraft scalars
    down their own horizontal gradients, and requires
    `edmfx_sgs_horizontal_diffusive_flux`.

Both terms are always explicit: they are applied in the explicit remainder
tendency, independently of `diff_mode`. Each requires a horizontal
discretization, so both return immediately on single columns.

## Horizontal eddy diffusivity

The horizontal eddy viscosity and diffusivity use the same TKE-based closure as
the vertical flux,

```math
K_{u,h} = c_m \, l_h \, \sqrt{\kappa_{\mathrm{iso}}}, \qquad K_{h,h} = K_{u,h} \,/\, \mathrm{Pr}_t ,
```

where ``l_h`` is the horizontal mixing length (see [Mixing length](@ref)),
``\kappa_{\mathrm{iso}}`` is the isotropic subgrid turbulent kinetic energy
(`Y.c.ρtke`, written ``e`` in some of the code comments), ``c_m`` is the
eddy-diffusivity coefficient, and ``\mathrm{Pr}_t`` the turbulent Prandtl
number. `set_horizontal_diffusivities!` evaluates ``K_{u,h}`` and ``K_{h,h}`` at
cell centers on each update of the explicit precomputed cache, storing them in
`ᶜK_u_h` and `ᶜK_h_h`. The horizontal tendencies and the `edth`/`evuh`
diagnostics read these cached fields.

## Grid-mean flux

The horizontal water-enthalpy and species fluxes mirror the vertical closure
([Diffusion](diffusion.md)): the enthalpy bracket of [Yatunin2026](@cite) (Eq.
33) is

```math
\mathcal{B}_h(T, q_{\mathrm{tot}}^{\mathrm{eff}}, q_{\mathrm{lcl}}, q_{\mathrm{icl}})
  = \nabla_h s_{\mathrm{d}}
  + (h_{\mathrm{v}} + \Phi) \, \nabla_h q_{\mathrm{tot}}^{\mathrm{eff}}
  + (h_{\mathrm{l}} - h_{\mathrm{v}}) \, \nabla_h q_{\mathrm{lcl}}
  + (h_{\mathrm{i}} - h_{\mathrm{v}}) \, \nabla_h q_{\mathrm{icl}} ,
```

where ``\nabla_h`` and ``\nabla_h\cdot`` are the horizontal gradient and weak
divergence, ``s_{\mathrm{d}}`` is the dry static energy, ``\Phi`` is the
geopotential, and ``q_{\mathrm{tot}}^{\mathrm{eff}} = q_{\mathrm{tot}} - q_{\mathrm{rai}} - q_{\mathrm{sno}}``
is the suspended water that diffuses. Each thermodynamic coefficient is
materialized into a center scratch scalar before the weak divergence, because
feeding the nested expression into ``\nabla_h\cdot`` exceeds GPU kernel
parameter limits on extruded spaces with warped topography.

For `PrognosticEDMFX`, the grid-mean horizontal diffusive flux is the subdomain
sum with center weights ``w^0 = \hat{\rho}^0 K_{h,h}`` and
``w^j = \hat{\rho}^j f_j K_{h,h}`` (``f_j =`` `edmfx_updraft_kh_diffusion_factor`,
included when `edmfx_horizontal_diffusion` is enabled):

```math
\partial_t (\rho e_{\mathrm{tot}}) \mathrel{+}=
  \nabla_h \cdot \left[ w^0 \, \mathcal{B}_h^0
    + \sum_{j=1}^{M} w^j \, \mathcal{B}_h^j \right] ,
```

```math
\partial_t (\rho q_{\mathrm{tot}}) \mathrel{+}=
  \nabla_h \cdot \left( w^0 \, \nabla_h q_{\mathrm{tot}}^{\mathrm{eff},0}
    + \sum_{j=1}^{M} w^j \, \nabla_h q_{\mathrm{tot}}^{\mathrm{eff},j} \right) ,
\qquad
\partial_t (\rho \chi) \mathrel{+}=
  \nabla_h \cdot \left( w^0 \, \nabla_h \chi^0
    + \sum_{j=1}^{M} w^j \, \nabla_h \chi^j \right) .
```

For `EDOnlyEDMFX`, the grid mean uses ``w = \rho K_{h,h}`` on the grid-mean
fields.

In the non-equilibrium 1-moment and 2-moment schemes, the suspended cloud
species ``q_{\mathrm{lcl}}``, ``q_{\mathrm{icl}}`` and their number
concentrations ``n_{\mathrm{lcl}}``, ``n_{\mathrm{ice}}`` diffuse down their own
horizontal gradients with the same weights, matching the phase-enthalpy fluxes
in ``\mathcal{B}_h``. Rain and snow do not diffuse horizontally, as they do not
diffuse in the vertical. Passive tracers diffuse with the same weights.

The aggregate water flux additionally enters the moist-air-mass tendency,

```math
\partial_t \rho \mathrel{+}= \partial_t (\rho q_{\mathrm{tot}})\big|_{\mathrm{diff},h} ,
```

while the suspended cloud-species fluxes do not.

When prognostic TKE is active, the horizontal flux transports TKE and adds the
horizontal shear production,

```math
\partial_t (\rho \kappa_{\mathrm{iso}}) \mathrel{+}= \nabla_h \cdot (\rho \, K_{u,h} \, \nabla_h \kappa_{\mathrm{iso}}) + 2 \rho \, K_{u,h} \, \mathcal{S}_h : \mathcal{S}_h ,
```

with ``\mathcal{S}_h`` the strain rate built from horizontal gradients only (the
tensor written ``\boldsymbol{\mathcal{E}}_D`` in
[Turbulence kinetic energy and eddy diffusivity](prophet_closures.md#Turbulence-kinetic-energy-and-eddy-diffusivity)).
The shear production is positive definite; the production from vertical
gradients and its stencil are applied by the vertical TKE tendency.

The momentum tendency is the horizontal weak divergence of the SGS stress
``\tau = -2 K_{u,h} \mathcal{S}``, with ``\mathcal{S}`` the full
three-dimensional strain rate of the grid-mean velocity,

```math
\partial_t u_h \mathrel{-}= \frac{1}{\rho} \nabla_h \cdot (\rho \, \tau),
\qquad
\partial_t u_3 \mathrel{-}= \frac{1}{\rho} \nabla_h \cdot (\rho \, \tau),
```

evaluated on cell centers for the horizontal wind and on cell faces for the
vertical wind. The vertical wind is included because the horizontal flux of
vertical momentum ``\overline{u_h' w'}`` is a covariance, which the TKE (the
half-trace of the velocity covariance) does not carry, so the down-gradient
stress is its only representation. The vertical flux ``\overline{w' w'}`` is a
variance represented by the TKE, the mass flux, and the pressure closure, so it
is not applied as down-gradient diffusion of the vertical wind; the vertical
stress divergence on the horizontal wind is handled by the vertical diffusion
pathway.

The stress and the shear production are not energetically consistent with each
other. The work done by the stress on the resolved flow is
``\tau : \nabla_h u``, which includes cross terms such as
``\mathcal{S}_{13} \, \partial_x w``, whereas the TKE receives only
``2 \rho \, K_{u,h} \, \mathcal{S}_h : \mathcal{S}_h``. The difference is
exactly those cross terms. Each form is chosen for a property the other lacks:
the stress uses the full strain rate because ``\mathcal{S}_{13}`` and
``\mathcal{S}_{23}`` are what represent ``\overline{u_h' w'}``, and the
production uses ``\mathcal{S}_h`` because
``2 \rho \, K_{u,h} \, \mathcal{S}_h : \mathcal{S}_h`` is positive definite
while ``\tau : \nabla_h u`` is sign-indefinite and can drive the TKE negative.
This is one of the reasons the option is off by default.

## Updraft horizontal diffusion

The option `edmfx_horizontal_diffusion` switches on horizontal diffusion of the
prognostic updraft variables and requires `edmfx_sgs_horizontal_diffusive_flux`.
For each updraft ``j``, the moist static energy ``\mathrm{mse}^j``, the total
specific humidity ``q_{\mathrm{tot}}^j``, the suspended cloud species
(`q_lcl`, `q_icl`, `n_lcl`, `n_ice`), and the passive SGS tracers diffuse down
their own horizontal gradients:

```math
\partial_t \chi^j \mathrel{+}= \frac{1}{\rho}
  \nabla_h \cdot \left( \rho f_j K_{h,h} \, \nabla_h \chi_{\mathrm{eff}}^j \right) ,
```

with
``q_{\mathrm{tot},\mathrm{eff}}^j = q_{\mathrm{tot}}^j - q_{\mathrm{rai}}^j - q_{\mathrm{sno}}^j``
and ``\mathrm{mse}_{\mathrm{eff}}^j`` excluding non-diffusing rain and snow
enthalpy, as in the vertical treatment of `edmfx_vertical_diffusion`. The
intensive updraft equation uses the grid-mean weight ``\rho f_j K_{h,h}``. In
the spectral-element weak divergence, the flux at a node draws on
``\hat{\rho}^j`` at every node of the element, so dividing the area-weighted
divergence by the local ``\hat{\rho}^j`` amplifies the tendency at draft edges
by up to the area ratio between nodes (in the vertical, the harmonic face mean
bounds that ratio). With the grid-mean weight the updraft relaxes toward its
horizontal neighbors at the rate of grid-mean diffusion. The grid mean receives
the conservative area-weighted flux ``w^j \nabla_h \chi^j`` above; the
difference between the two,
``\approx \rho f_j K_{h,h} \nabla_h a^j \cdot \nabla_h \chi^j`` at draft edges,
is absorbed by the environment residual, in the same way as the cross-area
terms of the vertical decomposition. The area-weighted density ``\hat{\rho}^j``
receives no tendency, matching `edmfx_sgs_diffusive_flux_tendency!`.

## Mixing length

The eddy diffusivity depends on direction only through the mixing length: the TKE ``\kappa_{\mathrm{iso}}`` is isotropic.
The horizontal mixing length is the full physical mixing length ``l_\text{phys}`` of the [Lopez2020](@cite) closure (the smooth minimum of the wall, TKE-balance, and static-stability scales, unchanged from the vertical closure), limited from above by a grid scale.
It differs from the vertical (master) mixing length only in that grid limiter: the horizontal length uses the spectral-element node scale ``\Delta x_h`` (`horizontal_filter_scale`), whereas the vertical pipeline uses the resolvability filter scale ``\Delta_f = \max(\Delta x_h, \Delta z)`` (`resolvability_filter_scale`),

```math
l_h = \min(l_\text{phys}, \, \Delta x_h),
\qquad
l = \min(l_\text{phys}, \, \Delta_f).
```

Wherever ``\Delta x_h \ge \Delta z`` (single columns and GCM resolutions), the two limiters coincide (``\Delta_f = \Delta x_h``), so ``l_h = l`` and the horizontal diffusivity equals the one built from the master mixing length.
They differ only where ``\Delta z > \Delta x_h`` (gray-zone and LES aspect ratios): there ``\Delta_f = \Delta z`` while the horizontal length remains limited by ``\Delta x_h``, giving the shorter horizontal scale ``l_h \le l`` and the ratio ``l_h / l = \Delta x_h / \Delta z`` wherever ``l_\text{phys}`` exceeds both spacings.

## When to enable it

Horizontal EDMF diffusion is intended for configurations whose horizontal node scale ``\Delta x_h`` is comparable to or smaller than the physical mixing length, so that the horizontal limiter ``\min(l_\text{phys}, \Delta x_h)`` binds.
This is the high-resolution and gray-zone regime, where an anisotropic SGS length scale matters.

At coarse horizontal resolution (for example global runs with ``\Delta x_h`` of tens of kilometers), the horizontal limiter rarely binds, the horizontal diffusivity reduces to the isotropic environment value, and the term is typically negligible next to resolved horizontal transport.
Both options are off by default for that reason.

The Smagorinsky–Lilly and anisotropic-minimum-dissipation closures ([Large-Eddy Simulation Closures](les_sgs.md)) already supply horizontal SGS diffusion of the same fields, so combining `edmfx_sgs_horizontal_diffusive_flux` with either is rejected at model construction, as is `edmfx_horizontal_diffusion` without `edmfx_sgs_horizontal_diffusive_flux`.

Because the term is explicit, it adds a horizontal diffusive stability limit on the timestep, ``\Delta t \lesssim \Delta x_h^2 / (2 K_{h,h})``.
At the fine horizontal resolutions where this closure is intended, the timestep is in practice already set by the explicit horizontal acoustic limit ``\Delta t \lesssim \Delta x_h / c_s``, which is the more restrictive of the two.

## Configuration

```yaml
edmfx_sgs_horizontal_diffusive_flux: true   # default: false
edmfx_horizontal_diffusion: true            # default: false; requires the flux option
```

## Diagnostics

Three diagnostic variables expose the horizontal closure fields:

  - `lmixh`: the horizontal mixing length ``l_h`` [m], recomputed on demand with the grid-scale limit set by the horizontal node spacing;
  - `edth`: the horizontal eddy diffusivity for scalars ``K_{h,h}`` [m² s⁻¹], read from the cached `ᶜK_h_h`;
  - `evuh`: the horizontal eddy viscosity ``K_{u,h}`` [m² s⁻¹], read from the cached `ᶜK_u_h`.

The vertical counterparts are `lmix`, `edt`, and `evu`.
