# Horizontal EDMF diffusion

The EDMFX sub-grid scale (SGS) diffusive closure is, by default, vertical only: it parameterizes turbulent transport along the column with an eddy-diffusivity closure and applies the resulting tendency to the grid mean (and, under `edmfx_vertical_diffusion`, applies the grid-mean tendencies to the prognostic updraft scalars).
Two opt-in configuration options add the analogous horizontal down-gradient terms, both defaulting to `false`:

  - `edmfx_sgs_horizontal_diffusive_flux` adds the horizontal component of the grid-mean environment SGS diffusive flux;
  - `edmfx_horizontal_diffusion` applies the grid-mean horizontal diffusion tendencies to the prognostic EDMFX updrafts, and requires `edmfx_sgs_horizontal_diffusive_flux`.

Both terms are always explicit: they are applied in the explicit remainder tendency, independently of `diff_mode`, and never enter the implicit column solve.
Each requires a horizontal discretization, so both return immediately on single columns.

## Horizontal eddy diffusivity

The horizontal eddy viscosity and diffusivity use the same TKE-based closure as the vertical flux,

```math
K_u = c_m \, l \, \sqrt{e}, \qquad K_h = K_u \,/\, \mathrm{Pr}_t ,
```

where ``l`` is the horizontal mixing length (see [Mixing length](@ref)), ``e`` is the environment turbulent kinetic energy (TKE), ``c_m`` is the eddy-diffusivity coefficient, and ``\mathrm{Pr}_t`` the turbulent Prandtl number.
`set_horizontal_diffusivities!` evaluates ``K_u`` and ``K_h`` at cell centers on each update of the explicit precomputed cache, storing them in `ᶜK_u_h` and `ᶜK_h_h`.
The horizontal tendencies and the `edth`/`evuh` diagnostics read these cached fields.

Unlike the vertical face pipeline (`set_face_diffusivities!`), the horizontal diffusivities do not include the interfacial entrainment diffusivity ``K_e``: that term parameterizes vertical entrainment across an unresolved inversion face (``K_e = \gamma \, w_e \, \Delta z``) and has no horizontal analogue.

## Grid-mean flux

The grid-mean tendencies from `edmfx_sgs_horizontal_diffusive_flux` are, on cell centers,

```math
\partial_t (\rho e_\text{tot}) \mathrel{+}= \nabla_h \cdot \left[ \rho \, K_h \left( \nabla_h s_\text{d} + \sum_{\mu} (h_\mu + \Phi) \, \nabla_h q_\mu \right) \right],
```

```math
\partial_t (\rho q_\text{tot}) \mathrel{+}= \nabla_h \cdot (\rho \, K_h \, \nabla_h q_\text{tot}),
\qquad
\partial_t (\rho \chi) \mathrel{+}= \nabla_h \cdot (\rho \, \alpha_\chi \, K_h \, \nabla_h \chi),
```

where ``\nabla_h`` and ``\nabla_h\cdot`` are the horizontal gradient and weak divergence, ``s_\text{d}`` is the dry static energy, ``h_\mu`` and ``q_\mu`` are the specific enthalpy and specific humidity of the water species ``\mu \in \{\text{vap}, \text{liq}, \text{ice}\}``, and ``\Phi`` is the geopotential.
The total enthalpy uses the dry-static-energy + water-enthalpy decomposition of the vertical flux, which avoids the spurious enthalpy flux that diffusing ``h_\text{tot}`` directly would carry with the dry-air mass.
The gradient is materialized by `ᶜtotal_enthalpy_gradientₕ!` in two accumulation broadcasts before the weak divergence is applied: a single broadcast holding all four spectral gradients exceeds GPU kernel parameter limits on extruded spaces with warped topography.
A regression test asserts that the ``\rho e_\text{tot}`` tendency equals the sum of the four constituent fluxes (dry static energy, vapor, liquid, ice).

The tracer ``\chi`` ranges over the grid-mean SGS tracers: microphysics species (cloud, precipitation, and, for two-moment schemes, number concentrations) and any passive tracers.
Sedimenting microphysics species carry the tracer diffusion factor ``\alpha_\chi = \alpha`` (`α_vert_diff_tracer`); all other tracers use the unscaled ``K_h`` (``\alpha_\chi = 1``), matching the vertical flux.
The total-specific-humidity flux additionally enters the moist-air-mass tendency,

```math
\partial_t \rho \mathrel{+}= \nabla_h \cdot (\rho \, K_h \, \nabla_h q_\text{tot}),
```

while the condensate and precipitation fluxes do not.

When prognostic TKE is active, the horizontal flux transports TKE and adds the horizontal shear production,

```math
\partial_t (\rho e) \mathrel{+}= \nabla_h \cdot (\rho \, K_u \, \nabla_h e) + 2 \rho \, K_u \, \mathcal{S}_h : \mathcal{S}_h ,
```

with ``\mathcal{S}_h`` the strain rate built from horizontal gradients only.
The shear production is positive definite; the production from vertical gradients and its stencil are applied by the vertical TKE tendency.

The momentum tendency is the horizontal weak divergence of the SGS stress ``\tau = -2 K_u \mathcal{S}``, with ``\mathcal{S}`` the full three-dimensional strain rate of the grid-mean velocity,

```math
\partial_t u_h \mathrel{-}= \frac{1}{\rho} \nabla_h \cdot (\rho \, \tau),
\qquad
\partial_t u_3 \mathrel{-}= \frac{1}{\rho} \nabla_h \cdot (\rho \, \tau),
```

evaluated on cell centers for the horizontal wind and on cell faces for the vertical wind.
The vertical wind is included because the horizontal flux of vertical momentum ``\overline{u_h' w'}`` is a covariance, which the TKE (the half-trace of the velocity covariance) does not carry, so the down-gradient stress is its only representation.
The vertical flux ``\overline{w' w'}`` is a variance represented by the TKE, the mass flux, and the pressure closure, so it is not applied as down-gradient diffusion of the vertical wind; the vertical stress divergence on the horizontal wind is handled by the vertical diffusion pathway.

## Updraft horizontal diffusion

The option `edmfx_horizontal_diffusion` switches on horizontal diffusion of the prognostic updraft variables.
It requires `edmfx_sgs_horizontal_diffusive_flux`: each subdomain scalar receives the specific tendency of the corresponding grid-mean flux, so every subdomain inherits the same horizontal diffusion as the grid box, matching the uniform vertical treatment of `edmfx_vertical_diffusion`.
For each updraft ``j``, the total specific humidity and the SGS tracers receive

```math
\partial_t \chi^j \mathrel{+}= \frac{1}{\rho} \nabla_h \cdot (\rho \, \alpha_\chi \, K_h \, \nabla_h \chi),
```

with ``\alpha_\chi`` as in the grid-mean fluxes, and the moist static energy receives the grid-mean total-enthalpy tendency, ``\partial_t \mathrm{mse}^j \mathrel{+}= \partial_t(\rho e_\text{tot}) \, / \, \rho``.
The updraft dry-air mass is unchanged by the water flux, so the area-weighted density ``\rho a^j`` receives the counterpart tendency ``\rho a^j / (1 - q_\text{tot}^j)`` times the ``q_\text{tot}^j`` tendency, mirroring the grid-mean moist-air mass correction.
Diffusing each subdomain's own scalars instead would erode the updraft-environment contrasts that the mass-flux decomposition maintains.

## Mixing length

The eddy diffusivity depends on direction only through the mixing length ``l``: the TKE ``e`` is isotropic.
The horizontal mixing length is the full physical mixing length ``l_\text{phys}`` of the Lopez-Gomez et al. (2020) closure (the smooth minimum of the wall, TKE-balance, and static-stability scales, unchanged from the vertical closure), limited from above by a grid scale.
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

At coarse horizontal resolution (for example global runs with ``\Delta x_h`` of tens of kilometers) the horizontal limiter rarely binds, the horizontal diffusivity reduces to the isotropic environment value, and the term is typically negligible next to resolved horizontal transport.
Both options are off by default for that reason.

The Smagorinsky-Lilly and anisotropic-minimum-dissipation closures already supply horizontal SGS diffusion of the same fields, so combining `edmfx_sgs_horizontal_diffusive_flux` with either is rejected at model construction, as is `edmfx_horizontal_diffusion` without `edmfx_sgs_horizontal_diffusive_flux`.

Because the term is explicit, it adds a horizontal diffusive stability limit on the timestep, ``\Delta t \lesssim \Delta x_h^2 / (2 K_h)``.
At the fine horizontal resolutions where this closure is intended, the timestep is in practice already set by the explicit horizontal acoustic limit ``\Delta t \lesssim \Delta x_h / c_s``, which is the more restrictive of the two.

## Configuration

```yaml
edmfx_sgs_horizontal_diffusive_flux: true   # default: false
edmfx_horizontal_diffusion: true            # default: false; requires the flux option
```

## Diagnostics

Three diagnostic variables expose the horizontal closure fields:

  - `lmixh`: the horizontal mixing length ``l_h`` [m], recomputed on demand with the grid-scale limit set by the horizontal node spacing;
  - `edth`: the horizontal eddy diffusivity for scalars ``K_h`` [m² s⁻¹], read from the cached `ᶜK_h_h`;
  - `evuh`: the horizontal eddy viscosity ``K_u`` [m² s⁻¹], read from the cached `ᶜK_u_h`.

The vertical counterparts are `lmix`, `edt`, and `evu`.
