# Horizontal EDMF diffusion

The EDMFX environment sub-grid scale (SGS) diffusive flux is, by default, vertical only: it parameterizes turbulent transport along the column with an eddy-diffusivity closure and applies the resulting tendency to the grid mean. The horizontal component adds the analogous horizontal down-gradient flux. It is enabled with the configuration option `edmfx_sgs_horizontal_diffusive_flux` (default `false`).

## Formulation

The horizontal flux uses the same TKE-based eddy diffusivity as the vertical flux,

```math
K = c_m \, l \, \sqrt{e} \,/\, \mathrm{Pr}_t ,
```

where ``l`` is the environment mixing length, ``e`` is the environment turbulent kinetic energy (TKE), ``c_m`` is the eddy-diffusivity coefficient, and ``\mathrm{Pr}_t`` the turbulent Prandtl number. The grid-mean tendencies are, on cell centers,

```math
\partial_t (\rho e_\text{tot}) \mathrel{+}= \nabla_h \cdot \left[ \rho \, K_h \left( \nabla_h s_\text{d} + \sum_{\mu} (h_\mu + \Phi) \, \nabla_h q_\mu \right) \right],
```

```math
\partial_t (\rho \chi) \mathrel{+}= \nabla_h \cdot (\rho \, K_h \, \nabla_h \chi),
\qquad
\partial_t (\rho e) \mathrel{+}= \nabla_h \cdot (\rho \, K_u \, \nabla_h e),
```

where ``\nabla_h`` and ``\nabla_h\cdot`` are the horizontal gradient and weak divergence, ``s_\text{d}`` is the dry static energy, ``h_\mu`` and ``q_\mu`` are the specific enthalpy and specific humidity of the water species ``\mu \in \{\text{vap}, \text{liq}, \text{ice}\}``, ``\Phi`` is the geopotential, ``K_u`` is the eddy viscosity and ``K_h = K_u/\mathrm{Pr}_t``. The total enthalpy uses the dry-static-energy + water-enthalpy decomposition of the vertical flux, which avoids the spurious enthalpy flux that diffusing ``h_\text{tot}`` directly would carry with the dry-air mass. The tracer ``\chi`` ranges over the total specific humidity and the environment SGS tracers — the microphysics species (cloud, precipitation, and, for two-moment schemes, number concentrations) and any passive tracers — i.e. the same set the vertical flux diffuses. Sedimenting microphysics species are scaled by the tracer diffusion factor ``\alpha`` (`α_vert_diff_tracer`); other tracers use the unscaled ``K_h``, matching the vertical flux. The total-specific-humidity flux additionally enters the air-mass tendency ``\partial_t \rho``; condensate and precipitation fluxes do not.

The momentum tendency is the horizontal weak divergence of the subgrid-scale stress ``\tau = -2 K_u \mathcal{S}``, with ``\mathcal{S}`` the full three-dimensional strain rate of the grid-mean velocity,

```math
\partial_t u_h \mathrel{-}= \frac{1}{\rho} \nabla_h \cdot (\rho \, \tau),
\qquad
\partial_t u_3 \mathrel{-}= \frac{1}{\rho} \nabla_h \cdot (\rho \, \tau),
```

evaluated on cell centers for the horizontal wind and on cell faces for the vertical wind, mirroring the Smagorinsky-Lilly stress split by divergence direction; the vertical stress divergence is handled by the vertical diffusion pathway.

The corresponding shear production of TKE uses the strain rate built from horizontal gradients only, ``\mathcal{S}_h``,

```math
\partial_t (\rho e) \mathrel{+}= 2 \rho \, K_u \, \mathcal{S}_h : \mathcal{S}_h ,
```

which is positive definite; the production from vertical gradients and its stencil are unchanged. This mirrors the decoupled Smagorinsky-Lilly split, in which each directional norm is built from that direction's gradients.

## Updraft horizontal diffusion

The separate option `edmfx_horizontal_diffusion` (default `false`) switches on horizontal diffusion of the prognostic updraft variables, mirroring the vertical updraft diffusion of `edmfx_vertical_diffusion` with the horizontal mixing length: for each updraft,

```math
\partial_t \chi^j \mathrel{+}= \frac{1}{\rho^j} \nabla_h \cdot (\rho^j \, K_h \, \nabla_h \chi^j),
```

for the moist static energy, the total specific humidity, and the updraft SGS tracers (the latter scaled by ``\alpha``). The updraft dry-air mass is unchanged by the water flux, so the area-weighted density ``\rho a^j`` receives the counterpart tendency ``\rho a^j/(1 - q_\text{tot}^j)`` times the ``q_\text{tot}^j`` tendency, mirroring the hyperdiffusion treatment.

The horizontal flux mirrors the vertical EDMFX diffusive flux: the same variables, the same tracer set, and the same scaling, with horizontal rather than vertical operators.

## Anisotropic length scale

The eddy diffusivity depends on direction only through the mixing length ``l``: the TKE ``e`` is isotropic. In the Lopez-Gomez et al. (2020) closure the mixing length is the smooth minimum of physical scales (wall distance, TKE production–dissipation balance, static stability), limited from above by a grid scale,

```math
l = \min(l_\text{phys}, \, \Delta).
```

Unlike the Smagorinsky-Lilly closure, whose length scale is purely geometric (``l = c_s \Delta``), the EDMF length is physically based; only its grid-scale limiter is geometric. The vertical and horizontal fluxes therefore share the same physical length but use different limiters — the cell thickness ``\Delta z`` for the vertical flux and the horizontal node spacing ``\Delta x`` for the horizontal flux:

```math
l_v = \min(l_\text{phys}, \, \Delta z),
\qquad
l_h = \min(l_\text{phys}, \, \Delta x).
```

This mirrors the Smagorinsky-Lilly horizontal/vertical split, in which the two length scales are ``c_s \Delta x`` and ``c_s \Delta z``. Two limits follow:

  - where ``l_\text{phys} < \min(\Delta x, \Delta z)`` (for example small near-surface
    eddies), ``l_h = l_v = l_\text{phys}`` and the diffusivity is isotropic;
  - where ``l_\text{phys} > \max(\Delta x, \Delta z)``, the ratio ``l_h / l_v = \Delta x / \Delta z`` recovers the Smagorinsky grid anisotropy.

## When to enable it

Horizontal EDMF diffusion is intended for box configurations whose horizontal grid spacing is comparable to or smaller than the physical mixing length, so that the horizontal limiter ``\min(l_\text{phys}, \Delta x)`` binds. This is the high-resolution and gray-zone regime, where the horizontal grid is fine enough for an anisotropic SGS length scale to matter.

At coarse horizontal resolution (for example global runs with ``\Delta x`` of tens of kilometers) the horizontal limiter rarely binds, the horizontal diffusivity reduces to the isotropic environment value, and the term is typically negligible next to resolved horizontal transport. The option is off by default for that reason.

The horizontal term is a grid-mean closure applied independently of the vertical flux and of the diffusion mode: it is always explicit, since the horizontal operators are not part of the column implicit solve, while the vertical flux remains implicit when `implicit_diffusion` is set. The Smagorinsky-Lilly and anisotropic-minimum-dissipation closures already supply horizontal SGS diffusion of the same fields, so combining either with this option is rejected at model construction.

Because the term is explicit, it adds a horizontal diffusive stability limit on the timestep, ``\Delta t \lesssim \Delta x^2 / (2 K_h)``. At the fine horizontal resolutions where this closure is intended, the timestep is in practice already set by the explicit horizontal acoustic limit ``\Delta t \lesssim \Delta x / c_s``, which is the more restrictive of the two.

## Configuration

```yaml
edmfx_sgs_horizontal_diffusive_flux: true   # default: false
```

## Diagnostics

Three diagnostic variables expose the horizontal closure fields:

  - `lmixh`: the mixing length with the grid-scale limit set by the horizontal node
    spacing, ``l_h`` [m];
  - `edth`: the horizontal eddy diffusivity for scalars, ``K_h`` [m² s⁻¹];
  - `evuh`: the horizontal eddy viscosity, ``K_u`` [m² s⁻¹].

The vertical counterparts are `lmix`, `edt`, and `evu`.
