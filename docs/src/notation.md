# Notation and Symbols

The [governing equations](equations.md) and the
[PROPHET equations](prophet.md) use the notation of the papers they
follow ([Yatunin2026](@cite) for the dynamical core). The source code uses
ASCII- and Unicode-flavored names that encode where a field is defined and which
subdomain it belongs to. This page is the bridge between the two.

## Staggered-grid prefixes

The vertical grid is staggered: some quantities live at cell **centers**, others
at cell **faces** (the interfaces between centers). The leading character of a
field or operator name says which.

| Prefix | Meaning                        | Example                              |
|:------ |:------------------------------ |:------------------------------------ |
| `ᶜ`    | Cell center                    | `ᶜρ`, `ᶜp`, `ᶜK`, `ᶜinterp`, `ᶜdivᵥ` |
| `ᶠ`    | Cell face (vertical interface) | `ᶠu₃`, `ᶠu³`, `ᶠinterp`, `ᶠgradᵥ`    |

The state vector mirrors this split: `Y.c` holds center variables and `Y.f`
holds face variables, and the same convention applies to the cache `p`.

For operators the prefix names the *output* location, so `ᶜinterp` interpolates
faces to centers and `ᶠinterp` interpolates centers to faces. A `ᵥ` subscript
marks a vertical (finite-difference) operator and an `ₕ` subscript a horizontal
(spectral-element) one: `ᶜdivᵥ` versus `divₕ`. A leading `w` on a horizontal
operator marks the weak form, as in `wdivₕ` and `wcurlₕ`.

## Subdomain superscripts

PROPHET partitions each grid cell into subdomains: the grid mean, one or more
drafts (updrafts), and the environment.

| Superscript | Meaning                        | Example                      |
|:----------- |:------------------------------ |:---------------------------- |
| (none)      | Grid mean                      | `ᶜρ`, `ᶜu`                   |
| `ʲ`         | A single draft (subdomain `j`) | `ᶜρaʲ`, `ᶠu₃ʲ`, `ᶜmseʲ`      |
| `ʲs`        | The tuple over all drafts      | `Y.c.sgsʲs`, `ᶜρaʲs`, `ᶜuʲs` |
| `⁰`         | The environment                | `ᶜmse⁰`, `ᶜq_tot⁰`, `ᶠu₃⁰`   |

The environment is not stored: it is recovered as the residual of the grid mean
minus the drafts.

The [PROPHET pages](prophet.md) index a general subdomain by ``m = 0, \dots, M``,
following the paper, and a draft (``m \ge 1``) by ``j``, following the code. So
``\sum_m`` runs over the drafts *and* the environment, while ``j`` is the loop
index in `for j in 1:n`.

## Vector components

Vectors are stored in the covariant or contravariant bases of the reference
element rather than as physical components. Subscripts denote covariant
components and superscripts contravariant ones.

| Symbol                               | Meaning                                             |
|:------------------------------------ |:--------------------------------------------------- |
| `uₕ`                                 | Horizontal covariant velocity (`Covariant12Vector`) |
| `u₃`                                 | Vertical covariant velocity (`Covariant3Vector`)    |
| `u³`                                 | Third contravariant velocity component              |
| `C1`, `C2`, `C12`, `C3`, `C123`      | Covariant vector constructors                       |
| `CT1`, `CT2`, `CT12`, `CT3`, `CT123` | Contravariant vector constructors                   |
| `UVec`, `VVec`, `WVec`, `UV`, `UVW`  | Physical (local orthonormal) vectors                |

Contravariant velocity components have units of [1/s], not [m/s]: they are
velocities per unit reference-element length.

## Prognostic variables

| Paper symbol         | Code name | Location | Description                                |
|:-------------------- |:--------- |:-------- |:------------------------------------------ |
| ``\rho``             | `ρ`       | center   | Moist air density [kg/m³].                 |
| ``\boldsymbol{u}_h`` | `uₕ`      | center   | Horizontal covariant velocity.             |
| ``u_3``              | `u₃`      | face     | Vertical covariant velocity.               |
| ``\rho e_{tot}``     | `ρe_tot`  | center   | Total energy density [J/m³].               |
| ``\rho q_t``         | `ρq_tot`  | center   | Total water content [kg/m³].               |
| ``\rho q_l^{cl}``    | `ρq_lcl`  | center   | Cloud liquid water content [kg/m³].        |
| ``\rho q_i^{cl}``    | `ρq_icl`  | center   | Cloud ice content [kg/m³].                 |
| ``\rho q_r``         | `ρq_rai`  | center   | Rain water content [kg/m³].                |
| ``\rho q_s``         | `ρq_sno`  | center   | Snow content [kg/m³].                      |
| ``\rho n_l``         | `ρn_lcl`  | center   | Cloud droplet number concentration [1/m³]. |
| ``\rho n_r``         | `ρn_rai`  | center   | Rain drop number concentration [1/m³].     |
| ``\rho e_{tke}``     | `ρtke`    | center   | Turbulence kinetic energy density [J/m³].  |

Which of these exist depends on the configuration: the moisture and
precipitation variables are added by the microphysics model, and `ρtke` by the
turbulence-convection model. Note the code writes cloud condensate as `lcl`/
`icl` (liquid-cloud, ice-cloud) to distinguish it from precipitating species.

Draft variables are stored in `Y.c.sgsʲs.:(j)` and are **specific** (not
density-weighted), except for the effective density itself:

| Paper symbol     | Code name | Description                                                 |
|:---------------- |:--------- |:----------------------------------------------------------- |
| ``\hat{\rho}^j`` | `ρa`      | Effective density ``\rho^j a^j`` [kg/m³].                   |
| ``h_s^j``        | `mse`     | Specific moist static energy [J/kg].                        |
| ``q_t^j``        | `q_tot`   | Specific total water [kg/kg].                               |
| ``u_3^j``        | `u₃`      | Vertical covariant velocity of the draft (a face variable). |

With a slab-ocean surface, the prognostic surface state is `Y.sfc`.

## Derived and cached quantities

| Paper symbol              | Code name      | Description                                                       |
|:------------------------- |:-------------- |:----------------------------------------------------------------- |
| ``p``                     | `ᶜp`           | Air pressure [Pa].                                                |
| ``K``                     | `ᶜK`           | Specific kinetic energy [J/kg].                                   |
| ``\Phi``                  | `ᶜΦ`           | Geopotential [m²/s²].                                             |
| ``\Pi``                   | `ᶜΠ`           | Exner function [-].                                               |
| ``\tilde{\boldsymbol u}`` | `ᶠu³`          | Mass-weighted face velocity (contravariant).                      |
| ``\bar{\boldsymbol u}``   | `ᶜu`           | Cell-center reconstruction of the velocity.                       |
| ``\rho^j``                | `ᶜρʲs`         | Draft air densities in kg/m³; the area fraction is `a^j = ρa/ρʲ`. |
| ``T``                     | `ᶜT`           | Air temperature [K].                                              |
| ``h_{tot}``               | `ᶜh_tot`       | Total specific enthalpy [J/kg].                                   |
| ``K_h``, ``K_u``          | `ᶠK_h`, `ᶠK_u` | Eddy diffusivity and viscosity [m²/s].                            |
| ``\ell``                  | `ᶜl_mix`       | Mixing length [m].                                                |

Tendencies carry a `ₜ` subscript: `Yₜ` is the tendency of the state, and
`Yₜ.c.ρe_tot` is the tendency of total energy.

## Common suffixes and shorthands

| Suffix / name       | Meaning                                                                                     |
|:------------------- |:------------------------------------------------------------------------------------------- |
| `_nonneg`           | A copy clipped at zero, e.g. `ᶜq_tot_nonneg`.                                               |
| `ʲs` / `⁰`          | Cache fields carry the same subdomain suffixes as the state, e.g. `ᶜTʲs`, `ᶜT⁰`, `ᶜq_liq⁰`. |
| `ᶜspecific`         | Conversion from a density-weighted variable to a specific one.                              |
| `FT`                | The working float type (`Float32` or `Float64`).                                            |
| `Y`, `Yₜ`, `p`, `t` | State, tendency, cache, and time; see the [Glossary](glossary.md).                          |

The operator shorthands (`ᶜinterp`, `ᶠwinterp`, `ᶜadvdivᵥ`, `ᶠupwind1`, …) are
defined and documented in `src/utils/abbreviations.jl`; the
[governing equations](equations.md) page maps the mathematical operators onto
their ClimaCore implementations.
