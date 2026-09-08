# Aerosols

ClimaAtmos treats each aerosol species in one of two ways:

  - **Prescribed** (`prescribed_aerosols` config option): bin mass mixing
    ratios are read from a MERRA-2 climatology and interpolated to the model
    grid and simulation time. The model does not transport or modify them.
  - **Prognostic** (`prognostic_aerosols` config option): bin mass mixing
    ratios are model state (`Y.c.ρ<bin>`), transported as a passive tracer
    and driven by emission and deposition tendencies.

The two treatments coexist: prognostic species are currently *passive*
(validated against the climatology), so radiation and aerosol-cloud
interaction read the prescribed bins for every species, including species
that also carry prognostic tracers.

Users that only need aerosol concentrations read `ᶜaerosol_bin_mmr` /
`ᶜaerosol_species_mmr`, which dispatch on the species' prognostic model: an
`AbstractPrognosticAerosol` reads the model state, `nothing` falls back to
the prescribed climatology (or zero if the bin is not prescribed either).

## Architecture

Each species slot of `AtmosAerosols` (`seasalt`, `dust`, `sulfate`,
`black_carbon`, `organic_carbon`) holds `nothing` (not prognostic) or an
`AbstractPrognosticAerosol`:

```
AbstractPrognosticAerosol
└── PrognosticSeaSalt{names}
```

The prescribed pathway is independent of these models: it is driven directly
by the MERRA-2 bin keys listed in `prescribed_aerosols` (see
`AEROSOL_SPECIES_BIN_NAMES` for the species-level grouping).

## Prescribed aerosols

All prescribed aerosol species share one pathway: at init,
`prescribed_aerosol_cache` (`src/cache/tracer_cache.jl`) builds one
`TimeVaryingInput` per bin from the MERRA-2 monthly climatology. In each radiation
update `update_prescribed_aerosol_concentrations!` evaluates the `TimeVaryingInput`
into `p.tracers.prescribed_aerosols_field`.

## Prognostic aerosols

A prognostic aerosol species stores each bin as a density-weighted grid-scale
tracer `Y.c.ρ<bin>`. Following the `ρχ` naming convention,
the [Passive Tracers](passive_tracers.md) machinery automatically applies
horizontal advection, vertical advection, vertical diffusion, and hyperdiffusion.

As per [PROPHET: Overview and Equations](prophet.md), with prognostic EDMF, each updraft carries a tracer `<bin>` in `Y.c.sgsʲs.:(j)`, automatically wired through SGS tracer machinery: updraft tracers are transported by the mass flux, exchanged with the environment through entrainment and detrainment, and
contribute an SGS mass-flux term to the grid-mean equation.

Prognostic aerosol species add three tendency verbs,
`aerosol_emission_tendency!`, `aerosol_settling_tendency!`, and
`aerosol_deposition_tendency!` (itself `aerosol_dry_deposition_tendency!`
plus `aerosol_wet_deposition_tendency!`), dispatching off
`AbstractPrognosticAerosol` to compute:

  - **Surface emission** (called from
    `src/prognostic_equations/remaining_tendency.jl`): per-bin upward mass fluxes,
    written by ClimaCoupler once per coupling step via
    `set_sslt_surface_fluxes!`, are applied as bottom boundary conditions
    on `Y.c.ρ<bin>` (and updrafts) via `boundary_tendency_scalar`.
  - **Gravitational settling** (called from
    `src/prognostic_equations/remaining_tendency.jl`): explicit downward
    vertical advection at the bin's slip-corrected Stokes velocity, with free
    outflow at the surface.
  - **Dry deposition** (called from
    `src/prognostic_equations/remaining_tendency.jl`): a surface sink at the
    Zhang et al. (2001) turbulent dry-deposition velocity with the Emerson
    et al. (2020) revised parameters, applied like the emission flux.
  - **Wet removal** (called from
    `src/prognostic_equations/remaining_tendency.jl`): first-order in-cloud
    (nucleation) scavenging plus below-cloud washout at the cached per-bin
    rate from `set_sslt_wet_deposition_rates!`, under 0M or 1M microphysics
    (see [Wet removal](@ref)).

Prognostic sea salt requires `turbconv: prognostic_edmfx`: settling is
evaluated per subdomain (environment and updrafts), and the surface fluxes
(emission, dry deposition) act on the grid mean and are mirrored onto the
updrafts.
`PrognosticSeaSalt` is a mass-only scheme: its `ρ<bin>` tracers carry dry
mass, and size-dependent processes (settling, deposition) act on each bin at
its mass-weighted settling radius.

## Prognostic sea salt

`PrognosticSeaSalt` tracks the five MERRA-2 size bins of
``(0.03, 0.1), (0.1, 0.5), (0.5, 1.5), (1.5, 5), (5, 10)`` μm.

### Surface emission

The sea salt source function is the whitecap parameterization of [Gong2003](@cite),
which extends the [Monahan1986](@cite) and is used in MERRA-2. These
whitecap methods decompose the number flux as:

```math
\frac{dF}{dr} = W(u_{10}) \cdot \frac{1}{\tau} \frac{dE}{dr},
```

where ``W = 3.84 \times 10^{-6} \, u_{10}^{3.41}`` is the fraction of the
sea surface covered by whitecaps, ``\tau`` is the whitecap decay time
(``\approx 3.53`` s), and ``dE/dr`` is the number of droplets produced per
unit whitecap area per radius increment over a whitecap's decay.
Gong (2003) empirically fits the latter arrive at the spectrum, in terms of
the nondimensionalized droplet radius at 80% relative humidity
``\hat r_{80}``:

```math
\frac{dF}{d\hat r_{80}} = 1.373 \, u_{10}^{3.41} \,
\hat r_{80}^{-A} \left(1 + 0.057 \, \hat r_{80}^{3.45}\right)
\times 10^{\,1.607 \, e^{-B^2}},
```

```math
A = 4.7 \, (1 + \Theta \hat r_{80})^{-0.017 \, \hat r_{80}^{-1.44}}, \qquad
B = \frac{0.433 - \log_{10} \hat r_{80}}{0.433}, \qquad
\Theta = 30
```

Because MERRA-2 size bins are defined by dry radius, ClimaAtmos works in
the dimensionless dry radius ``\hat r = r_\mathrm{dry} / r_\mathrm{ref}``,
converting the spectrum with ``\hat r_{80} = \chi \, \hat r``, with
``\chi`` = `ssa_r80_per_dry` = 2 as per Lewis & Schwartz (2004).

#### 3-mode lognormal fit

ClimaAtmos uses a 3-mode lognormal fit to approximate the Gong parameterization:

```math
\frac{dF}{d\hat r} = \sum_{i=1}^{3} \frac{F_i}{\hat r}
\exp\!\left[-\frac{\ln^2(\hat r / r_i)}{2 \ln^2 \sigma_{g,i}}\right],
```

where ``F_i`` is mode i's peak ``dF/d\ln\hat r`` amplitude [m⁻² s⁻¹] at
``u_{10} = 1`` m/s, ``r_i`` its modal dry radius in units of
``r_\mathrm{ref}``, and ``\sigma_{g,i}`` its geometric standard deviation.

| mode | ``F_i`` [m⁻² s⁻¹] | ``r_i`` | ``\sigma_{g,i}`` |
|:----:|:-----------------:|:-------:|:----------------:|
| 1    | 0.2157            | 0.05545 | 17.02            |
| 2    | 60.93             | 0.0914  | 1.813            |
| 3    | 5.949             | 0.776   | 1.759            |

```@eval
# SeaSaltFitSkill is loaded into Main by docs/make.jl (via
# docs/src/sea_salt_emission_fit.jl) from its home in
# test/parameterized_tendencies/aerosols/, where the unit tests guard it.
Main.SeaSaltFitSkill.markdown()
```

This fit is physically interpretable as mode 1 allowing a non-zero tail of spume droplets, mode 2 capturing bubble-burst film drops, and mode 3 bubble-burst jet drops.

![3-mode lognormal fit of the Gong (2003) spectrum](assets/gong_ln3_modes.png)

#### Per-bin flux scales

ClimaAtmos tracers carry mass, so the number flux spectrum must be
converted to mass flux (the third moment of the number spectrum):

```math
\frac{dF_m}{d\hat r} = m(\hat r) \, \frac{dF}{d\hat r}
= \frac{4\pi}{3} \rho_\mathrm{dry} \, (\hat r \, r_\mathrm{ref})^3
\, \frac{dF}{d\hat r},
```

with ``\rho_\mathrm{dry}`` the dry sea salt density.
Per-bin emission scales are these spectra
integrated over the bin edges, precomputed offline and stored in ClimaParams:

```math
k_i^{(0)} = \int_{\hat r_i}^{\hat r_{i+1}} \frac{dF}{d\hat r} \, d\hat r,
\qquad
k_i^{(3)} = \int_{\hat r_i}^{\hat r_{i+1}} \frac{dF_m}{d\hat r} \, d\hat r,
```

giving the number (`ssa_gong_logfit_bin_0M_flux` [m⁻² s⁻¹]) and dry-mass
(`ssa_gong_logfit_bin_3M_flux` [kg m⁻² s⁻¹]) flux scales at
``u_\mathrm{10, ref} = 1`` m/s. The per-bin upward mass flux is

```math
\mathcal{F}_i = k_i^{(3)}
\left(\frac{u_{10}}{u_\mathrm{ref}}\right)^{3.41}
```

### Hygroscopic growth

Sea salt particles deliquesce, so their transport sizes depend on the ambient
relative humidity. Consumers (settling, dry deposition) evaluate the per-bin
growth factor ``\xi = r_\mathrm{wet}/r_\mathrm{dry}`` in-kernel with
`sslt_growth_factor` from the relative humidity of the subdomain state they
act on, so nothing is cached. The growth factor is the size-dependent fit of
Lewis (2008, Eq. 34)

```math
\xi = a \left( b + \frac{1}{1 - \mathrm{RH} + (\xi_{\sigma,0}/a)^{3/2}} \right)^{1/3},
\qquad \xi_{\sigma,0} = \frac{2\, \sigma_w}{\rho_w\, R_v\, T\, r_\mathrm{dry}},
```

with the NaCl coefficients ``a = 1.08``, ``b = 1.10`` (`ssa_lewis_a`,
`ssa_lewis_b`). The Kelvin (curvature) term ``\xi_{\sigma,0}`` lowers the
effective humidity, more so for smaller dry particles, so ξ stays finite at
RH = 1 without a cap. Its temperature-independent part
``C_i = (2\sigma_w/(\rho_w R_v a\, r_i))^{3/2}`` is precomputed per bin at the
bin's settling radius (`sslt_kelvin_coefficient`), and the kernel evaluates
``(\xi_{\sigma,0}/a)^{3/2} = C_i\, T^{-3/2}``. Below the efflorescence RH
(`ssa_rh_efflorescence`, 0.45 for NaCl) the particles are dry, ``\xi = 1``,
without hysteresis. The κ-Köhler form ``(1 + \kappa\, a_w/(1 - a_w))^{1/3}``
and the bulk Lewis fit ``a\,(b + 1/(1 - a_w))^{1/3}`` (Eq. 33), both with
``a_w = \min(\mathrm{RH}, \mathrm{RH}_\mathrm{cap})`` (`ssa_rh_cap`), are
available as `sslt_kappa_kohler_growth_factor` and `sslt_lewis33_growth_factor`
but are not used by the tendencies.

Under `PrognosticEDMFX` the grid-mean RH blends saturated updraft air into the
drier environment, and ξ is steepest exactly where that blending happens, so
consumers evaluate their velocities or rates on the environment and updraft
states separately and combine those into grid-mean fluxes the way
`set_precipitation_velocities!` combines subdomain sedimentation velocities; ξ
itself is never area-averaged, because every size-to-flux map is convex in ξ.

### Gravitational settling

Each bin tracer is advected downward at the slip-corrected Stokes terminal
velocity of its mass-weighted settling radius:

```math
v_g = \frac{2}{9} \frac{(\rho_\mathrm{wet} - \rho_\mathrm{air})\, g\,
r_\mathrm{wet}^2\, C_c(\mathrm{Kn})}{\mu(T)},
```

with the air viscosity ``\mu(T)`` and mean free path of Seinfeld & Pandis (2006,
Eqs. 9.6–9.7) and the Cunningham slip correction ``C_c`` of Zhang et al. (2001, Eq. 3).
The working radius is the bin's wet settling radius
``\xi \cdot \sqrt{\langle r^5\rangle/\langle r^3\rangle}``, whose Stokes
speed carries the bin's mass settling flux. The sub-bin weights come from the
same lognormal fit of the Gong spectrum that sets the emission scales: the
per-bin radius moments ``\hat M_k``, ``k = 0,\dots,6``, of the fitted
spectrum are evaluated in closed form (erf of the shifted lognormal) once at
cache construction (`sslt_bin_moments`, stored in `p.tracers`), and the
settling radii are read off them (`sslt_settling_radii`). Settling is explicit with a
per-cell Courant cap (`ssa_settling_courant_max`), using the
`ᶠright_bias`/`ᶜprecipdivᵥ` free-outflow stencil, so the gravitational flux
``v_g \cdot \rho\chi`` deposits at the surface.

Under `PrognosticEDMFX` settling follows the subdomain treatment of the
microphysics species (`set_precipitation_velocities!` and the updraft
sedimentation in `edmfx_sgs_vertical_advection_tendency!`). The environment
velocity ``w^0`` is evaluated on the environment state and each updraft
velocity ``w^j`` on its draft state. The grid-mean tracer settles at the
mass-weighted velocity

```math
w = \frac{\rho a^0 \chi^0 w^0 + \sum_j \rho a^j \chi^j w^j}
         {\rho a^0 \chi^0 + \sum_j \rho a^j \chi^j},
```

so the grid-scale settling flux equals the sum of the subdomain fluxes, and
each updraft tracer receives its within-updraft flux convergence with the
lateral-detrainment correction (`updraft_sedimentation!`), with the
environment flux density ``\rho^0 w^0 \chi^0`` supplied directly rather than
reconstructed from the grid mean.

### Deposition

The turbulent part of dry removal is a surface-flux sink at the
[Zhang2001](@cite) deposition velocity, with the revised parameters and
functional forms of [Emerson2020](@cite),

```math
V_{d,\mathrm{turb}} = \frac{1}{R_a + R_s},
\qquad
R_s = \frac{1}{\varepsilon_0\, u_\star\,
(E_B + E_\mathrm{IM})\, R_1},
```

with the MOST aerodynamic resistance ``R_a = F_m/(\kappa u_\star)``, evaluated
with the momentum profile and momentum roughness that the emission wind uses
rather than Zhang's scalar pair, Brownian collection
``E_B = C_B\,\mathrm{Sc}^{-\gamma}``, impaction
``E_\mathrm{IM} = C_\mathrm{Im}(\mathrm{St}/(\alpha + \mathrm{St}))^\beta``,
and rebound ``R_1 = e^{-\sqrt{\mathrm{St}}}`` at the smooth-surface Stokes
number ``\mathrm{St} = \tau u_\star^2/\nu`` with the particle relaxation time
``\tau = V_g/g``, using the water/ocean land-use
category everywhere for now. The revised values ``C_B = 0.2``,
``\gamma = 2/3``, ``C_\mathrm{Im} = 0.4`` and ``\beta = 1.7`` (the
`emerson_*` parameters; the Zhang values remain in ClimaParams, deprecated,
and their forms are kept commented out in `sslt_dry_deposition_velocity` for
side-by-side comparison runs)
lower the deposition velocity of accumulation-mode particles by roughly an
order of magnitude, which is what the measurements of [Emerson2020](@cite)
support. Their revised interception term
``E_\mathrm{IN} = C_\mathrm{In}(d_p/A)^\upsilon`` is not used over water,
which has no characteristic collector radius ``A``. The gravitational contribution is already deposited by
the settling boundary, so the two sum to the full deposition velocity without
double counting. ``V_{d,\mathrm{turb}}`` is Courant-capped with the same
`ssa_settling_courant_max` as settling so the explicit sink cannot
over-deplete the lowest cell in one step (a numerical device; the settling
speed inside the deposition Stokes number is uncapped).

The velocity is evaluated on the grid-mean lowest-level state and the flux
``-V_{d,\mathrm{turb}}\, \rho\chi|_1`` is cached per bin in
`p.tracers.sslt_drydep_fluxes`. It enters
the tracers exactly as the emission flux does: as the bottom boundary
condition of the grid-mean tracer, with the specific tendency mirrored onto
each updraft tracer so subdomain and grid-mean concentrations do not drift
apart at the surface.

### Wet removal

With 1-moment microphysics each bin is removed at the first-order rate

```math
k = \underbrace{F\, f_\mathrm{act}\, \min\!\left(\frac{Q}{q_\mathrm{lcl} + q_\mathrm{icl}}, \frac{1}{\Delta t}\right)}_{\text{in-cloud}}
  + \underbrace{(1 - F\, f_\mathrm{act})\, E_\mathrm{bin}\, \Lambda(q_\mathrm{rai}, \rho)}_{\text{below-cloud}},
```

the sum of in-cloud (nucleation) scavenging (`sslt_in_cloud_scavenging_rate`)
— within the cloudy area ``F`` the activated fraction ``f_\mathrm{act}`` (one
for every sea salt bin) is dissolved in cloud condensate and removed at the
intensive rate at which that condensate, of either phase, converts to
precipitation; ``Q`` is the process-level sum of every sink of
``q_\mathrm{lcl}`` and ``q_\mathrm{icl}`` that lands in rain or snow
(`cloud_precip_formation_rate`), so a glaciated subdomain is scavenged at the
rate its ice converts to snow — and below-cloud washout
(`sslt_below_cloud_scavenging_rate`) at the swept-volume collection rate
``\Lambda`` of the Marshall–Palmer rain population
(`rain_swept_collection_rate`, the same closed form as the accretion kernel)
times the per-bin collection efficiency (`ssa_collection_efficiency`).
Washout carries the weight ``1 - F f_\mathrm{act}``, the aerosol mass *not*
dissolved in condensate: the cloud-free area plus the interstitial fraction of
the cloudy area. The droplet-borne remainder is already removed by the
in-cloud term, whose driver ``Q`` contains the rain-accretes-cloud-liquid arm,
so weighting washout by the full area would count that channel twice. The sink is applied in the unconditionally
stable form ``\partial_t(\rho\chi) = \rho\chi\,(e^{-k\Delta t} - 1)/\Delta t``.

Under `PrognosticEDMFX` every ingredient is evaluated per subdomain. The
environment uses the residual water species, its own ``Q^0`` and
``\Lambda^0``, and the environment cloud fraction recovered from the
area-weighted grid-mean value, ``F^0 = (F - \sum_j a^j\,\mathbb{1}[\text{condensate}^j]) / a^0`` (which is why
`prognostic_aerosols` requires `cloud_model: quadrature` or `MLCloud`;
`grid_scale` does not produce the area-weighted form);
each updraft uses its own water species, a binary cloud indicator, ``Q^j``
and ``\Lambda^j``. The grid-mean tracer is removed at the rate whose *survival*
fraction is the mass-weighted mean of the subdomain survivals,

```math
e^{-k \Delta t} = \frac{\rho a^0 \chi^0 e^{-k^0 \Delta t}
                        + \sum_j \rho a^j \chi^j e^{-k^j \Delta t}}
                       {\rho a^0 \chi^0 + \sum_j \rho a^j \chi^j},
```

so the grid-scale sink is exactly the sum of the subdomain sinks for any
``k\Delta t`` — weighting the rate instead would make the grid mean
over-remove, since ``k \mapsto 1 - e^{-k\Delta t}`` is concave, and the
residual environment would absorb the difference. The opt-in
`wetss` diagnostic (`src/diagnostics/local_diagnostics.jl`), which reads it,
closes the tracer budget, while each updraft
tracer is scavenged at its own rate (`p.tracers.sslt_wetdep_ratesʲs`), so
detrainment returns depleted rather than pristine updraft air.

With 0-moment microphysics the same two-term rate is assembled from the
0-moment ingredients. The in-cloud driver is the total-water sink itself,
``Q = \max(0, -\partial_t q_\mathrm{tot})`` (`precipitation_conversion_rate_0m`),
drained from the equilibrium condensate ``q_\mathrm{liq} + q_\mathrm{ice}``:
the scheme resolves neither phase nor process, so the activated aerosol share
follows whatever condensate converts to precipitation. There is no rain
state, since precipitation leaves the column the instant it forms, so the
washout follows the *precipitation shadow*, the downward rain flux through
each cell recovered from the cached sink (`set_sslt_precipitation_shadow!`),

```math
P(z) = \int_z^{z_\mathrm{top}} \max\!\left(0, -\rho\,\partial_t q_\mathrm{tot}\right)\big|_{T \ge T_\mathrm{freeze}}\, dz',
```

at the empirical power law of Feng (2007),
``\Lambda_\mathrm{bin} = a_\mathrm{bin}\, R^{b_\mathrm{bin}}`` with ``R`` the
rain rate in mm h⁻¹ (`power_law_washout_rate`; `ssa_washout_prefactor`,
`ssa_washout_exponent`, exponents 0.6–0.8 across the marine size range):

```math
k = F\, \min\!\left(\frac{Q}{q_\mathrm{liq} + q_\mathrm{ice}}, \frac{1}{\Delta t}\right)
  + (1 - F)\, a_\mathrm{bin}\, R^{b_\mathrm{bin}}.
```

Under `PrognosticEDMFX` the in-cloud term and the cloudy area are per
subdomain exactly as for 1M, from each subdomain's own 0-moment sink and
condensate, while the shadow is a column quantity, so every subdomain is
washed at the same rain rate.

## Adding a prognostic aerosol species

`PrognosticSeaSalt` is the template. A new species plugs into the same
dispatch points (all dispatching off the species model type, so nothing
else needs to change):

 1. **Model type** (`src/types.jl`): define
    `struct PrognosticMySpecies{names} <: AbstractPrognosticAerosol end`
    with the bin-name tuple as the type parameter, and implement
    `bin_names(::Type{PrognosticMySpecies{names}})`. The bin names double
    as the MERRA-2 variable names and, `ρ`-prefixed, as the prognostic
    state names.
 2. **Registration** (`src/types.jl`): replace the `nothing` in the
    species' `AEROSOL_SPECIES` entry with a constructor closure
    `ap -> PrognosticMySpecies(...)`, where `ap` is the
    `prognostic_aerosol_params` bundle. This makes the species key valid
    in the `prognostic_aerosols` config.
 3. **Parameters** (`src/parameters/create_parameters.jl`): add the
    species' physical constants to the `prognostic_aerosol_params`
    bundle (backed by ClimaParams; do not hard-code them).
 4. **Cache** (`src/cache/tracer_cache.jl`): implement
    `species_aerosol_cache(Y, params, ::PrognosticMySpecies)` returning
    the fields the tendencies need (there is no generic fallback, so this
    method is required).
 5. **Tendencies** (new file under `src/parameterized_tendencies/aerosols/`,
    included from `aerosols.jl`): implement the per-species methods of the
    process hooks dispatched in `aerosols.jl` (e.g.
    `aerosol_emission_tendency!(Yₜ, Y, p, t, ::PrognosticMySpecies)`).
    Species without a process can define the method as a no-op.

State variables (`Y.c.ρ<bin>` plus the EDMF updraft tracers) and the
passive-tracer transport come for free from `aerosol_variables` and the
`ρχ` naming convention, and `ᶜaerosol_bin_mmr` automatically reads the
prognostic state for any `AbstractPrognosticAerosol`.
