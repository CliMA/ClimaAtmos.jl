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

As per [PROPHET: Prognostic Equations](edmf_equations.md), with prognostic EDMF, each updraft carries a tracer `<bin>` in `Y.c.sgsʲs.:(j)`, automatically wired through SGS tracer machinery: updraft tracers are transported by the mass flux, exchanged with the environment through entrainment and detrainment, and
contribute an SGS mass-flux term to the grid-mean equation.

Prognostic aerosol species add three processes,
`aerosol_emission_tendency!`, `aerosol_settling_tendency!`, and
`aerosol_deposition_tendency!`, dispatching off
`AbstractPrognosticAerosol` to compute:

  - **Surface emission** (called from
    `src/prognostic_equations/surface_flux.jl`): per-bin upward mass fluxes,
    written by ClimaCoupler once per coupling step via
    `set_sslt_surface_fluxes!`, are applied as bottom boundary conditions
    on `Y.c.ρ<bin>` (and updrafts) via `boundary_tendency_scalar`.
  - **Gravitational settling** (called from
    `src/prognostic_equations/remaining_tendency.jl`): explicit downward
    vertical advection at the bin's slip-corrected Stokes velocity, with free
    outflow at the surface.
  - **Deposition** (called from
    `src/prognostic_equations/remaining_tendency.jl`): sink tendencies
    applied with the other remaining tendencies; currently a residence-time
    decay placeholder that forthcoming branches turn into the accumulated
    dry and wet deposition sinks.

Prognostic sea salt requires `turbconv: prognostic_edmfx`: every size-dependent
process is evaluated per subdomain (environment and updrafts).
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

with Sutherland viscosity ``\mu(T)`` and Cunningham slip correction ``C_c``.
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
``v_g \cdot \rho\chi`` deposits at the surface. The turbulent part of dry
removal and wet removal are forthcoming.

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

Beyond the settled gravitational flux, sea salt removal is currently a
uniform residence-time decay:

```math
\frac{\partial \rho\chi_i}{\partial t} = -\frac{\rho\chi_i}{\tau},
```

with ``\tau = 0.55`` days (`ssa_residence`), the AeroCom phase III
ensemble-mean sea salt lifetime [Gliss2021](@cite). This uniform rate
over-deposits small bins and under-deposits large ones; forthcoming
branches replace it with the accumulated size-resolved dry and wet
deposition sinks.

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
