# Aerosols

ClimaAtmos treats each aerosol species in one of two ways:

  - **Prescribed** (`prescribed_aerosols` config option): bin mass mixing
    ratios are read from a MERRA-2 climatology and interpolated to the model
    grid and simulation time. The model does not transport or modify them.
  - **Prognostic** (`prognostic_aerosols` config option): bin mass mixing
    ratios are model state (`Y.c.ρ<bin>`), transported as a passive tracer
    and driven by emission and deposition tendencies.

Users that only need aerosol concentrations are agnostic to the treatment: `ᶜaerosol_bin_mmr` and `ᶜaerosol_species_mmr` returns mass mixing ratio from whichever source the aerosol species uses.

## Architecture

Each species slot of `AtmosAerosols` (`seasalt`, `dust`, `sulfate`,
`black_carbon`, `organic_carbon`) holds `nothing` an `AbstractPrescribedAerosol` or an `AbstractPrognosticAerosol`:

```
AbstractAerosolModel
├── AbstractPrescribedAerosol
│   ├── PrescribedSeaSalt        (:SSLT01 … :SSLT05)
│   ├── PrescribedDust           (:DST01 … :DST05)
│   ├── PrescribedSulfate        (:SO4)
│   ├── PrescribedBlackCarbon    (:CB1, :CB2)
│   └── PrescribedOrganicCarbon  (:OC1, :OC2)
└── AbstractPrognosticAerosol
    └── PrognosticSeaSalt{names}
```

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

Prognostic aerosol species add two processes,
`aerosol_emission_tendency!` and `aerosol_deposition_tendency!`
dispatching off `AbstractPrognosticAerosol` to compute:

  - **Surface emission** (called from
    `src/prognostic_equations/surface_flux.jl`): per-bin upward mass fluxes
    computed during the precomputed-quantities update are applied as
    bottom boundary conditions on `Y.c.ρ<bin>` (and updrafts) via `boundary_tendency_scalar`.
  - **Deposition** (called from
    `src/prognostic_equations/remaining_tendency.jl`): sink tendencies
    applied with the other remaining tendencies.

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
include("sea_salt_fit_skill.jl")
SeaSaltFitSkill.markdown()
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
integrated over the bin edges — precomputed offline in
[`docs/src/sea_salt_emission_fit.jl`](sea_salt_emission_fit.jl)

```math
k_i^{(0)} = \int_{\hat r_i}^{\hat r_{i+1}} \frac{dF}{d\hat r} \, d\hat r,
\qquad
k_i^{(3)} = \int_{\hat r_i}^{\hat r_{i+1}} \frac{dF_m}{d\hat r} \, d\hat r,
```

giving the number (`ssa_gong_logfit_bin_0M_flux` [m⁻² s⁻¹]) and dry-mass
(`ssa_gong_logfit_bin_3M_flux` [kg m⁻² s⁻¹]) flux scales at
``u_\mathrm{10, ref} = 1`` m/s. At runtime, ``u_{10}`` is reconstructed
via `set_wind_at_height!` and `set_sea_salt_surface_fluxes!` scales per-bin 
upward mass flux by ``u_{10}``:

```math
\mathcal{F}_i = k_i^{(3)}
\left(\frac{u_{10}}{u_\mathrm{ref}}\right)^{3.41}
\times \text{(ocean fraction)}.
```

### Deposition

Sea salt aerosol removal is currently a uniform residence-time decay:

```math
\frac{\partial \rho\chi_i}{\partial t} = -\frac{\rho\chi_i}{\tau},
```

with ``\tau = 0.55`` days (`ssa_residence`), the AeroCom phase III
ensemble-mean sea salt lifetime [Gliss2021](@cite). This uniform rate
over-deposits small bins and under-deposits large ones; size-resolved wet
and dry deposition are forthcoming.
