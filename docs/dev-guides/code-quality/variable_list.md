# CliMA Variable List

This document unifies the naming conventions used across the CliMA codebase. It defines 'reserved' variable names in `<property>_<species>` format with the default working fluid (no-subscript) being moist air.

## Type parameters

The Julia code typically uses `T` as a type parameter, however this conflicts with the typical usage for temperature. Instead, good choices are:

- `FT` for floating point values

## Names reserved for debug variables

- `dummy`
- `scratch`

## Working Fluid and Equation of State

- `q_dry` = dry air mass fraction
- `q_vap` = specific humidity, vapor
- `q_liq` = specific humidity, liquid
- `q_ice` = specific humidity, ice
- `q_tot` = specific humidity, total

- `ρ` = density (no subscript = moist air)
- `R_m` = gas constant, moist
- `R_d` = gas constant, dry
- `R_v` = gas constant, water vapor
- `T` = temperature, moist air

## Time

- `dt` = time increment

## Momentum

- `u` = x-velocity
- `v` = y-velocity
- `w` = z-velocity

## Energy balance

Lowercase `e_<type>` indicates a *specific* (per-unit-mass) quantity. The corresponding density-weighted volumetric forms used as prognostic variables are `ρe_<type>` (see "Prognostic variable conventions" below).

- `e_kin_<spe>` = specific energy per unit mass, kinetic
- `e_pot_<spe>` = specific energy per unit mass, potential
- `e_int_<spe>` = specific energy per unit mass, internal
- `e_tot_<spe>` = specific energy per unit mass, total

- `cv_m`, `cv_d`, `cv_l`, `cv_v`, `cv_i` = isochoric specific heat capacities [J/(kg·K)] (moist, dry, liquid, vapor, ice)
- `cp_m`, `cp_d`, `cp_l`, `cp_v`, `cp_i` = isobaric specific heat capacities [J/(kg·K)] (moist, dry, liquid, vapor, ice)

## Microphysics

Specific humidities of precipitation and cloud-condensate species:

- `q_rai` = specific humidity, rain [kg/kg]
- `q_sno` = specific humidity, snow [kg/kg]
- `q_lcl` = specific humidity, cloud liquid [kg/kg]
- `q_icl` = specific humidity, cloud ice [kg/kg]

By convention, when all partitions of the phase of water are included, we use
- `q_liq` = specific humidity, all liquid 
- `q_ice` = specific humidity, all ice

Terminal velocities are per-species:

- `terminal_velocity_<spe>` = mass-weighted terminal fall speed of `<spe>` [m/s] — e.g. `terminal_velocity_rai`, `terminal_velocity_sno`

Microphysical tendencies [kg/kg/s]. Sign convention: positive means a *source* for the species in the *to*-position of the name.

- `conv_q_lcl_to_q_rai` = autoconversion: cloud liquid → rain
- `conv_q_icl_to_q_sno` = ice autoconversion: cloud ice → snow
- `conv_q_vap_to_q_lcl_icl` = condensation / deposition: vapor → cloud condensate (signed; negative values represent evaporation / sublimation back to vapor)
- `evaporation_sublimation` = rain evaporation / snow sublimation; positive = vapor source
- `accretion` = collection of cloud condensate by precipitation; positive = precipitation source

## Thermodynamic state and pressure

- `p` = pressure (no subscript = total pressure of moist air) [Pa]
- `θ` = potential temperature [K]
- `θ_liq_ice` = liquid–ice potential temperature [K]
- `Φ` = geopotential [m²/s²]
- `grav` = gravitational acceleration [m/s²]
- `L_v` = latent heat of vaporization [J/kg]
- `L_s` = latent heat of sublimation [J/kg]
- `L_f` = latent heat of fusion [J/kg]

Note: in CliMA tendency-style signatures `f!(Yₜ, Y, p, t, …)`, the local name `p` refers to the cache, not pressure.

## Prognostic variable conventions

CliMA models typically integrate density-weighted forms as prognostic variables and diagnose specific quantities from them inside tendencies:

- `ρq_<spe>` = density × specific humidity of species `<spe>` (e.g. `ρq_tot`, `ρq_liq`, `ρq_rai`) [kg/m³]
- `ρe_tot` = density × total specific energy [J/m³]

## Field Prefixes (ClimaCore-based repos)

In repos that use ClimaCore for spatial discretization, functions that *return fields* are prefixed by their staggered-grid location:

- `ᶜ<name>` (typed `\^c<TAB>`) = field at cell *centers*
- `ᶠ<name>` (typed `\^f<TAB>`) = field at cell *faces*

For example, `ᶜρ(Y, p)` is a cell-center density field. The prefix lives on the *function*, not on the stored field: state-vector fields like `Y.c.ρ` are themselves cell-centered but do not carry the prefix.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
