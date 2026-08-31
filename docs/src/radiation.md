# Radiation

Radiation enters the [governing equations](equations.md) as one term: the
divergence of the net radiative flux in the total-energy equation,

```math
\frac{\partial}{\partial t} (\rho e_{tot}) \supset
  - \nabla \cdot (\rho \boldsymbol{\mathcal{F}}_R) .
```

Everything on this page is about how ``\boldsymbol{\mathcal{F}}_R`` is obtained.
There are three routes: a full correlated-``k`` radiative transfer solve with
[RRTMGP.jl](https://clima.github.io/RRTMGP.jl/stable/), a set of idealized flux
profiles for single-column cases, and the Held–Suarez temperature relaxation,
which replaces radiation altogether in the dry dynamical-core benchmark. The
`rad` configuration key chooses among them.

The radiative transfer itself, and the derivations behind the
correlated-``k`` method, belong to RRTMGP.jl and to
[Pincus2019](@cite). This page states what ClimaAtmos hands to the solver, how
often, and what it does with the answer. For the configuration surface, see
[Running with Radiation](radiation_howto.md).

## The radiation callback

Radiative transfer is far too slow to run every timestep, and radiative heating
rates change slowly, so the solve is a callback rather than a tendency. Every
`dt_rad` (6 hours by default), `rrtmgp_solver_callback!`

 1. copies the current column state into the solver's input arrays
    (`update_atmospheric_state!`),
 2. refreshes the insolation (`set_insolation_variables!`) and the surface albedo
    (`set_surface_albedo!`),
 3. calls `RRTMGP.update_fluxes!`, and
 4. copies the net upward flux into `p.radiation.ᶠradiation_flux`, a face field
    [W m⁻²].

Between callbacks, `ᶠradiation_flux` is held fixed. The tendency
`radiation_tendency!` takes its divergence at every stage, so the heating rate a
layer receives is constant over the `dt_rad` interval but the state it acts on
is not. A shorter `dt_rad` resolves the diurnal cycle better and calls the
solver more often; see
[Choosing the cadence](radiation_howto.md#Choosing-the-cadence).

Radiation is always explicit. It never enters the implicit solve, and the flux
divergence is applied with `ᶜdivᵥ`, so flux convergence heats the layer.

!!! note "TODO: not yet implemented"

    With PROPHET, the same grid-mean heating rate is applied to every draft
    (divided by the draft density). The formulation instead partitions the
    heating rate between subdomains by their cloud fraction; see
    [PROPHET: Overview and Equations](prophet.md#Radiation) for the form and for
    what is missing.

## What RRTMGP is given

RRTMGP works on columns of layers, and it wants the layer state, the boundary
conditions, and the optical properties of whatever absorbs and scatters. The
wrapper in
[RRTMGPInterface.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/RRTMGPInterface.jl)
builds the solver once, at cache construction, and refreshes its inputs on every
callback.

**Layer state.** Pressure and temperature come from the model's cell centers.
RRTMGP needs them on cell faces too, and reconstructs the face values itself
with the interpolation scheme the solver was built with (`BestFit()`, with
`SameAsInterpolation()` at the bottom face). Nothing in ClimaAtmos interpolates
the thermodynamic state for radiation.

**The isothermal boundary layer.** RRTMGP integrates to negligible pressure, but
the model top is at a finite height, 30 km by default and 60 km in the global
production configurations. With `add_isothermal_boundary_layer` (the default),
RRTMGP appends one extra layer above the model top, isothermal and reaching
effectively zero pressure, and fills it internally. Without it, the atmosphere
above the model top is simply absent from the calculation, and the
top-of-atmosphere fluxes are the fluxes at the model top rather than at the top
of the real atmosphere.

**Spherical geometry.** In a deep atmosphere, the area of a spherical shell
grows with height, so a flux that is uniform per unit area at the surface
carries less energy per unit area aloft. With `deep_atmosphere`, RRTMGP scales
the fluxes by ``(a/r)^2``. This requires the cell heights and the planet radius,
which the wrapper passes only for spherical geometries.

**Boundary conditions.** The surface temperature comes from
`sfc_conditions.T_sfc` (see [Surface Conditions](surface_conditions.md)); the
longwave surface emissivity and the direct and diffuse shortwave albedos are per
band. The albedo is set by the `albedo_model` key and can depend on the solar
zenith angle and the near-surface wind; see
[Ocean Surface Albedo](surface_albedo.md). The incoming solar flux is the
insolation, below. Incoming diffuse shortwave at the top of the atmosphere is
accepted by the constructor but ignored, because RRTMGP.jl does not implement
it.

## Modes

Four modes run RRTMGP, differing in what optical properties they include.

| Mode                                                                                                               | `rad`             | Gases | Clouds | Aerosols |
|:------------------------------------------------------------------------------------------------------------------ |:----------------- |:----- |:------ |:-------- |
| [`GrayRadiation`](@ref ClimaAtmos.RRTMGPInterface.GrayRadiation)                                                   | `gray`            | none  | none   | none     |
| [`ClearSkyRadiation`](@ref ClimaAtmos.RRTMGPInterface.ClearSkyRadiation)                                           | `clearsky`        | yes   | none   | optional |
| [`AllSkyRadiation`](@ref ClimaAtmos.RRTMGPInterface.AllSkyRadiation)                                               | `allsky`          | yes   | yes    | optional |
| [`AllSkyRadiationWithClearSkyDiagnostics`](@ref ClimaAtmos.RRTMGPInterface.AllSkyRadiationWithClearSkyDiagnostics) | `allskywithclear` | yes   | yes    | optional |

**Gray radiation** is a one-band model whose optical depth is a prescribed
function of pressure and latitude, from RRTMGP's
`GrayOpticalThicknessOGorman2008` following [OGorman2008](@cite). It needs no
gas, cloud, or aerosol inputs at all, only temperature and pressure, which makes
it the right choice for dynamical-core work where radiation should be present
but uncomplicated. The `lapse_rate` and `optical_thickness_parameter` arguments
the wrapper passes are validated but never reach the radiative transfer, which
uses RRTMGP's own defaults.

**Clear-sky radiation** is the full correlated-``k`` band model with gases and
no cloud optics. Clouds are absent from the radiation, not from the model: a
clear-sky run of a cloudy atmosphere is a legitimate diagnostic, and the solve
is much faster than all-sky.

**All-sky radiation** adds cloud optics. RRTMGP samples cloud overlap
stochastically (McICA), so all-sky fluxes carry sampling noise and depend on the
random number stream. `radiation_reset_rng_seed` reseeds the generator from the
timestep number before each call, which makes runs and restarts bitwise
reproducible, but it correlates the sampling noise in time. It is off by default
and should stay off for production runs.

**All-sky with clear-sky diagnostics** runs the all-sky solve and additionally
computes what the fluxes would have been with the clouds removed. The difference
is the cloud radiative effect, available through the `*cs` diagnostics
(`rsutcs`, `rlutcs`, and so on). It is the mode to use when cloud radiative
effect is one of the quantities being evaluated.

## Gases

Water vapor and ozone are given as profiles; every other gas is a single global
value by default (`use_global_means_for_well_mixed_gases`), which is what "well
mixed" means here. The full list RRTMGP accepts is `co2`, `n2o`, `co`,
`ch4`, `o2`, `n2`, `ccl4`, `cfc11`, `cfc12`, `cfc22`, `hfc143a`, `hfc125`,
`hfc23`, `hfc32`, `hfc134a`, `cf4`, and `no2`, with fixed values from
ClimaParams.

Ozone and carbon dioxide can instead vary in time, read from files and
interpolated at each radiation call. Ozone defaults to the analytic RCEMIP
profile [Wing2018](@cite) of `idealized_ozone`. See
[Trace Gases](trace_gases.md) for both, and for how to switch them on.

`idealized_h2o` replaces the model's water vapor with a prescribed
relative-humidity profile, ramped up over the first 30 days. This decouples the
radiation from the model's own moisture, which is useful for isolating
circulation responses but is not a physical configuration.

## Clouds

Cloud optics need three things per layer: how much condensate there is, how it
is distributed within the layer, and how large the particles are.

**Water paths and cloud fraction.** The condensate comes from the microphysics
model as the cloud liquid and cloud ice water contents, and the cloud fraction
from the cloud model (see
[PROPHET: Closures](prophet_closures.md#Variances-and-cloud-fraction) and
[Microphysics](microphysics.md)). RRTMGP wants *in-cloud* water paths, so
`update_cloud_properties!` converts the grid-mean contents to in-cloud values by
dividing by the cloud fraction,

```math
\mathrm{LWP} = \frac{\rho \, q_l^{cl} \, \Delta z}{\max(f_c, \epsilon)} ,
```

and likewise for ice. Cloud fraction is passed separately, so the radiation sees
a partially filled layer rather than a uniformly thin cloud.

**Effective radii.** The liquid effective radius follows the ``1/3``-power law
of [LiuHallett1997](@cite), evaluated with a droplet number concentration
diagnosed from the aerosol loading. That closure, `ml_N_cloud_liquid_droplets`,
is log-linear in the dust, sea-salt, and ammonium-sulfate concentrations and in
the liquid water. The ice effective radius is a constant from the microphysics
parameters.

!!! warning "Known inconsistency"

    `ml_N_cloud_liquid_droplets` was calibrated with a specific humidity [kg
    kg⁻¹] as its liquid-water argument, but `update_cloud_properties!` passes
    the column-integrated liquid water path [kg m⁻²]. The reference value it is
    compared against is the specific-humidity one, so the liquid-water term of
    the closure is evaluated far from where it was fitted. This is recorded in
    the docstring and tracked as a code-side task.

**Where the clouds come from.** Three sources, selected independently of the
mode:

  - [`InteractiveCloudInRadiation`](@ref ClimaAtmos.InteractiveCloudInRadiation)
    (the default): condensate and cloud fraction from the model state, refreshed
    every callback. This is the coupled configuration, in which radiation and
    clouds affect each other.
  - [`PrescribedCloudInRadiation`](@ref ClimaAtmos.PrescribedCloudInRadiation)
    (`prescribe_clouds_in_radiation: true`): cloud liquid, cloud ice, and cloud
    fraction read from an ERA5 monthly climatology, with the year 2010 repeated
    indefinitely. The model's own clouds still form and precipitate; they just
    do not radiate. This breaks the cloud-radiative feedback deliberately, which
    is what makes it useful for attributing circulation changes.
  - `idealized_clouds: true`: two fixed cloud layers, liquid between 1 and
    1.5 km and ice between 4 and 5 km, prescribed once at construction and never
    updated.

## Aerosols

With `aerosol_radiation: true`, prescribed aerosols contribute to the shortwave
optics. The wrapper passes the column mass densities of each species and, for
the dust and sea-salt bins, fixed radii. The aerosol concentrations themselves
are read from a prescribed dataset through the `prescribed_aerosols` key, which
also feeds the droplet-number closure above and, with two-moment microphysics,
aerosol activation. Enabling `aerosol_radiation` without any aerosol species in
`prescribed_aerosols` raises an error rather than silently doing nothing.

## Insolation

The incoming solar flux at the top of the atmosphere is set by the `insolation`
key, which selects one of five models. `set_insolation_variables!` writes the
cosine of the solar zenith angle and the solar irradiance into the solver on
each callback, so the insolation is refreshed on the `dt_rad` cadence like
everything else.

| Model                                                            | `insolation`       | What it gives                                                                                                                                            |
|:---------------------------------------------------------------- |:------------------ |:-------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`IdealizedInsolation`](@ref ClimaAtmos.IdealizedInsolation)     | `idealized`        | Annual-mean insolation as a function of latitude only, with no diurnal cycle [OGorman2008](@cite)                                                        |
| [`TimeVaryingInsolation`](@ref ClimaAtmos.TimeVaryingInsolation) | `timevarying`      | The full orbital calculation at the current date, from [Insolation.jl](https://clima.github.io/Insolation.jl/stable/), with a diurnal and seasonal cycle |
| [`RCEMIPIIInsolation`](@ref ClimaAtmos.RCEMIPIIInsolation)       | `rcemipii`         | Fixed uniform values from the RCEMIP-II protocol [Wing2018](@cite)                                                                                       |
| [`Larcform1Insolation`](@ref ClimaAtmos.Larcform1Insolation)     | `larcform1`        | Perpetual polar night: zero incoming solar flux                                                                                                          |
| [`ExternalTVInsolation`](@ref ClimaAtmos.ExternalTVInsolation)   | `externaldriventv` | Values from a column forcing file, time-varying or constant depending on the file                                                                        |

Removing the diurnal cycle is not a small approximation for the boundary layer,
but it removes the constraint that `dt_rad` be short enough to resolve a day,
and it is the conventional choice for aquaplanet climate runs.
`TimeVaryingInsolation` also accepts an explicit `latitude` and `longitude`,
which pins a single-column run to a real site.

## Idealized radiation for single-column cases

Three modes replace radiative transfer with a prescribed profile tuned to a
specific case. All three are applied at every stage rather than through a
callback, since each is a handful of column integrals rather than a radiative
transfer solve, and all three require moist microphysics.

[`RadiationDYCOMS`](@ref ClimaAtmos.RadiationDYCOMS) (`rad: DYCOMS`) is the
longwave parameterization of [Stevens2005](@cite) for the DYCOMS RF01 and RF02
stratocumulus cases [Ackerman2009](@cite). The net upward flux combines
cloud-top cooling, cloud-base warming, and free-tropospheric warming above the
inversion:

```math
F(z) = F_0 \, e^{-Q(z, \infty)} + F_1 \, e^{-Q(0, z)}
  + \rho_i \, c_{pd} \, D \, \alpha_z
    \left[ \tfrac{1}{4}(z - z_i)^{4/3} + z_i (z - z_i)^{1/3} \right] ,
```

with ``Q(z_1, z_2) = \int \kappa \rho q_l \, dz`` the liquid-water optical path,
``D`` the large-scale divergence, and the last term active only above the
inversion height ``z_i``, taken as the level whose ``q_t`` is closest to 0.008
kg kg⁻¹. Two departures from the reference are deliberate and documented in the
source: the optical path uses the specific liquid water content rather than the
mixing ratio, and the third term uses the dry ``c_{pd}``. Both match the
original TurbulenceConvection implementation.

[`RadiationISDAC`](@ref ClimaAtmos.RadiationISDAC) (`rad: ISDAC`) is the
two-stream liquid-water-path form used for the ISDAC mixed-phase Arctic
stratocumulus case,

```math
F(z) = F_0 \, e^{-\kappa (\mathrm{LWP}_{z_t} - \mathrm{LWP}_z)}
  + F_1 \, e^{-\kappa \, \mathrm{LWP}_z} ,
```

with ``\mathrm{LWP}_z`` the liquid water path from the surface to ``z`` and
``\mathrm{LWP}_{z_t}`` its value at the domain top. The first term is cloud-top
cooling, the second cloud-base warming.

[`RadiationTRMM_LBA`](@ref ClimaAtmos.RadiationTRMM_LBA) (`rad: TRMM_LBA`) is
the odd one out: it prescribes a heating rate directly, with no flux at all. The
observational profile varies with both height and time of day, and is converted
to an energy tendency with the moist isochoric heat capacity,
``\rho \, c_{vm} \, \partial T / \partial t``. It is used for the TRMM-LBA
deep-convection case [Grabowski2006](@cite).

## Held–Suarez forcing

`rad: held_suarez` performs no radiative transfer. It relaxes temperature toward
a prescribed radiative-equilibrium profile and applies Rayleigh friction to the
low-level winds, the standard dry dynamical-core benchmark of
[HeldSuarez1994](@cite):

```math
\frac{\partial T}{\partial t} \supset -k_T(\phi, \sigma) \, (T - T_{eq}(\phi, p)) ,
\qquad
\frac{\partial \boldsymbol{u}_h}{\partial t} \supset -k_v(\sigma) \, \boldsymbol{u}_h .
```

It occupies the `radiation_mode` slot because it replaces radiation, not because
it is radiation. Two consequences follow: it is applied from
`remaining_tendency!` at every stage rather than from a callback, so `dt_rad` is
ignored; and none of the radiation diagnostics are available. The
equator-to-pole contrast and equatorial equilibrium temperature take different
values for dry and moist microphysics (`ΔT_y_dry`/`ΔT_y_wet` and
`T_equator_dry`/`T_equator_wet`).

## Diagnostics

The flux diagnostics follow the CMIP short-name convention: `r` for radiation,
then `s` or `l` for shortwave or longwave, then `d` or `u` for downwelling or
upwelling. A bare name is the three-dimensional field, a trailing `t` is the
top-of-atmosphere value, and a trailing `s` is the surface value. A `cs` suffix
marks the clear-sky counterpart, available only in the `allskywithclear` mode.

|                               | Downwelling           | Upwelling                   |
|:----------------------------- |:--------------------- |:--------------------------- |
| Shortwave, 3D / TOA / surface | `rsd`, `rsdt`, `rsds` | `rsu`, `rsut`, `rsus`       |
| Longwave, 3D / TOA / surface  | `rld`, `rlds`         | `rlu`, `rlut`, `rlus`       |
| Clear-sky shortwave           | `rsdcs`, `rsdscs`     | `rsucs`, `rsutcs`, `rsuscs` |
| Clear-sky longwave            | `rldcs`, `rldscs`     | `rlucs`, `rlutcs`           |

Alongside the fluxes, `reffclw` and `reffcli` report the cloud liquid and ice
effective radii the radiation actually used, `od550aer` and `odsc550aer` the
aerosol optical depths, and `clt` and `cltl` the total cloud cover as seen by
the shortwave and longwave McICA sampling respectively. Those last two are not
the model's own cloud fraction: they are what the stochastic overlap sampling
produced, and the shortwave and longwave values need not agree. Requesting any
of these in a mode that does not compute them raises an error naming the
variable and the mode, rather than writing zeros. See
[Available Diagnostics](available_diagnostics.md).

## Where this is implemented

| Component                                       | Source                                                                                                                                                                                                               |
|:----------------------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mode types, solver construction, input plumbing | [radiation/RRTMGPInterface.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/RRTMGPInterface.jl)                                                                           |
| Per-callback input updates                      | [radiation/update_inputs.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/update_inputs.jl)                                                                               |
| Cache, tendencies, idealized profiles, ozone    | [radiation/radiation.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/radiation.jl)                                                                                       |
| Held–Suarez forcing                             | [radiation/held_suarez.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/parameterized_tendencies/radiation/held_suarez.jl)                                                                                   |
| The callback, insolation, and albedo updates    | [callbacks/callbacks.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/callbacks/callbacks.jl), [callbacks/get_callbacks.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/callbacks/get_callbacks.jl) |
| Configuration dispatch                          | [config/model_getters.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/config/model_getters.jl)                                                                                                              |
| Diagnostics                                     | [diagnostics/radiation_diagnostics.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/diagnostics/radiation_diagnostics.jl)                                                                                    |
